
import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, List
from scipy.integrate import quad    # compute numerical integration

from utilities.ex1_utilities import (
    get_discount_factor_by_zero_rates_linear_interp,
    year_frac_act_x,
)

def simulate_ou_process(
    simulation_grid: pd.DatetimeIndex,
    today: pd.Timestamp,
    mean_reversion: float,
    sigma: float,
    n_sim: int,
    x0=0.0
) -> np.ndarray :
    """
    Simulate the Ornstein–Uhlenbeck (Hull–White) state process x_t.

    Parameters
    ----------
    simulation_grid : pd.DatetimeIndex
        Simulation dates
    today : pd.Timestamp
        Initial date
    mean_reversion : float
        Mean reversion parameter a
    sigma : float
        Volatility parameter
    n_sim : int
        Number of Monte Carlo scenarios
    x0 : float, default 0.0
        Initial value of the process

    Returns
    -------
    np.ndarray
        Simulated paths of x_t, shape (n_sim, n_dates)
    """

    # !!! COMPLETE AS APPROPRIATE !!! 
    
    n_steps = len(simulation_grid)

    # Generate standard normal variables
    Z = np.random.standard_normal(size = (n_sim , n_steps))

    # Allocate paths
    x_paths = np.zeros((n_sim, n_steps))
    x_prev = np.full(n_sim, x0)
    prev_date = today

    # Time stepping
    for i, sim_date in enumerate(simulation_grid):

        dt = year_frac_act_x(prev_date, sim_date, 360)
        
        
        u_hat = 1 - np.exp(-mean_reversion * dt)
        sigma_hat = sigma * np.sqrt((1 - np.exp(-2 * mean_reversion * dt)) / (2 * mean_reversion))
        
        x_current = x_prev - u_hat * x_prev + sigma_hat * Z[:, i]
        
        x_paths[:, i] = x_current
        x_prev = x_current
        

        prev_date = sim_date

    return x_paths


def affine_trick(
    valuation_date: dt.datetime,
    pricing_grid: List[dt.datetime],
    mean_reversion: float,
    sigma: float,
    discount_factors: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Affine trick: Exploits the affine structure of the Hull-White model. Output the functions
    A(t, t_j) and C(t, t_j) pre-computed on the pricing grid s.t.
    B(t, t_j) = A(t, t_j) * exp(-C(t, t_j) * x(t)).

    Parameters:
        valuation date = t (dt.datetime): Date w.r.t. which the computations are made.
        pricing_grid (List[dt.datetime]): Pricing grid.
        mean_reversion (float): Hull-white mean reversion speed.
        sigma (float): Hull-white interest rate volatility.
        discount_factors (pd.Series): Discount factors curve.

    Returns:
        Tuple[pd.Series, pd.Series]: Tuple with the precomputed functions A(t, t_j) and C(t, t_j).
    """
    
    # Reference date and containers
    reference_date = discount_factors.index[0].to_pydatetime()

    A = pd.Series(index=pricing_grid, dtype=float)
    C = pd.Series(index=pricing_grid, dtype=float)

    # Ensure discount factors exist on the whole pricing grid
    missing_dates = [d for d in pricing_grid if d not in discount_factors.index]
   
    if missing_dates:
        interpolated_array = np.array(
            [
            get_discount_factor_by_zero_rates_linear_interp(
                discount_factors.index[0],
                date,
                discount_factors.index,
                discount_factors.values,
            )
            for date in missing_dates
            ]
        )
        
        interpolated_dfs = pd.Series(interpolated_array, index=missing_dates)
        
        discount_factors = (
            pd.concat([discount_factors, interpolated_dfs])
            .sort_index()
        )

    # Volatility integrand # fix tau missing
    sigma_func = lambda u, tau: (sigma / mean_reversion) * (1 - np.exp(-mean_reversion * (tau - u)))

    for T in pricing_grid:

        if T <= valuation_date:
            A[T] = 1
            C[T] = 0
            continue

        tau_t_T = year_frac_act_x(valuation_date, T, 360)
        tau_0_T = year_frac_act_x(reference_date, T, 360)
        tau_0_t = year_frac_act_x(reference_date, valuation_date, 360)

        sigma_integrand = lambda u: sigma_func(u, tau_0_T) ** 2 - sigma_func(u, tau_0_t) ** 2


        # Numerical integration with scipy quad
        integral_val, _ = quad(sigma_integrand, 0.0, tau_0_t)

        A[T] = discount_factors[T] / discount_factors[valuation_date] * np.exp(
        -0.5 * integral_val
        )

        C[T] = (1 - np.exp(-mean_reversion * tau_t_T)) / mean_reversion

    return A, C


    
def update_collateral(
    mtm: np.ndarray,
    VM: np.ndarray,
    THR_B: float = 0.0,
    THR_C: float = 0.0,
    MTA_B: float = 0.0,
    MTA_C: float = 0.0,
    Cap_B: float = np.inf,
    Cap_C: float = np.inf,
) -> np.ndarray:
    """
    Update the Variation Margin (VM) at a single time step under a bilateral CSA,
    from the perspective of party B, for multiple Monte Carlo scenarios.

    The function applies:
    - computation of target variation margin VM*,
    - margin calls subject to minimum transfer amounts (MTA),
    - application of collateral caps.

    Independent amount (IA), haircuts, rounding rules, settlement lags,
    collateral revaluation and remuneration effects are ignored here.

    Parameters
    ----------
    mtm : np.ndarray
        Array of mark-to-market values for each Monte Carlo scenario,
        measured from the perspective of party B.
        Shape: (n_scenarios,)

    VM : np.ndarray
        Array of variation margin currently posted for each scenario.
        Shape: (n_scenarios,)

    THR_B : float, default 0.0
        Threshold granted to party B (applies when MtM < 0).

    THR_C : float, default 0.0
        Threshold granted to counterparty C (applies when MtM > 0).

    MTA_B : float, default 0.0
        Minimum Transfer Amount when B is required to post collateral.

    MTA_C : float, default 0.0
        Minimum Transfer Amount when C is required to post collateral.

    Cap_B : float, default +inf
        Maximum collateral that B can be required to post.

    Cap_C : float, default +inf
        Maximum collateral that C can be required to post.

    Returns
    -------
    np.ndarray
        Updated variation margin for each scenario.
        Shape: (n_scenarios,)

    Notes
    -----
    Target variation margin is defined scenario-wise as:

        VM* =
            max(MtM - THR_C, 0)    if MtM >= 0
            min(MtM + THR_B, 0)    if MtM < 0

    Margin calls are applied only when the required change exceeds
    the relevant MTA.
    """

    VM_star = np.where(
        mtm >= 0,
        np.maximum(mtm - THR_C, 0),
        np.minimum(mtm + THR_B, 0),
    )

    delta_VM = VM_star - VM

    margin_call = np.where(
        (np.abs(delta_VM) >= MTA_C) & (delta_VM >= 0), delta_VM,
        np.where(
            (np.abs(delta_VM) >= MTA_B) & (delta_VM < 0), delta_VM,
            0.0,
        ),
    )

    VM_no_cap = VM + margin_call

    VM_updated = np.where(
        VM_no_cap >= 0,
        np.minimum(VM_no_cap, Cap_C),
        np.maximum(VM_no_cap, -Cap_B),
    )

    return VM_updated