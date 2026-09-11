"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from utilities.date_functions import (
    year_frac_act_x,
    date_series,
    year_frac_30e_360,
    schedule_year_fraction,
)
from utilities.ex0_utilities import (
    get_discount_factor_by_zero_rates_linear_interp,
)

from scipy.stats import norm

from typing import Union, List, Tuple


class SwapType(Enum):
    """
    Types of swaptions.
    """

    RECEIVER = "receiver"
    PAYER = "payer"


def swaption_price_calculator(
    S0: float,
    strike: float,
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    underlying_expiry: Union[dt.date, pd.Timestamp],
    sigma_black: float,
    freq: int,
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Return the swaption price defined by the input parameters.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Value date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Underlying forward starting swap expiry.
        sigma_black (float): Swaption implied volatility.
        freq (int): Number of times a year the fixed leg pays the coupon.
        discount_factors (pd.Series): Discount factors.
        swaption_type (SwapType): Swaption type, default to receiver.

    Returns:
        Union[float, Tuple[float, float]]: Swaption price (and possibly delta).
    """

    ttm = year_frac_act_x(ref_date, expiry, 365)
    # Standard black model
    d1 = (np.log(S0 / strike) + (sigma_black**2 / 2) * ttm) / (
        sigma_black * np.sqrt(ttm)
    )
    d2 = d1 - sigma_black * np.sqrt(ttm)

    fixed_leg_payment_dates = date_series(expiry, underlying_expiry, freq)
    # Modified
    bpv = basis_point_value(fixed_leg_payment_dates, discount_factors, expiry)

    if swaption_type == SwapType.PAYER:
        price = bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))
        delta = bpv * norm.cdf(d1)

    elif swaption_type == SwapType.RECEIVER:
        price = bpv * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        delta = bpv * (norm.cdf(d1) - 1)
    else:
        raise ValueError("Invalid swaption type.")

    if compute_delta:
        return price, delta
    else:
        return price


def irs_proxy_duration(
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: pd.Series,
) -> float:
    """
    Given the specifics of an interest rate swap (IRS), return its rate sensitivity calculated as
    the duration of a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Reference date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.

    Returns:
        (float): Swap duration.
    """

    #Modified Aloïs

    year_fractions = np.array(
        [year_frac_act_x(ref_date, date, 365) for date in fixed_leg_payment_dates]
    )

    #If they coincide just gives the already existing value
    discount_factors_array = np.array(
        [
            get_discount_factor_by_zero_rates_linear_interp(
                discount_factors.index[0],
                date,
                discount_factors.index,
                discount_factors.values,
            )
            for date in fixed_leg_payment_dates
        ]
    )
    schedule = [ref_date] + list(fixed_leg_payment_dates)  # or use actual period start dates
    period_fractions = np.array([
        year_frac_act_x(schedule[i], schedule[i + 1], 365)
        for i in range(len(fixed_leg_payment_dates))
    ])

    coupons = swap_rate * period_fractions
    coupons[-1] += 1.0  # Add principal


    # Weighted average time to cash flows
    weighted_cf = coupons * discount_factors_array
    pv_cf = np.sum(weighted_cf)
    
    duration = np.sum(year_fractions * weighted_cf) / pv_cf

    return duration


def basis_point_value(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    settlement_date: Union[dt.datetime, None] = None,
) -> float:
    """
    Given a swap fixed leg payment dates and the discount factors, return the basis point value.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        settlement_date (Union[dt.datetime, None]): Settlement date, default to None, i.e. to today.
            Needed in case of forward starting swaps.

    Returns:
        float: Basis point value.
    """

    # modified (Aloïs)

    bpv = sum(
        np.array(schedule_year_fraction(fixed_leg_schedule))
        * np.array(
            [
                get_discount_factor_by_zero_rates_linear_interp(
                    discount_factors.index[0],
                    date,                         
                    discount_factors.index,
                    discount_factors.values,
                )
                for date in fixed_leg_schedule[1:]  
            ]
        )
    )
    
    return bpv


def swap_par_rate(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: Union[dt.datetime, None] = None,
) -> float:
    """
    Given a fixed leg payment schedule and the discount factors, return the swap par rate. If a
    forward start date is provided, a forward swap rate is returned.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        fwd_start_date (Union[dt.datetime, None]): Forward start date, default to None.

    Returns:
        float: Swap par rate.
    """

    ### Modified by Aloïs
    discount_factor_t0 = (
        get_discount_factor_by_zero_rates_linear_interp(
            discount_factors.index[0],
            fwd_start_date,
            discount_factors.index,
            discount_factors.values,
        )
        if fwd_start_date is not None
        else discount_factors.iloc[0]
    )

    bpv = basis_point_value(fixed_leg_schedule, discount_factors, fwd_start_date)

    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )
    float_leg = discount_factor_t0 - discount_factor_tN

    return float_leg / bpv


def swap_mtm(
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Given a swap rate, a fixed leg payment schedule and the discount factors, return the swap
    mark-to-market.

    Parameters:
        swap_rate (float): Swap rate.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        swap_type (SwapType): Swap type, either 'payer' or 'receiver', default to 'payer'.

    Returns:
        float: Swap mark-to-market.
    """

    #Modified (Aloïs)

    # Single curve framework, returns price and basis point value
    bpv = basis_point_value(fixed_leg_schedule, discount_factors)
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )
    float_leg = 1.0 - P_term
    fixed_leg = swap_rate * bpv

    if swap_type == SwapType.RECEIVER:
        multiplier = 1
    elif swap_type == SwapType.PAYER:
        multiplier = -1
    else:
        raise ValueError("Unknown swap type.")

    return multiplier * (float_leg - fixed_leg)
