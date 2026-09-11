"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 0: Discount Factors Bootstrap
"""

import numpy as np
import pandas as pd
import datetime as dt
from utilities.date_functions import (
    business_date_offset,
    year_frac_act_x,
    year_frac_30e_360,
)
from typing import Iterable, Union, List, Union, Tuple


def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> List[float]:
    """
    Compute the zero rates from the discount factors.

    Parameters:
        dates (Union[List[float], pd.DatetimeIndex]): List of year fractions or dates.
            If a DatetimeIndex, the first element is assumed to be the settlement date
            (with B=1) and is skipped; year fractions are computed as ACT/360 from it.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        List[float]: List of continuously-compounded zero rates (ACT/360).
    """

    effDates, effDf = dates, discount_factors
    if isinstance(effDates, pd.DatetimeIndex):
        # The first entry is the settlement date (B=1), skip it.
        settlement = effDates[0]
        effDates = np.array([year_frac_act_x(settlement, d, 360) for d in effDates[1:]])
        effDf = np.array(discount_factors[1:])

    # Zero rate (continuous, ACT/360): z = -ln(B) / yf
    zero_rates = list(-np.log(np.array(effDf)) / np.array(effDates))
    return zero_rates


def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    """
    Given a list of discount factors, return the discount factor at a given date by linear
    interpolation on zero rates.

    Parameters:
        reference_date (Union[dt.datetime, pd.Timestamp]): Reference date.
        interp_date (Union[dt.datetime, pd.Timestamp]): Date at which the discount factor is
            interpolated.
        dates (Union[List[dt.datetime], pd.DatetimeIndex]): List of dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        float: Discount factor at the interpolated date.
    """

    if len(dates) != len(discount_factors):
        raise ValueError(
            f"Length mismatch: {len(dates)} dates vs {len(discount_factors)} discounts."
        )

    # Compute ACT/360 year fractions from the reference date for the available pillars
    yfs = np.array([year_frac_act_x(reference_date, d, 360) for d in dates])
    dfs = np.array(discount_factors, dtype=float)

    # Convert to continuously-compounded zero rates; guard against yf=0 (reference date, B=1)
    zero_rates = np.where(yfs > 0.0, -np.log(dfs) / yfs, 0.0)

    # Target year fraction
    target_yf = year_frac_act_x(reference_date, interp_date, 360)

    # Linear interpolation of the zero rate
    interp_zero = float(np.interp(target_yf, yfs, zero_rates))

    # Convert back to discount factor
    discount = np.exp(-interp_zero * target_yf)
    return float(discount)


def bootstrap(
    reference_date: dt.datetime,
    depo: pd.DataFrame,
    futures: pd.DataFrame,
    swaps: pd.DataFrame,
    shock: float = 0.0,
) -> pd.Series:
    """
    Bootstrap the discount factors from the given bid/ask market data. Deposit rates are used until
    the first future settlement date (included), futures rates are used until the 2y-swap settlement.

    Parameters:
        reference_date (dt.datetime): Reference date (settlement date, T+2 from today).
        depo (pd.DataFrame): Deposit rates dataframe. Index is parsed from the 'Depos' column
            (datetime), columns 'BID' and 'ASK' already in decimal form (e.g. 0.0399).
        futures (pd.DataFrame): Futures dataframe. Index is the IMM date (datetime),
            columns 'BID', 'ASK' (prices, e.g. 95.68), 'Settle' and 'Expiry' (datetime).
        swaps (pd.DataFrame): Swap rates dataframe. Index is parsed from the 'Swaps' column
            (datetime), columns 'BID' and 'ASK' in percentage (e.g. 4.12).
        shock (Union[float, pd.Series]): Parallel shift to apply to the market rates, default to
            zero.

    Returns:
        Tuple[pd.Series, pd.Series]: (discount_factors, zero_rates)
    """

    # Initialize: settlement date has a discount of 1
    termDates, discounts = [reference_date], [1.0]

    # ------------------------------------------------------------------
    # DEPOS
    # ------------------------------------------------------------------
    # The depo index is datetime (the expiry of each deposit).
    # Rates are already in decimal. Use all depos whose expiry is on or
    # before the settle date of the first future.

    first_future_settle = futures["Settle"].iloc[0]

    depoDates = [d for d in depo.index if d <= first_future_settle]
    depoRates = depo.loc[depoDates].mean(axis=1).values  # already in decimal

    # Apply parallel shock (shock in decimal form as well)
    shock_depo = shock if isinstance(shock, float) else shock[depoDates].values
    depoRates = depoRates + shock_depo

    # B(t0, ti) = 1 / (1 + L * yf_act360)
    for date, rate in zip(depoDates, depoRates):
        yf = year_frac_act_x(reference_date, date, 360)
        df = 1.0 / (1.0 + rate * yf)
        termDates.append(date)
        discounts.append(df)

    # ------------------------------------------------------------------
    # FUTURES
    # ------------------------------------------------------------------
    # Use the first 7 STIR futures since they're considered to be be liquid enough.
    fut_of_interest = futures.iloc[0:7].copy()

    fut_prices = fut_of_interest[["BID", "ASK"]].mean(axis=1).values  # prices
    futFwdRates = (100.0 - fut_prices) / 100.0  # fwd rates (decimal)

    # Apply shock (shock is in rate not price space; sign: +shock → +rate → -price)
    shock_fut = (
        shock if isinstance(shock, float) else shock[fut_of_interest.index].values
    )
    futFwdRates = futFwdRates + shock_fut

    for i in range(len(fut_of_interest)):
        row = fut_of_interest.iloc[i]
        fut_start = row["Settle"]  # start date of the 3m period
        fut_end = row["Expiry"]  # end date of the 3m period

        fwd_rate = futFwdRates[i]
        yf = year_frac_act_x(fut_start, fut_end, 360)

        # Forward discount: B(t0; fut_start, fut_end) = 1 / (1 + L_fwd * yf)
        fwd_df = 1.0 / (1.0 + fwd_rate * yf)

        # Spot discount at fut_start by interpolating the current curve
        spot_df_start = get_discount_factor_by_zero_rates_linear_interp(
            reference_date, fut_start, termDates, discounts
        )

        # Compound: B(t0, fut_end) = B(t0, fut_start) * B(t0; fut_start, fut_end)
        spot_df_end = spot_df_start * fwd_df

        termDates.append(fut_end)
        discounts.append(spot_df_end)

    # ------------------------------------------------------------------
    # SWAPS
    # ------------------------------------------------------------------
    # Annual fixed coupon, 30E/360 day count on the fixed leg.
    # Bootstrap formula for the N-th pillar:
    #   B(t0, tN) = (1 - R * sum_{i=1}^{N-1} yf_i * B(t0, ti)) / (1 + R * yf_N)
    # Swap rates in the CSV are in % (e.g. 4.12) → divide by 100.
    # We only use swaps that extend BEYOND the last future we used.
    last_future_end = fut_of_interest["Expiry"].iloc[-1]
    swaps_of_interest = swaps[swaps.index > last_future_end]

    swapRates = swaps_of_interest.mean(axis=1).values / 100.0 + (
        shock if isinstance(shock, float) else shock[swaps_of_interest.index].values
    )

    swap_old = reference_date  # rolling previous coupon date
    bpv_sum = 0.0

    for idx, swapDate in enumerate(swaps_of_interest.index):
        rate = swapRates[idx]

        # Annual coupon dates strictly between the previous swap pillar and this one.
        # (Only relevant for pillars that jump >1 year from the previous quoted pillar,
        #  which never happens here since all consecutive swap pillars are 1y apart.)
        # Look-up intermediate annual coupon dates between the previous swap pillar and this one.
        prev_swap_date = (
            swaps_of_interest.index[idx - 1]
            if idx > 0
            else (
                swaps.index[swaps.index < swapDate][-1]
                if len(swaps.index[swaps.index < swapDate]) > 0
                else reference_date
            )
        )

        intermediate_dates = []
        test_date = business_date_offset(prev_swap_date, year_offset=1)
        while test_date < swapDate:
            intermediate_dates.append(test_date)
            test_date = business_date_offset(test_date, year_offset=1)

        # Add BPV contributions from any intermediate coupon dates
        for idate in intermediate_dates:
            yf_i = year_frac_30e_360(swap_old, idate)
            df_i = get_discount_factor_by_zero_rates_linear_interp(
                reference_date, idate, termDates, discounts
            )
            bpv_sum += yf_i * df_i
            swap_old = idate

        # Bootstrap the discount at this swap's maturity
        yf_n = year_frac_30e_360(swap_old, swapDate)
        df_n = (1.0 - rate * bpv_sum) / (1.0 + rate * yf_n)

        bpv_sum += yf_n * df_n
        swap_old = swapDate

        termDates.append(swapDate)
        discounts.append(df_n)

    discount_factors = pd.Series(index=termDates, data=discounts).sort_index()
    zero = from_discount_factors_to_zero_rates(
        discount_factors.index, discount_factors.values
    )
    zero_rates = pd.Series(index=termDates[1:], data=zero)

    return discount_factors, zero_rates
