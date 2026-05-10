"""
event_study.py – Extract event windows and compute summary statistics.

Windows use positional (trading-day) indexing, not calendar-day offsets.
The main grouping variable is pre_event_regime (regime on Day -1).

Limitations
-----------
- The Stressed regime has very few events (≤2 for CPI, 1 for FOMC).
  Do not draw strong conclusions from those buckets.
- K-Means was fitted on the full dataset; these results are exploratory,
  not a walk-forward backtest.
- No multiple-testing correction is applied.
"""

import numpy as np
import pandas as pd

from regimelab.config import DATA_DIR, EVENT_WINDOW

EVENT_WINDOWS_FILE   = DATA_DIR / "event_windows.csv"
SUMMARY_FILE         = DATA_DIR / "event_study_summary.csv"


def extract_event_windows(
    aligned_events_df: pd.DataFrame,
    df: pd.DataFrame,
    window: int = EVENT_WINDOW,
) -> pd.DataFrame:
    """
    Extract [-window, +window] trading-day return windows for each event.

    Parameters
    ----------
    aligned_events_df : pd.DataFrame
        Output of align_events_to_trading_days().
    df : pd.DataFrame
        Main SPY DataFrame with 'Log_Return' column.
    window : int
        Number of trading days on each side of the event.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with one row per (event, relative_day).
    """
    trading_dates = df.index
    records = []

    for _, row in aligned_events_df.iterrows():
        trade_date = pd.Timestamp(row["trading_event_date"])
        trade_idx  = trading_dates.get_loc(trade_date)

        # Skip if window extends beyond available data
        if trade_idx - window < 0 or trade_idx + window >= len(trading_dates):
            continue

        for i in range(-window, window + 1):
            curr_date = trading_dates[trade_idx + i]
            log_ret   = df.loc[curr_date, "Log_Return"]
            simple_ret = np.exp(log_ret) - 1

            records.append({
                "event_type":          row["event_type"],
                "original_event_date": row["original_event_date"],
                "trading_event_date":  row["trading_event_date"],
                "relative_day":        i,
                "log_return":          log_ret,
                "simple_return":       simple_ret,
                "pre_event_regime":    row["pre_event_regime"],
                "event_day_regime":    row["event_day_regime"],
            })

    windows_df = pd.DataFrame(records)
    windows_df.to_csv(EVENT_WINDOWS_FILE, index=False)
    print(f"[event_study] Extracted {len(windows_df)} records → {EVENT_WINDOWS_FILE}")
    return windows_df


def compute_event_study_summary(event_windows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(event_type, pre_event_regime) summary statistics.

    Metrics
    -------
    N                          : number of events in this bucket
    Mean Day 0 Ret (%)         : average event-day simple return
    Std Day 0 Ret (%)          : std-dev of event-day simple return
    Mean Cum Ret [-5,+5] (%)   : average cumulative simple return over the window
    Hit Rate (%)               : % of events with positive Day 0 return
    Day 0 Sharpe               : Mean Day 0 Ret / Std Day 0 Ret (simple ratio)

    Results are saved to data/event_study_summary.csv.

    Parameters
    ----------
    event_windows_df : pd.DataFrame
        Output of extract_event_windows().

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    records = []

    for (ev_type, regime), group in event_windows_df.groupby(["event_type", "pre_event_regime"]):
        day0 = group[group["relative_day"] == 0]
        n    = len(day0)
        if n == 0:
            continue

        mean_ret = day0["simple_return"].mean()
        std_ret  = day0["simple_return"].std()
        hit_rate = (day0["simple_return"] > 0).mean() * 100
        sharpe   = mean_ret / std_ret if std_ret and std_ret > 0 else np.nan

        # Average cumulative return over the full window per event
        cum_rets = []
        for _, ev_group in group.groupby("original_event_date"):
            ev_group = ev_group.sort_values("relative_day")
            cum_rets.append(np.exp(ev_group["log_return"].sum()) - 1)
        mean_cum = np.mean(cum_rets) * 100

        records.append({
            "Event Type":              ev_type,
            "Pre-Event Regime":        regime,
            "N":                       n,
            "Mean Day 0 Ret (%)":      round(mean_ret * 100, 2),
            "Std Day 0 Ret (%)":       round(std_ret  * 100, 2),
            "Mean Cum Ret [-5,+5] (%)": round(mean_cum, 2),
            "Hit Rate (%)":            round(hit_rate, 2),
            "Day 0 Sharpe":            round(sharpe, 2) if not np.isnan(sharpe) else np.nan,
        })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"[event_study] Summary saved → {SUMMARY_FILE}")
    return summary_df
