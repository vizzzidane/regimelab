"""
events.py – Load CPI/FOMC event dates and align them to SPY trading days.

The pre_event_regime (regime on the trading day before the event) is used
as the primary grouping variable throughout the event study.  This reduces
lookahead bias because the event-day return itself does not influence the
regime label used for grouping.
"""

import sys
import pandas as pd

from regimelab.config import DATA_DIR, EVENTS_DIR

ALIGNED_EVENTS_FILE = DATA_DIR / "aligned_events.csv"


def load_event_dates() -> pd.DataFrame:
    """
    Load CPI and FOMC event dates from events/event_dates.py.

    Returns
    -------
    pd.DataFrame
        Columns: event_type (str), original_event_date (datetime).
    """
    # Temporarily add events/ to sys.path so we can import event_dates
    events_dir_str = str(EVENTS_DIR)
    if events_dir_str not in sys.path:
        sys.path.insert(0, events_dir_str)

    from event_dates import CPI_DATES, FOMC_DATES  # type: ignore

    cpi_rows  = [{"event_type": "CPI",  "original_event_date": pd.Timestamp(d)} for d in CPI_DATES]
    fomc_rows = [{"event_type": "FOMC", "original_event_date": pd.Timestamp(d)} for d in FOMC_DATES]

    events_df = pd.DataFrame(cpi_rows + fomc_rows).sort_values("original_event_date").reset_index(drop=True)
    print(f"[events] Loaded {len(events_df)} events  "
          f"(CPI={len(cpi_rows)}, FOMC={len(fomc_rows)})")
    return events_df


def align_events_to_trading_days(
    events_df: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map each event date to the nearest valid SPY trading date on or after it.

    For each aligned event we record:
      - trading_event_date   : nearest trading day ≥ original_event_date
      - pre_event_trading_date : trading day immediately before trading_event_date
      - pre_event_regime     : regime on pre_event_trading_date  (grouping variable)
      - event_day_regime     : regime on trading_event_date

    Events outside the available SPY date range are silently skipped.
    Results are saved to data/aligned_events.csv.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of load_event_dates().
    df : pd.DataFrame
        Main SPY DataFrame with a 'Regime' column.

    Returns
    -------
    pd.DataFrame
        Aligned events with the columns listed above.
    """
    trading_dates = df.index
    records = []

    for _, row in events_df.iterrows():
        orig_date = row["original_event_date"]

        # Nearest trading day on or after the event date
        valid = trading_dates[trading_dates >= orig_date]
        if len(valid) == 0:
            continue  # past the end of available data

        trade_date = valid[0]
        trade_idx  = trading_dates.get_loc(trade_date)

        if trade_idx < 1:
            continue  # need at least one day before for pre-event regime

        pre_date = trading_dates[trade_idx - 1]

        records.append({
            "event_type":            row["event_type"],
            "original_event_date":   orig_date.date(),
            "trading_event_date":    trade_date.date(),
            "pre_event_trading_date": pre_date.date(),
            "pre_event_regime":      df.loc[pre_date,  "Regime"],
            "event_day_regime":      df.loc[trade_date, "Regime"],
        })

    aligned_df = pd.DataFrame(records)
    aligned_df.to_csv(ALIGNED_EVENTS_FILE, index=False)
    print(f"[events] Aligned {len(aligned_df)} events → {ALIGNED_EVENTS_FILE}")
    return aligned_df
