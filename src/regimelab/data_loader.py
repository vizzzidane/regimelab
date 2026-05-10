"""
data_loader.py – Download or load cached SPY daily OHLCV data.

The adjusted Close price is used as the basis for all feature calculations.
Data is cached to data/spy_daily.csv to avoid repeated API calls.
"""

import pandas as pd
import yfinance as yf

from regimelab.config import DATA_DIR, TICKER, START_DATE

CACHE_FILE = DATA_DIR / "spy_daily.csv"


def load_or_download_spy_data() -> pd.DataFrame:
    """
    Return a clean SPY daily DataFrame indexed by date.

    If data/spy_daily.csv already exists it is loaded from disk.
    Otherwise SPY data is downloaded from Yahoo Finance (auto_adjust=True),
    cleaned, and saved to the cache file.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV data with a timezone-naive DatetimeIndex.
    """
    if CACHE_FILE.exists():
        print(f"[data_loader] Loading from cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, index_col="Date", parse_dates=True)
    else:
        print(f"[data_loader] Downloading {TICKER} from yfinance (start={START_DATE}) …")
        ticker = yf.Ticker(TICKER)
        df = ticker.history(start=START_DATE, auto_adjust=True)
        # Remove timezone info so the index is plain dates
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(CACHE_FILE)
        print(f"[data_loader] Cached to: {CACHE_FILE}")

    df = _clean(df)
    print(f"[data_loader] Loaded {len(df)} rows  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return df


# ── Private helpers ────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate dates and rows with NaN Close prices."""
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Close"])
    n_removed = n_before - len(df)
    if n_removed:
        print(f"[data_loader] Removed {n_removed} duplicate/NaN rows.")
    return df
