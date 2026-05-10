"""
features.py – Compute regime features from SPY daily price data.

All features are strictly trailing (no lookahead bias in construction):
  - Log_Return      : daily log return
  - Vol_20d         : 20-day annualised realised volatility
  - Mom_20d         : 20-day cumulative log return (momentum)
  - Drawdown_60d    : (Close − 60-day rolling max) / 60-day rolling max

Rolling windows use min_periods equal to the window size so that rows
with insufficient history are returned as NaN and dropped downstream.

Lookahead-bias note
-------------------
The features themselves are constructed without lookahead.  However, the
K-Means model in clustering.py is fitted on the full dataset, so the
resulting regime labels are exploratory and not walk-forward.
"""

import numpy as np
import pandas as pd

from regimelab.config import (
    VOL_WINDOW,
    MOMENTUM_WINDOW,
    DRAWDOWN_WINDOW,
)

FEATURE_COLS = ["Vol_20d", "Mom_20d", "Drawdown_60d"]


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime feature columns to *df* in-place and return it.

    Parameters
    ----------
    df : pd.DataFrame
        SPY daily data with at least a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with four new columns added.
    """
    df = df.copy()

    # 1. Daily log return
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2. 20-day annualised realised volatility
    df["Vol_20d"] = (
        df["Log_Return"]
        .rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW)
        .std() * np.sqrt(252)
    )

    # 3. 20-day momentum (cumulative log return)
    df["Mom_20d"] = (
        df["Log_Return"]
        .rolling(window=MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW)
        .sum()
    )

    # 4. 60-day rolling drawdown
    roll_max = df["Close"].rolling(window=DRAWDOWN_WINDOW, min_periods=DRAWDOWN_WINDOW).max()
    df["Drawdown_60d"] = (df["Close"] - roll_max) / roll_max

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the three clustering features and drop NaN warmup rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has already been processed by add_regime_features().

    Returns
    -------
    pd.DataFrame
        A clean feature DataFrame with no NaNs, indexed by date.
    """
    features_df = df[FEATURE_COLS].dropna().copy()
    print(f"[features] Feature matrix: {features_df.shape[0]} rows × {features_df.shape[1]} cols "
          f"(dropped {len(df) - len(features_df)} NaN warmup rows)")
    return features_df
