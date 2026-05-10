"""
tests/test_basic_pipeline.py – Basic smoke tests for the RegimeLab pipeline.

These tests verify that the core data and transformation steps work correctly.
They are intentionally simple and do not test statistical properties.

Run from the project root:
    pytest tests/test_basic_pipeline.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow `import regimelab` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regimelab.data_loader  import load_or_download_spy_data
from regimelab.features     import add_regime_features, build_feature_matrix, FEATURE_COLS
from regimelab.clustering   import fit_regime_model, assign_regime_labels
from regimelab.events       import load_event_dates


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spy_df():
    """Load (or download) SPY data once for the whole test module."""
    return load_or_download_spy_data()


@pytest.fixture(scope="module")
def spy_with_features(spy_df):
    return add_regime_features(spy_df)


@pytest.fixture(scope="module")
def feature_matrix(spy_with_features):
    return build_feature_matrix(spy_with_features)


@pytest.fixture(scope="module")
def regime_df(spy_with_features, feature_matrix):
    _, _, labels, centroids = fit_regime_model(feature_matrix)
    return assign_regime_labels(spy_with_features, feature_matrix, labels, centroids)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_spy_data_is_non_empty(self, spy_df):
        assert len(spy_df) > 0, "SPY DataFrame should not be empty"

    def test_spy_has_close_column(self, spy_df):
        assert "Close" in spy_df.columns, "SPY DataFrame must have a 'Close' column"

    def test_spy_index_is_datetime(self, spy_df):
        assert isinstance(spy_df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

    def test_spy_no_duplicate_dates(self, spy_df):
        assert not spy_df.index.duplicated().any(), "No duplicate dates after cleaning"

    def test_spy_no_nan_close(self, spy_df):
        assert spy_df["Close"].isna().sum() == 0, "No NaN Close values after cleaning"

    def test_spy_starts_from_2005(self, spy_df):
        assert spy_df.index.min().year <= 2005, "Data should start from 2005 or earlier"


class TestFeatures:
    def test_feature_columns_exist(self, spy_with_features):
        for col in ["Log_Return"] + FEATURE_COLS:
            assert col in spy_with_features.columns, f"Column '{col}' missing"

    def test_feature_matrix_no_nans(self, feature_matrix):
        assert feature_matrix.isna().sum().sum() == 0, "Feature matrix must have no NaNs"

    def test_feature_matrix_has_expected_columns(self, feature_matrix):
        assert list(feature_matrix.columns) == FEATURE_COLS

    def test_feature_matrix_smaller_than_raw(self, spy_df, feature_matrix):
        # Warmup rows should have been dropped
        assert len(feature_matrix) < len(spy_df)

    def test_vol_is_positive(self, feature_matrix):
        assert (feature_matrix["Vol_20d"] > 0).all(), "Volatility should always be positive"

    def test_drawdown_is_non_positive(self, feature_matrix):
        assert (feature_matrix["Drawdown_60d"] <= 0).all(), "Drawdown should be ≤ 0"


class TestClustering:
    def test_regime_column_exists(self, regime_df):
        assert "Regime" in regime_df.columns

    def test_regime_labels_are_valid(self, regime_df):
        valid = {"Calm", "Choppy", "Stressed", np.nan}
        actual = set(regime_df["Regime"].unique())
        # NaN is allowed for warmup rows
        assert actual <= {"Calm", "Choppy", "Stressed"} | {np.nan}

    def test_all_three_regimes_present(self, regime_df):
        regimes = set(regime_df["Regime"].dropna().unique())
        assert regimes == {"Calm", "Choppy", "Stressed"}


class TestEventDates:
    def test_event_dates_load(self):
        events_df = load_event_dates()
        assert len(events_df) > 0, "Event dates should not be empty"

    def test_event_types_are_cpi_and_fomc(self):
        events_df = load_event_dates()
        assert set(events_df["event_type"].unique()) == {"CPI", "FOMC"}

    def test_event_dates_are_timestamps(self):
        events_df = load_event_dates()
        assert pd.api.types.is_datetime64_any_dtype(events_df["original_event_date"])

    def test_cpi_count_is_reasonable(self):
        events_df = load_event_dates()
        n_cpi = (events_df["event_type"] == "CPI").sum()
        # 2010-2024 = 15 years × 12 months = 180 expected
        assert 170 <= n_cpi <= 190, f"Expected ~180 CPI dates, got {n_cpi}"

    def test_fomc_count_is_reasonable(self):
        events_df = load_event_dates()
        n_fomc = (events_df["event_type"] == "FOMC").sum()
        # 2010-2024 = 15 years × ~8 meetings = ~120 expected
        assert 110 <= n_fomc <= 135, f"Expected ~120 FOMC dates, got {n_fomc}"


class TestEventWindows:
    def test_relative_days_range(self, spy_df):
        """Event windows should contain exactly relative days -5 … +5."""
        from regimelab.events     import load_event_dates, align_events_to_trading_days
        from regimelab.event_study import extract_event_windows

        df = add_regime_features(spy_df)
        features_df = build_feature_matrix(df)
        _, _, labels, centroids = fit_regime_model(features_df)
        df = assign_regime_labels(df, features_df, labels, centroids)

        events_df    = load_event_dates()
        aligned_df   = align_events_to_trading_days(events_df, df)
        windows_df   = extract_event_windows(aligned_df, df)

        expected_days = set(range(-5, 6))
        actual_days   = set(windows_df["relative_day"].unique())
        assert actual_days == expected_days, (
            f"Expected relative days {expected_days}, got {actual_days}"
        )

    def test_window_has_no_nan_returns(self, spy_df):
        """No NaN log_return values in the extracted windows."""
        from regimelab.events     import load_event_dates, align_events_to_trading_days
        from regimelab.event_study import extract_event_windows

        df = add_regime_features(spy_df)
        features_df = build_feature_matrix(df)
        _, _, labels, centroids = fit_regime_model(features_df)
        df = assign_regime_labels(df, features_df, labels, centroids)

        events_df  = load_event_dates()
        aligned_df = align_events_to_trading_days(events_df, df)
        windows_df = extract_event_windows(aligned_df, df)

        assert windows_df["log_return"].isna().sum() == 0
