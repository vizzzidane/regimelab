"""
scripts/run_phase1.py – Run the full RegimeLab Phase 1 pipeline.

Usage (from the project root):
    python scripts/run_phase1.py

This script reproduces all CSV and PNG outputs that the notebook generates.
It does not change any methodology; it simply calls the src/regimelab modules
in order.

Outputs written to data/:
    spy_daily.csv
    regime_timeline.png
    aligned_events.csv
    event_windows.csv
    event_study_summary.csv
    cpi_event_window_heatmap.png
    fomc_event_window_heatmap.png
    cpi_cumulative_event_returns.png
    fomc_cumulative_event_returns.png
    event_day_return_boxplot.png
"""

import sys
from pathlib import Path

# Allow `import regimelab` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regimelab.data_loader  import load_or_download_spy_data
from regimelab.features     import add_regime_features, build_feature_matrix
from regimelab.clustering   import (
    compute_silhouette_scores,
    fit_regime_model,
    assign_regime_labels,
)
from regimelab.events       import load_event_dates, align_events_to_trading_days
from regimelab.event_study  import extract_event_windows, compute_event_study_summary
from regimelab.visualization import (
    plot_silhouette_scores,
    plot_regime_timeline,
    plot_event_window_heatmaps,
    plot_cumulative_event_returns,
    plot_event_day_boxplot,
)


def main() -> None:
    print("=" * 60)
    print("RegimeLab Phase 1 Pipeline")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────
    df = load_or_download_spy_data()

    # ── 2. Compute features ────────────────────────────────────────
    df = add_regime_features(df)
    features_df = build_feature_matrix(df)

    # ── 3. Silhouette analysis ─────────────────────────────────────
    scores_df = compute_silhouette_scores(features_df)
    plot_silhouette_scores(scores_df)   # displayed inline; not saved separately

    # ── 4. Fit regime model ────────────────────────────────────────
    scaler, kmeans, labels, centroids = fit_regime_model(features_df)
    df = assign_regime_labels(df, features_df, labels, centroids)

    # ── 5. Validate stress periods ─────────────────────────────────
    print()
    print("Regime validation – known stress periods:")
    stress_periods = {
        "GFC (2008-2009)":        ("2008-01-01", "2009-12-31"),
        "COVID Crash (Mar 2020)":  ("2020-03-01", "2020-03-31"),
        "2022 Bear Market":        ("2022-01-01", "2022-12-31"),
    }
    for name, (start, end) in stress_periods.items():
        mask = (df.index >= start) & (df.index <= end)
        counts = df.loc[mask, "Regime"].value_counts(normalize=True) * 100
        sc_pct = counts.get("Stressed", 0) + counts.get("Choppy", 0)
        print(f"  {name}")
        for regime, pct in counts.items():
            print(f"    {regime:10s}: {pct:.1f}%")
        print(f"    Stressed + Choppy: {sc_pct:.1f}%")

    # ── 6. Regime timeline chart ───────────────────────────────────
    plot_regime_timeline(df)

    # ── 7. Load and align event dates ─────────────────────────────
    events_df         = load_event_dates()
    aligned_events_df = align_events_to_trading_days(events_df, df)

    # ── 8. Extract event windows ───────────────────────────────────
    event_windows_df = extract_event_windows(aligned_events_df, df)

    # ── 9. Compute summary ─────────────────────────────────────────
    summary_df = compute_event_study_summary(event_windows_df)

    # ── 10. Generate event-study charts ───────────────────────────
    plot_event_window_heatmaps(event_windows_df)
    plot_cumulative_event_returns(event_windows_df)
    plot_event_day_boxplot(event_windows_df)

    # ── 11. Print key results ──────────────────────────────────────
    print()
    print("=" * 60)
    print("Event-Study Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    print("Pipeline complete.  All outputs written to data/")


if __name__ == "__main__":
    main()
