"""
visualization.py – Chart generation for RegimeLab Phase 1.

All functions save PNG files to data/ and also return the Figure object
so callers can display or further customise them.

Charts use the same colour palette and style as the original notebook.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from regimelab.config import DATA_DIR, REGIME_COLORS

plt.style.use("ggplot")


# ── Silhouette scores ──────────────────────────────────────────────────────

def plot_silhouette_scores(scores_df: pd.DataFrame) -> plt.Figure:
    """
    Line chart of silhouette scores for each k.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of compute_silhouette_scores() – columns: k, silhouette_score.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scores_df["k"], scores_df["silhouette_score"], marker="o", linewidth=2)
    ax.set_title("Silhouette Scores for K-Means (k = 2 to 5)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(scores_df["k"])
    fig.tight_layout()
    return fig


# ── Regime timeline ────────────────────────────────────────────────────────

def plot_regime_timeline(df: pd.DataFrame) -> plt.Figure:
    """
    SPY price (log scale) with regime-coloured background shading.

    Saves to data/regime_timeline.png.
    """
    out = DATA_DIR / "regime_timeline.png"

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df.index, df["Close"], color="black", linewidth=0.8, zorder=5, label="SPY Close")

    price_min = df["Close"].min() * 0.95
    price_max = df["Close"].max() * 1.05

    for regime, color in REGIME_COLORS.items():
        mask = df["Regime"] == regime
        ax.fill_between(df.index, price_min, price_max,
                        where=mask, color=color, alpha=0.25, linewidth=0)

    price_line = plt.Line2D([0], [0], color="black", linewidth=1, label="SPY Close")
    patches = [mpatches.Patch(color=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    ax.legend(handles=[price_line] + patches, loc="upper left", fontsize=10)

    ax.set_title("SPY Market Regimes (K-Means, k=3)", fontsize=14)
    ax.set_ylabel("Price (log scale)")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.set_xlim(df.index.min(), df.index.max())
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved {out}")
    return fig


# ── Event-window heatmaps ──────────────────────────────────────────────────

def plot_event_window_heatmaps(event_windows_df: pd.DataFrame) -> dict[str, plt.Figure]:
    """
    One heatmap per event type: mean simple return (%) by relative day × regime.

    Saves to data/cpi_event_window_heatmap.png and
              data/fomc_event_window_heatmap.png.

    Returns
    -------
    dict mapping event_type → Figure.
    """
    figs = {}
    for ev_type in ["CPI", "FOMC"]:
        ev_data = event_windows_df[event_windows_df["event_type"] == ev_type]
        if ev_data.empty:
            continue

        pivot = (
            ev_data
            .pivot_table(values="simple_return", index="pre_event_regime",
                         columns="relative_day", aggfunc="mean")
            * 100
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
        ax.set_title(f"{ev_type} Mean Simple Return (%) by Relative Day and Pre-Event Regime")
        ax.set_xlabel("Relative Trading Day")
        ax.set_ylabel("Pre-Event Regime")
        fig.tight_layout()

        out = DATA_DIR / f"{ev_type.lower()}_event_window_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved {out}")
        figs[ev_type] = fig

    return figs


# ── Cumulative event-return paths ──────────────────────────────────────────

def plot_cumulative_event_returns(event_windows_df: pd.DataFrame) -> dict[str, plt.Figure]:
    """
    Average cumulative return path from Day -5 to +5, one line per regime.

    Saves to data/cpi_cumulative_event_returns.png and
              data/fomc_cumulative_event_returns.png.

    Returns
    -------
    dict mapping event_type → Figure.
    """
    figs = {}
    for ev_type in ["CPI", "FOMC"]:
        ev_data = event_windows_df[event_windows_df["event_type"] == ev_type]
        if ev_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        for regime in ["Calm", "Choppy", "Stressed"]:
            regime_data = ev_data[ev_data["pre_event_regime"] == regime]
            if regime_data.empty:
                continue

            avg_log = regime_data.groupby("relative_day")["log_return"].mean()
            cum_path = (np.exp(avg_log.cumsum()) - 1) * 100

            ax.plot(cum_path.index, cum_path.values,
                    marker="o", label=regime, color=REGIME_COLORS[regime])

        ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="Event Day")
        ax.axhline(0, color="gray",  linestyle="-",  alpha=0.3)
        ax.set_title(f"{ev_type} Average Cumulative Return Path [-5 to +5]")
        ax.set_xlabel("Relative Trading Day")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        fig.tight_layout()

        out = DATA_DIR / f"{ev_type.lower()}_cumulative_event_returns.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved {out}")
        figs[ev_type] = fig

    return figs


# ── Event-day return boxplot ───────────────────────────────────────────────

def plot_event_day_boxplot(event_windows_df: pd.DataFrame) -> plt.Figure:
    """
    Boxplot of Day 0 simple returns grouped by event type and pre-event regime.

    Saves to data/event_day_return_boxplot.png.
    """
    out = DATA_DIR / "event_day_return_boxplot.png"

    day0 = event_windows_df[event_windows_df["relative_day"] == 0].copy()
    day0["simple_return_pct"] = day0["simple_return"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x="event_type", y="simple_return_pct",
        hue="pre_event_regime", data=day0,
        palette=REGIME_COLORS, ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Event-Day (Day 0) Return Distribution by Regime")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Simple Return (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved {out}")
    return fig
