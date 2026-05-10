"""
config.py – Project-wide paths and constants.

All other modules import from here so that paths and hyperparameters
are defined in exactly one place.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# PROJECT_ROOT is the top-level regimelab/ directory regardless of where
# the calling script lives.
PROJECT_ROOT  = Path(__file__).resolve().parents[2]   # src/regimelab → src → project root
DATA_DIR      = PROJECT_ROOT / "data"
EVENTS_DIR    = PROJECT_ROOT / "events"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure data directory exists when config is imported
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Data constants ─────────────────────────────────────────────────────────
TICKER     = "SPY"
START_DATE = "2005-01-01"

# ── Feature windows ────────────────────────────────────────────────────────
VOL_WINDOW      = 20   # days for annualised realised volatility
MOMENTUM_WINDOW = 20   # days for cumulative log-return momentum
DRAWDOWN_WINDOW = 60   # days for rolling-max drawdown

# ── Clustering ─────────────────────────────────────────────────────────────
K_FINAL      = 3
RANDOM_STATE = 42
N_INIT       = 10

# ── Event study ────────────────────────────────────────────────────────────
EVENT_WINDOW = 5   # trading days on each side of the event (−5 … +5)

# ── Regime colour palette (shared by visualisation) ───────────────────────
REGIME_COLORS = {
    "Calm":     "#2ecc71",
    "Choppy":   "#f39c12",
    "Stressed": "#e74c3c",
}
