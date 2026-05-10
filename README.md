# RegimeLab

Event-driven market regime analysis using SPY daily data, K-Means clustering, and CPI/FOMC event-window study.

---

## Project Structure

```
regimelab/
├── data/                        # Cached data and generated outputs
├── events/
│   ├── event_dates.py           # CPI_DATES and FOMC_DATES (2010–2024)
│   └── events.csv               # Combined event table
├── notebooks/
│   └── 01_regime_analysis.ipynb # Exploratory notebook (Phase 1A + 1B)
├── src/
│   └── regimelab/               # Refactored Python modules (Phase 1C)
│       ├── __init__.py
│       ├── config.py            # Paths and constants
│       ├── data_loader.py       # SPY data download / cache
│       ├── features.py          # Rolling regime features
│       ├── clustering.py        # K-Means regime model
│       ├── events.py            # Event-date loading and alignment
│       ├── event_study.py       # Window extraction and summary metrics
│       └── visualization.py    # Chart generation
├── scripts/
│   └── run_phase1.py            # End-to-end pipeline runner
├── tests/
│   └── test_basic_pipeline.py   # Basic smoke tests
├── requirements.txt
└── README.md
```

---

## Phases

### Phase 1A – Regime Clustering (Notebook)
The notebook `notebooks/01_regime_analysis.ipynb` downloads SPY data, computes rolling features (volatility, momentum, drawdown), and clusters the market into **Calm**, **Choppy**, and **Stressed** regimes using K-Means (k=3).

### Phase 1B – Event-Window Analysis (Notebook)
The same notebook extends the analysis by aligning CPI and FOMC dates to SPY trading days, extracting [-5, +5] trading-day return windows, and computing event-study summary statistics and charts.

### Phase 1C – Python Modules (this refactor)
The notebook logic has been refactored into clean, importable modules under `src/regimelab/`.  The notebook remains the exploratory artifact; the modules reproduce the same outputs programmatically.

---

## Running the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full Phase 1 pipeline from the project root:

```bash
python scripts/run_phase1.py
```

This will:
1. Load (or download) SPY daily data and cache it to `data/spy_daily.csv`.
2. Compute regime features and fit K-Means (k=3).
3. Validate regimes against known stress periods (GFC, COVID, 2022).
4. Align CPI/FOMC event dates to trading days.
5. Extract event windows and compute summary metrics.
6. Save all charts and CSVs to `data/`.

---

## Running Tests

```bash
pytest tests/test_basic_pipeline.py -v
```

---

## Methodology Notes

- All rolling features use only past data (`min_periods` enforced). No lookahead bias in feature construction.
- K-Means is fitted on the full historical dataset, so regime labels are **exploratory** and not suitable for walk-forward backtesting without further adaptation.
- The primary grouping variable for event-study analysis is `pre_event_regime` (regime on Day −1), which reduces event-day lookahead bias.
- The Stressed regime has very few events (≤2 CPI, 1 FOMC). Do not draw strong conclusions from those buckets.
- **This project makes no claims of alpha, profitability, or trading readiness.**

---

## Disclaimer

RegimeLab is an analytical research tool. It is not a trading system and does not provide investment advice.
