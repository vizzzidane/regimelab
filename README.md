# RegimeLab

RegimeLab is an event-driven market regime analysis project.

It studies how SPY behaves around CPI releases and FOMC decision dates under different market regimes. The current version is a local Jupyter notebook prototype. It is not a trading bot, stock predictor, or live trading system.

The goal is to build a credible finance/ML analysis pipeline first, then later package it into a full-stack application.

---

## Current Status

Phase 1 is complete.

The notebook currently does two things:

1. Classifies SPY market history into regimes using K-Means.
2. Studies CPI and FOMC event-window returns from `-5` to `+5` trading days, grouped by pre-event regime.

---

## Why This Project Exists

Many beginner finance projects try to predict the next stock price directly. That is usually hard to justify and easy to overclaim.

RegimeLab takes a more careful approach:

> Instead of predicting prices, it studies how asset behaviour around macro events differs across historical market regimes.

This makes the analysis more defensible because the focus is on methodology, assumptions, and limitations rather than unsupported claims of profitability.

---

## Methodology

### 1. Data

The project uses daily SPY data from 2005 onwards.

The data is cached locally as:

```text
data/spy_daily.csv