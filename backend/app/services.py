from pathlib import Path
import os
import subprocess
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def load_csv(filename: str) -> list[dict]:
    path = DATA_DIR / filename

    if not path.exists():
        return []

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def load_event_windows(
    event_type: str | None = None,
    pre_event_regime: str | None = None,
) -> list[dict]:
    rows = load_csv("event_windows.csv")

    if event_type:
        rows = [row for row in rows if row.get("event_type") == event_type]

    if pre_event_regime:
        rows = [
            row
            for row in rows
            if row.get("pre_event_regime") == pre_event_regime
        ]

    return rows


def run_pipeline() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "scripts/run_phase1.py"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return {
        "status": "completed",
        "message": "Pipeline completed successfully",
    }