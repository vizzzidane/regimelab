from fastapi import FastAPI, HTTPException

from backend.app.services import (
    load_csv,
    load_event_windows,
    run_pipeline,
)


app = FastAPI(title="RegimeLab API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/summary")
def get_summary() -> list[dict]:
    return load_csv("event_study_summary.csv")


@app.get("/api/aligned-events")
def get_aligned_events() -> list[dict]:
    return load_csv("aligned_events.csv")


@app.get("/api/event-windows")
def get_event_windows(
    event_type: str | None = None,
    pre_event_regime: str | None = None,
) -> list[dict]:
    return load_event_windows(
        event_type,
        pre_event_regime,
    )


@app.post("/api/run-pipeline")
def post_run_pipeline() -> dict:
    try:
        return run_pipeline()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc