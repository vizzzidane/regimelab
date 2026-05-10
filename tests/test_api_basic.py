from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary_returns_list():
    response = client.get("/api/summary")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_event_windows_filter_by_event_type():
    response = client.get(
        "/api/event-windows?event_type=CPI"
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)

    for row in response.json():
        assert row["event_type"] == "CPI"