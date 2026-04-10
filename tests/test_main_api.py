from fastapi.testclient import TestClient

from app.main import app
from app.services.gptzero import GPTZeroAPIError
from app.services.originality import OriginalityAPIError


client = TestClient(app)


def test_home_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_analyze_success(monkeypatch):
    async def fake_gptzero(_text):
        return {"score": 0.8, "label": "Likely AI"}

    async def fake_originality(_text):
        return {"score": 0.2, "label": "Likely Human"}

    monkeypatch.setattr("app.main.check_gptzero", fake_gptzero)
    monkeypatch.setattr("app.main.check_originality", fake_originality)

    response = client.post("/analyze", json={"text": "hello"})
    assert response.status_code == 200
    assert response.json() == {
        "gptzero": {"score": 0.8, "label": "Likely AI"},
        "originality": {"score": 0.2, "label": "Likely Human"},
    }


def test_analyze_maps_service_failure_to_error_payload(monkeypatch):
    async def failing_gptzero(_text):
        raise GPTZeroAPIError("failed")

    async def failing_originality(_text):
        raise OriginalityAPIError("failed")

    monkeypatch.setattr("app.main.check_gptzero", failing_gptzero)
    monkeypatch.setattr("app.main.check_originality", failing_originality)

    response = client.post("/analyze", json={"text": "hello"})
    assert response.status_code == 200
    assert response.json() == {
        "gptzero": {"error": "API failed"},
        "originality": {"error": "API failed"},
    }


def test_analyze_handles_generic_exception(monkeypatch):
    async def failing_gptzero(_text):
        raise RuntimeError("boom")

    async def ok_originality(_text):
        return {"score": 0.5, "label": "Mixed"}

    monkeypatch.setattr("app.main.check_gptzero", failing_gptzero)
    monkeypatch.setattr("app.main.check_originality", ok_originality)

    response = client.post("/analyze", json={"text": "hello"})
    assert response.status_code == 200
    assert response.json() == {
        "gptzero": {"error": "API failed"},
        "originality": {"score": 0.5, "label": "Mixed"},
    }
