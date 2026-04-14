import asyncio

import httpx
import pytest

from app.services.gptzero import (
    GPTZeroAPIError,
    _extract_label,
    _extract_score,
    _normalize_label,
    check_gptzero,
)


def test_extract_score_prefers_document_value():
    payload = {
        "documents": [{"average_generated_prob": 0.87}],
        "score": 0.2,
    }
    assert _extract_score(payload) == 0.87


def test_extract_score_uses_top_level_fallback():
    payload = {"score": 0.42}
    assert _extract_score(payload) == 0.42


def test_extract_score_raises_when_missing():
    with pytest.raises(GPTZeroAPIError, match="Missing AI score"):
        _extract_score({})


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        (" mixed content ", "Mixed"),
        ("Human-written", "Likely Human"),
        (" mostly AI ", "Likely AI"),
        ("unknown verdict", "Unknown Verdict"),
    ],
)
def test_normalize_label(raw_label, expected):
    assert _normalize_label(raw_label) == expected


def test_extract_label_uses_top_level_and_normalizes():
    payload = {"label": "mostly ai"}
    assert _extract_label(payload) == "Likely AI"


def test_extract_label_raises_when_missing():
    with pytest.raises(GPTZeroAPIError, match="Missing classification label"):
        _extract_label({})


def test_check_gptzero_requires_api_key(monkeypatch):
    monkeypatch.delenv("GPTZERO_API_KEY", raising=False)

    with pytest.raises(GPTZeroAPIError, match="Missing GPTZERO_API_KEY"):
        asyncio.run(check_gptzero("hello"))


def test_check_gptzero_success(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "documents": [
                    {"average_generated_prob": 0.33, "predicted_class": "human"}
                ]
            }

    class MockClient:
        def __init__(self, timeout):
            assert timeout == 20.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            assert headers["x-api-key"] == "test-key"
            assert json == {"document": "sample text"}
            assert url
            return MockResponse()

    monkeypatch.setenv("GPTZERO_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    result = asyncio.run(check_gptzero("sample text"))
    assert result == {"score": 0.33, "label": "Likely Human"}


def test_check_gptzero_wraps_http_errors(monkeypatch):
    class MockClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            raise httpx.ConnectError("boom")

    monkeypatch.setenv("GPTZERO_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    with pytest.raises(GPTZeroAPIError, match="GPTZero request failed"):
        asyncio.run(check_gptzero("sample text"))
