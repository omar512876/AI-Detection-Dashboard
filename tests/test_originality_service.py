import asyncio

import httpx
import pytest

from app.services.originality import (
    OriginalityAPIError,
    _coerce_score,
    _extract_score,
    _label_from_score,
    check_originality,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (75, 0.75),
        (0.75, 0.75),
        (0, 0.0),
        (100, 1.0),
        (-1, None),
        (101, None),
        ("0.9", None),
    ],
)
def test_coerce_score(value, expected):
    assert _coerce_score(value) == expected


def test_extract_score_from_top_level():
    assert _extract_score({"aiScore": 0.7}) == 0.7


def test_extract_score_from_nested_data():
    assert _extract_score({"data": {"probability": 82}}) == 0.82


def test_extract_score_raises_when_missing():
    with pytest.raises(OriginalityAPIError, match="Missing AI score"):
        _extract_score({})


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.8, "Likely AI"),
        (0.65, "Likely AI"),
        (0.5, "Mixed"),
        (0.35, "Likely Human"),
        (0.1, "Likely Human"),
    ],
)
def test_label_from_score(score, expected):
    assert _label_from_score(score) == expected


def test_check_originality_requires_api_key(monkeypatch):
    monkeypatch.delenv("ORIGINALITY_API_KEY", raising=False)

    with pytest.raises(OriginalityAPIError, match="Missing ORIGINALITY_API_KEY"):
        asyncio.run(check_originality("hello"))


def test_check_originality_success(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"score": 0.9}

    class MockClient:
        def __init__(self, timeout):
            assert timeout == 20.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            assert headers["X-OAI-API-KEY"] == "test-key"
            assert json == {"content": "sample text"}
            assert url
            return MockResponse()

    monkeypatch.setenv("ORIGINALITY_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    result = asyncio.run(check_originality("sample text"))
    assert result == {"score": 0.9, "label": "Likely AI"}


def test_check_originality_wraps_invalid_json(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("bad json")

    class MockClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            return MockResponse()

    monkeypatch.setenv("ORIGINALITY_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    with pytest.raises(OriginalityAPIError, match="Originality request failed"):
        asyncio.run(check_originality("sample text"))
