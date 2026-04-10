import os
from typing import Any

import httpx

GPTZERO_API_URL = os.getenv("GPTZERO_API_URL", "https://api.gptzero.me/v2/predict/text")


class GPTZeroAPIError(Exception):
    pass


def _read_first_document(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload.get("documents")
    if isinstance(documents, list) and documents:
        first_document = documents[0]
        if isinstance(first_document, dict):
            return first_document
    return {}


def _extract_score(payload: dict[str, Any]) -> float:
    document = _read_first_document(payload)
    for source in (document, payload):
        for key in ("average_generated_prob", "ai_probability", "score", "probability"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    raise GPTZeroAPIError("Missing AI score")


def _normalize_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    if "mixed" in normalized:
        return "Mixed"
    if "human" in normalized:
        return "Likely Human"
    if "ai" in normalized:
        return "Likely AI"
    return raw_label.strip().title()


def _extract_label(payload: dict[str, Any]) -> str:
    document = _read_first_document(payload)
    for source in (document, payload):
        for key in ("predicted_class", "classification", "label", "verdict"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_label(value)
    raise GPTZeroAPIError("Missing classification label")


async def check_gptzero(text: str, api_key: str | None = None) -> dict[str, Any]:
    api_key = api_key or os.getenv("GPTZERO_API_KEY")
    if not api_key:
        raise GPTZeroAPIError("Missing GPTZERO_API_KEY")

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = {"document": text}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(GPTZERO_API_URL, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise GPTZeroAPIError("GPTZero request failed") from error

    return {
        "score": _extract_score(payload),
        "label": _extract_label(payload),
    }


analyze_text_with_gptzero = check_gptzero
