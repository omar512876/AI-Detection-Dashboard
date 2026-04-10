import os
from typing import Any

import httpx

ORIGINALITY_API_URL = os.getenv(
    "ORIGINALITY_API_URL",
    "https://api.originality.ai/api/v1/scan/ai",
)


class OriginalityAPIError(Exception):
    pass


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        score = float(value)
        if score > 1.0 and score <= 100.0:
            return score / 100.0
        if 0.0 <= score <= 1.0:
            return score
    return None


def _extract_score(payload: dict[str, Any]) -> float:
    for key in ("score", "ai_score", "aiScore", "ai_probability", "probability", "originalityAI"):
        score = _coerce_score(payload.get(key))
        if score is not None:
            return score

    if isinstance(payload.get("data"), dict):
        nested_data = payload["data"]
        for key in ("score", "ai_score", "aiScore", "ai_probability", "probability", "originalityAI"):
            score = _coerce_score(nested_data.get(key))
            if score is not None:
                return score

    raise OriginalityAPIError("Missing AI score")


def _label_from_score(score: float) -> str:
    if score >= 0.65:
        return "Likely AI"
    if score <= 0.35:
        return "Likely Human"
    return "Mixed"


async def check_originality(text: str) -> dict[str, Any]:
    api_key = os.getenv("ORIGINALITY_API_KEY")
    if not api_key:
        raise OriginalityAPIError("Missing ORIGINALITY_API_KEY")

    headers = {
        "X-OAI-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    body = {"content": text}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(ORIGINALITY_API_URL, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise OriginalityAPIError("Originality request failed") from error

    score = _extract_score(payload)
    return {"score": score, "label": _label_from_score(score)}
