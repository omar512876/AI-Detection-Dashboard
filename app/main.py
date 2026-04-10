import asyncio
from typing import Any, Coroutine

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from app.services.gptzero import GPTZeroAPIError, check_gptzero
from app.services.originality import OriginalityAPIError, check_originality

app = FastAPI(title="AI Detection Dashboard")

BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class AnalyzeRequest(BaseModel):
    text: str
    gptzero_api_key: str | None = None
    originality_api_key: str | None = None


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    gptzero_api_key = (payload.gptzero_api_key or "").strip()
    originality_api_key = (payload.originality_api_key or "").strip()

    if not gptzero_api_key and not originality_api_key:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one API key (GPTZero or Originality.ai).",
        )

    pending_services: list[tuple[str, Coroutine[Any, Any, dict[str, Any]]]] = []
    if gptzero_api_key:
        pending_services.append(
            ("gptzero", check_gptzero(payload.text, api_key=gptzero_api_key))
        )
    if originality_api_key:
        pending_services.append(
            ("originality", check_originality(payload.text, api_key=originality_api_key))
        )

    service_results: dict[str, dict[str, Any]] = {}
    if pending_services:
        names = [name for name, _ in pending_services]
        coroutines = [coroutine for _, coroutine in pending_services]
        responses = await asyncio.gather(*coroutines, return_exceptions=True)

        for name, response in zip(names, responses):
            if name == "gptzero" and isinstance(response, (GPTZeroAPIError, Exception)):
                service_results[name] = {"error": "API failed"}
            elif name == "originality" and isinstance(
                response, (OriginalityAPIError, Exception)
            ):
                service_results[name] = {"error": "API failed"}
            else:
                service_results[name] = response

    gptzero_result = service_results.get("gptzero", {"error": "API key not provided"})
    originality_result = service_results.get("originality", {"error": "API key not provided"})

    return {"gptzero": gptzero_result, "originality": originality_result}
