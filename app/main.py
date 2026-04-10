import asyncio

from fastapi import FastAPI, Request
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


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    gptzero_data, originality_data = await asyncio.gather(
        check_gptzero(payload.text),
        check_originality(payload.text),
        return_exceptions=True,
    )

    gptzero_result = (
        {"error": "API failed"}
        if isinstance(gptzero_data, (GPTZeroAPIError, Exception))
        else gptzero_data
    )
    originality_result = (
        {"error": "API failed"}
        if isinstance(originality_data, (OriginalityAPIError, Exception))
        else originality_data
    )

    return {"gptzero": gptzero_result, "originality": originality_result}
