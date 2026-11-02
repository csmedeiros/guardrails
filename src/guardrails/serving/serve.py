"""FastAPI guard endpoint.  run: uv run uvicorn guardrails.serving.serve:app --port 8100"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app_logger = logging.getLogger(__name__)

from guardrails.core.registry import load_guard

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["m"] = load_guard()
    yield


app = FastAPI(title="Guardrail classifier (EN/PT-BR/ES)", lifespan=lifespan)


class Req(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class BatchReq(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class Res(BaseModel):
    label: str
    unsafe_prob: float
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok", "threshold": state["m"].thr, "model": getattr(state["m"], "name", "env")}


@app.post("/classify", response_model=Res)
def classify(r: Req):
    # logger.info(f"Classifying text using model{state["m"].name}: {r.text}")
    return state["m"].classify([r.text])[0]


@app.post("/classify_batch", response_model=list[Res])
def classify_batch(r: BatchReq):
    app_logger.info(f"Classifying texts using model{state["m"].name}: {r.texts}")
    return state["m"].classify(r.texts)
