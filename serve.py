"""Serve the trained model over HTTP.

    uvicorn serve:app --host 0.0.0.0 --port 8000

GET  /            -> small web UI
POST /generate    -> {"prompt": "...", "max_new_tokens": 300, "temperature": 0.8, "top_k": 40}
POST /stream      -> same body, streams text as it is generated
GET  /health
"""
import os

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from model import load_checkpoint

CKPT = os.environ.get("MODEL_PATH", "checkpoints/model.pt")
torch.set_num_threads(max(1, os.cpu_count() or 1))

model, tok = load_checkpoint(CKPT)

app = FastAPI(title="tiny-gpt")


class GenRequest(BaseModel):
    prompt: str = "ROMEO:"
    max_new_tokens: int = Field(300, ge=1, le=2000)
    temperature: float = Field(0.8, gt=0, le=2.0)
    top_k: int = Field(40, ge=0, le=200)


def _tokens(req: GenRequest):
    ids = tok.encode(req.prompt) or [tok.stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long)
    for t in model.generate(idx, req.max_new_tokens, req.temperature, req.top_k):
        yield tok.decode([t])


@app.get("/health")
def health():
    return {"status": "ok", "params": sum(p.numel() for p in model.parameters())}


@app.post("/generate")
def generate(req: GenRequest):
    return {"prompt": req.prompt, "completion": "".join(_tokens(req))}


@app.post("/stream")
def stream(req: GenRequest):
    return StreamingResponse(_tokens(req), media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8").read()
