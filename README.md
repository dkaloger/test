# Tiny GPT — train and deploy a small LLM from scratch

A ~3.2M-parameter, character-level GPT (decoder-only transformer), trained on Tiny Shakespeare and served over HTTP.

| File | What it is |
|---|---|
| `model.py` | The transformer: embeddings → causal self-attention blocks → next-token logits |
| `train.py` | Training loop (AdamW, warmup + cosine LR schedule, train/val loss) |
| `serve.py` | FastAPI server: `/generate`, `/stream`, `/health`, and a web UI at `/` |
| `Dockerfile` | Container for deployment |

## Train
```bash
pip install -r requirements.txt
python train.py --data data/input.txt --iters 3000   # ~15 min on 4 CPU cores
```
Writes `checkpoints/model.pt`. Train on your own text by passing a different `--data` file.

## Run locally
```bash
uvicorn serve:app --port 8000
# open http://localhost:8000, or:
curl -X POST localhost:8000/generate -H 'Content-Type: application/json' \
     -d '{"prompt":"ROMEO:","max_new_tokens":200}'
```

## Deploy
```bash
docker build -t tiny-gpt .
docker run -p 8000:8000 tiny-gpt
```
The image runs on any container host (Fly.io, Render, Railway, Google Cloud Run, Hugging Face Spaces with the Docker SDK). It needs no GPU.
