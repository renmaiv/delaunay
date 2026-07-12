# Backend image (judge-only — no ML encoders). Serves the FastAPI API; the
# frontend is deployed separately (Vercel) and points at this via VITE_API_BASE.
FROM python:3.11-slim

WORKDIR /app

# Core deps only (fastapi/uvicorn/pydantic/anthropic/...). The ML encoder stack
# (requirements-ml.txt) is intentionally omitted — the Anthropic judge covers
# every category, and torch/transformers would bloat the image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY shared ./shared
COPY config.yaml run.py ./

EXPOSE 8000

# Render (and most PaaS) inject $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn server.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
