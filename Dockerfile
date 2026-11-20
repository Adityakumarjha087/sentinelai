# SentinelAI - one image for the API and the dashboard.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# The knowledge base's embedding and reranking models, baked in so containers
# need neither network access nor a writable cache at run time. Keep the names
# in sync with rag/embeddings.py and rag/retriever.py.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY common ./common
COPY data ./data
COPY detection ./detection
COPY rag ./rag
COPY agent ./agent
COPY guardrails ./guardrails
COPY eval ./eval
COPY api ./api
COPY ui ./ui
RUN uv sync --locked --no-dev

# Run as an unprivileged user; it only needs to read /app.
RUN useradd --create-home --uid 10001 sentinel
USER sentinel
ENV HOME=/home/sentinel

EXPOSE 8000 8501
CMD ["uvicorn", "--factory", "api.main:default_app", "--host", "0.0.0.0", "--port", "8000"]
