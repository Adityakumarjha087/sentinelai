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

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 sentinel && chown -R sentinel /app
USER sentinel
ENV HOME=/home/sentinel

EXPOSE 8000 8501
CMD ["uvicorn", "--factory", "api.main:default_app", "--host", "0.0.0.0", "--port", "8000"]
