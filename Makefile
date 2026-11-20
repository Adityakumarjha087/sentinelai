.PHONY: install lint typecheck test check data demo-data train mlflow-ui kb eval

install:
	uv sync

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest --cov=common --cov=data --cov=detection --cov=rag --cov=agent --cov=guardrails --cov=api --cov=eval --cov-report=term-missing

check: lint typecheck test

# Real dataset: ~306 MB download, ~2 minutes of processing.
data:
	uv run sentinel-data download
	uv run sentinel-data normalize
	uv run sentinel-data features
	uv run sentinel-data stats

# Offline: tiny synthetic flows, no download.
demo-data:
	uv run sentinel-data synth
	uv run sentinel-data normalize --dataset synthetic
	uv run sentinel-data features --dataset synthetic
	uv run sentinel-data stats --dataset synthetic

# Train both detectors on the real feature tables (~1 minute) and save the best.
train:
	uv run sentinel-detect

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Build the ATT&CK + CVE knowledge base (downloads ~50 MB of data and two small models).
kb:
	uv run sentinel-kb build

# End-to-end evaluation with the offline baseline analysts.
eval:
	uv run sentinel-eval
