.PHONY: install lint typecheck test check data demo-data

install:
	uv sync

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest --cov=common --cov=data --cov-report=term-missing

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
