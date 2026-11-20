"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.parsers.cicids import parse_cicids_file
from data.synthetic import SYNTHETIC_FILE_NAME, generate_cicids_like_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sysmon_sample() -> Path:
    """The synthetic Sysmon export committed with the repository."""
    return REPO_ROOT / "data" / "samples" / "sysmon_sample.xml"


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic flow file in the original (quirky) CICIDS2017 CSV format."""
    path = tmp_path_factory.mktemp("raw") / SYNTHETIC_FILE_NAME
    return generate_cicids_like_csv(path, n_benign=400, n_scan=120, n_ddos=100)


@pytest.fixture(scope="session")
def flows(synthetic_csv: Path) -> pd.DataFrame:
    """The synthetic flows, normalized."""
    return pd.concat(parse_cicids_file(synthetic_csv, chunk_rows=250), ignore_index=True)
