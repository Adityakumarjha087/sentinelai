from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from common.settings import get_settings
from data.cli import main


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sysmon_sample: Path) -> Iterator[Path]:
    """An isolated data directory holding only the Sysmon sample."""
    root = tmp_path / "data"
    (root / "samples").mkdir(parents=True)
    shutil.copy(sysmon_sample, root / "samples" / sysmon_sample.name)
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(root))
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


def test_offline_pipeline_end_to_end(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth"]) == 0
    assert main(["normalize", "--dataset", "synthetic"]) == 0
    assert main(["features", "--dataset", "synthetic"]) == 0
    assert main(["stats", "--dataset", "synthetic"]) == 0

    processed = data_dir / "processed"
    events = pd.read_parquet(processed / "events" / "synthetic" / "Synthetic-WorkingHours.parquet")
    features = pd.read_parquet(processed / "features" / "synthetic" / "Synthetic-WorkingHours.parquet")
    assert len(events) == len(features) == 2_600
    assert features["event_id"].tolist() == events["event_id"].tolist()
    assert (processed / "features" / "synthetic" / "featurizer.json").is_file()
    assert len(pd.read_parquet(processed / "events" / "sysmon" / "sysmon.parquet")) == 13

    output = capsys.readouterr().out
    assert "events: 2,600" in output
    assert "PortScan" in output


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["explode"])
    assert excinfo.value.code == 2
