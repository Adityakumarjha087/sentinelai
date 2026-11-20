from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.normalize import (
    load_events,
    normalize_cicids_dir,
    normalize_sysmon_files,
    write_normalized,
)
from data.schema import COMMON_COLUMNS, EventType, frame_to_events


def test_cicids_dir_round_trips_through_parquet(
    synthetic_csv: Path, flows: pd.DataFrame, tmp_path: Path
) -> None:
    written = normalize_cicids_dir(synthetic_csv.parent, tmp_path / "events", chunk_rows=128)
    (out_path,) = written
    assert out_path.name == "Synthetic-WorkingHours.parquet"
    assert written[out_path] == len(flows)
    assert not list(out_path.parent.glob("*.part"))
    pd.testing.assert_frame_equal(load_events(out_path), flows, check_dtype=False)
    assert len(load_events(out_path.parent, columns=["event_id"])) == len(flows)


def test_sysmon_round_trips_through_parquet(sysmon_sample: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "sysmon.parquet"
    assert normalize_sysmon_files([sysmon_sample], out_path) == 13
    frame = load_events(out_path)
    assert list(frame.columns[: len(COMMON_COLUMNS)]) == list(COMMON_COLUMNS)
    events = frame_to_events(frame)
    assert sum(event.event_type is EventType.PROCESS_CREATE for event in events) == 4
    assert all(event.is_attack is None for event in events)


def test_empty_inputs_raise_instead_of_writing_empty_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        write_normalized([], tmp_path / "empty.parquet")
    assert not (tmp_path / "empty.parquet").exists()
    with pytest.raises(FileNotFoundError):
        normalize_cicids_dir(tmp_path, tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        load_events(tmp_path)
