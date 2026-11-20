from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.parsers.cicids import (
    file_tag,
    normalize_cicids_frame,
    normalize_label,
    parse_cicids_file,
    parse_local_timestamps,
    snake_case,
)
from data.schema import COMMON_COLUMNS, frame_to_events


def test_snake_case_and_file_tag() -> None:
    assert snake_case(" Flow Bytes/s") == "flow_bytes_s"
    assert snake_case("Init_Win_bytes_forward") == "init_win_bytes_forward"
    name = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet"
    assert file_tag(Path(name)) == "friday-workinghours-afternoon-portscan"


def test_local_timestamps_handle_12_hour_clock_and_timezone() -> None:
    parsed = parse_local_timestamps(
        pd.Series(["03/07/2017 08:55:58", "7/7/2017 1:05", "7/7/2017 12:30", "garbage", None])
    )
    # Fredericton is UTC-3 in July; "1:05" is 13:05 local.
    assert parsed[0] == pd.Timestamp("2017-07-03 11:55:58", tz="UTC")
    assert parsed[1] == pd.Timestamp("2017-07-07 16:05", tz="UTC")
    assert parsed[2] == pd.Timestamp("2017-07-07 15:30", tz="UTC")
    assert parsed[3:].isna().all()


def test_labels_are_trimmed_and_dash_is_repaired() -> None:
    labels = normalize_label(
        pd.Series([" BENIGN ", "Web Attack \x96 XSS", "Web Attack \ufffd Sql Injection"])
    )
    assert labels.tolist() == ["BENIGN", "Web Attack - XSS", "Web Attack - Sql Injection"]


def test_csv_quirks_are_absorbed(flows: pd.DataFrame) -> None:
    # 400 benign + 120 scan + 100 ddos; the 5 trailing empty rows are dropped.
    assert len(flows) == 620
    assert list(flows.columns[: len(COMMON_COLUMNS)]) == list(COMMON_COLUMNS)
    assert flows["event_id"].is_unique
    assert flows["label"].value_counts().to_dict() == {"BENIGN": 400, "PortScan": 120, "DDoS": 100}
    assert flows.loc[flows["label"] == "BENIGN", "is_attack"].eq(False).all()
    assert flows.loc[flows["label"] != "BENIGN", "is_attack"].all()
    # Leading-space headers are cleaned; the duplicated header is not kept twice.
    assert "flow_bytes_s" in flows.columns
    assert not [column for column in flows.columns if column.endswith("_1")]
    # "Infinity" survives parsing as inf; cleaning is the featurizer's job.
    assert np.isinf(flows["flow_bytes_s"]).any()


def test_afternoon_attacks_land_at_the_right_utc_time(flows: pd.DataFrame) -> None:
    scan = flows[flows["label"] == "PortScan"]["timestamp"]
    assert scan.min() == pd.Timestamp("2017-07-07 16:05", tz="UTC")  # written as "1:05"
    assert flows["timestamp"].min() >= pd.Timestamp("2017-07-07 12:00", tz="UTC")
    assert flows["timestamp"].max() < pd.Timestamp("2017-07-07 20:00", tz="UTC")


def test_rows_validate_against_the_event_model(flows: pd.DataFrame) -> None:
    events = frame_to_events(flows.head(25))
    assert {event.protocol for event in events} <= {"tcp", "udp"}
    assert all(event.raw.count(",") >= 18 for event in events)
    assert all("flow_duration" in event.attributes for event in events)


def test_chunking_does_not_change_the_result(synthetic_csv: Path, flows: pd.DataFrame) -> None:
    whole = pd.concat(parse_cicids_file(synthetic_csv, chunk_rows=10_000), ignore_index=True)
    pd.testing.assert_frame_equal(whole, flows)


def test_max_rows_stops_early(synthetic_csv: Path) -> None:
    limited = pd.concat(parse_cicids_file(synthetic_csv, chunk_rows=64, max_rows=150))
    assert len(limited) == 150


def test_parquet_mirror_layout_is_supported(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "Flow ID": ["a", "b"],
            "Source IP": ["192.168.10.5", "172.16.0.1"],
            "Source Port": [50000, 40000],
            "Destination IP": ["192.168.10.50", "192.168.10.50"],
            "Destination Port": [443, 22],
            "Protocol": [6, 0],
            "Timestamp": pd.to_datetime(["2017-07-07 16:00:00", "2017-07-07 16:01:00"]),
            "Flow Duration": [1200, 3],
            "Fwd Header Length": [40, 20],
            "Fwd Header Length.1": [40, 20],
            "Label": ["BENIGN", "PortScan"],
        }
    )
    path = tmp_path / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet"
    source.to_parquet(path)
    frame = pd.concat(parse_cicids_file(path))
    assert frame["timestamp"].tolist() == [
        pd.Timestamp("2017-07-07 16:00", tz="UTC"),  # mirror timestamps are already UTC
        pd.Timestamp("2017-07-07 16:01", tz="UTC"),
    ]
    assert frame["protocol"].tolist() == ["tcp", "other"]
    assert frame["event_id"].tolist() == [
        "cic17-friday-workinghours-afternoon-portscan-0000000",
        "cic17-friday-workinghours-afternoon-portscan-0000001",
    ]
    assert "fwd_header_length_1" not in frame.columns


def test_machine_learning_variant_is_rejected_with_guidance() -> None:
    frame = pd.DataFrame({" Destination Port": [80], " Flow Duration": [1], " Label": ["BENIGN"]})
    with pytest.raises(ValueError, match="TrafficLabelling"):
        normalize_cicids_frame(frame, "x")
