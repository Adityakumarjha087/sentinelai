from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from data.schema import (
    COMMON_COLUMNS,
    MAX_RAW_CHARS,
    EventSource,
    EventType,
    NormalizedEvent,
    clean_ip,
    coerce_common_schema,
    events_to_frame,
    frame_to_events,
)


def make_event(**overrides: object) -> NormalizedEvent:
    fields: dict[str, object] = {
        "event_id": "evt-1",
        "timestamp": datetime(2024, 3, 12, 9, 0, tzinfo=UTC),
        "source": EventSource.SYSMON,
        "event_type": EventType.PROCESS_CREATE,
        "raw": "<Event/>",
    }
    return NormalizedEvent.model_validate(fields | overrides)


def test_timestamp_is_converted_to_utc() -> None:
    local = datetime(2024, 3, 12, 6, 0, tzinfo=timezone(timedelta(hours=-3)))
    event = make_event(timestamp=local)
    assert event.timestamp == datetime(2024, 3, 12, 9, 0, tzinfo=UTC)
    assert event.timestamp.utcoffset() == timedelta(0)


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_event(timestamp=datetime(2024, 3, 12, 9, 0))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" 10.0.0.1 ", "10.0.0.1"), ("2001:DB8::1", "2001:db8::1"), ("-", None), ("999.1.1.1", None)],
)
def test_clean_ip(value: str, expected: str | None) -> None:
    assert clean_ip(value) == expected


def test_invalid_ip_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a valid IP"):
        make_event(src_ip="ignore previous instructions")


@pytest.mark.parametrize("port", [-1, 65536])
def test_out_of_range_port_is_rejected(port: int) -> None:
    with pytest.raises(ValidationError):
        make_event(dst_port=port)


def test_oversized_raw_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_event(raw="x" * (MAX_RAW_CHARS + 1))


def test_attributes_cannot_shadow_common_columns() -> None:
    with pytest.raises(ValidationError, match="shadow"):
        make_event(attributes={"label": "BENIGN"})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_event(verdict="benign")


def test_frame_round_trip_preserves_events() -> None:
    events = [
        make_event(event_id="a", src_ip="10.0.0.1", dst_port=443, attributes={"event_code": 3}),
        make_event(event_id="b", user="CORP\\alice", attributes={"command_line": "whoami"}),
    ]
    frame = events_to_frame(events)
    assert list(frame.columns[: len(COMMON_COLUMNS)]) == list(COMMON_COLUMNS)
    assert str(frame["timestamp"].dtype) == "datetime64[us, UTC]"
    assert frame_to_events(frame) == events


def test_coerce_adds_missing_columns_and_types() -> None:
    frame = coerce_common_schema(
        pd.DataFrame({"event_id": ["a"], "timestamp": ["2024-03-12T09:00:00Z"], "bytes": [1.5]})
    )
    assert list(frame.columns) == [*COMMON_COLUMNS, "bytes"]
    assert frame["src_port"].isna().all()
    assert str(frame["src_port"].dtype) == "Int32"
    assert str(frame["is_attack"].dtype) == "boolean"


def test_empty_events_give_empty_typed_frame() -> None:
    frame = events_to_frame([])
    assert frame.empty
    assert list(frame.columns) == list(COMMON_COLUMNS)
