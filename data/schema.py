"""The common event schema every log source is normalized into.

Two equivalent representations are supported:

* :class:`NormalizedEvent` - a validated Pydantic model, used for single events
  (alerts handed to the agent, API payloads, tests).
* a *normalized frame* - a ``pandas.DataFrame`` whose first columns are
  :data:`COMMON_COLUMNS` followed by source-specific attribute columns. Used for
  bulk processing, where validating millions of rows one by one is too slow.

Security note: ``raw`` and every free-text field (``process``, ``user``,
attribute values) are attacker-controllable. Downstream code must treat them as
untrusted data and never as instructions.
"""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_RAW_CHARS = 16_384

AttributeValue = float | int | str | None


class EventSource(StrEnum):
    """Where an event came from."""

    CICIDS2017 = "cicids2017"
    SYSMON = "sysmon"


class EventType(StrEnum):
    """Source-independent event category."""

    NETWORK_FLOW = "network_flow"
    PROCESS_CREATE = "process_create"
    PROCESS_TERMINATE = "process_terminate"
    NETWORK_CONNECTION = "network_connection"
    IMAGE_LOAD = "image_load"
    CREATE_REMOTE_THREAD = "create_remote_thread"
    PROCESS_ACCESS = "process_access"
    FILE_CREATE = "file_create"
    REGISTRY_EVENT = "registry_event"
    DNS_QUERY = "dns_query"
    OTHER = "other"


COMMON_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "source",
    "event_type",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "host",
    "process",
    "user",
    "label",
    "is_attack",
    "raw",
)
_STRING_COLUMNS = (
    "event_id",
    "source",
    "event_type",
    "src_ip",
    "dst_ip",
    "protocol",
    "host",
    "process",
    "user",
    "label",
    "raw",
)
_PORT_COLUMNS = ("src_port", "dst_port")


def clean_ip(value: object) -> str | None:
    """Return a canonical IP string, or ``None`` if ``value`` is not an IP address."""
    if value is None:
        return None
    try:
        return str(ipaddress.ip_address(str(value).strip()))
    except ValueError:
        return None


class NormalizedEvent(BaseModel):
    """One security event in the common schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    timestamp: datetime
    source: EventSource
    event_type: EventType
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = Field(default=None, ge=0, le=65535)
    dst_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str | None = None
    host: str | None = None
    process: str | None = None
    user: str | None = None
    label: str | None = None
    is_attack: bool | None = None
    raw: str = Field(max_length=MAX_RAW_CHARS)
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("src_ip", "dst_ip")
    @classmethod
    def _canonical_ip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = clean_ip(value)
        if cleaned is None:
            raise ValueError(f"not a valid IP address: {value!r}")
        return cleaned

    @field_validator("attributes")
    @classmethod
    def _no_reserved_keys(cls, value: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
        reserved = sorted(set(value) & set(COMMON_COLUMNS))
        if reserved:
            raise ValueError(f"attribute keys shadow common columns: {reserved}")
        return value


def coerce_common_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with every common column present and correctly typed.

    Missing common columns are added as nulls. Extra (attribute) columns are kept
    after the common ones, in their original order.
    """
    out = frame.copy()
    for column in COMMON_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).astype("datetime64[us, UTC]")
    for column in _STRING_COLUMNS:
        out[column] = out[column].astype("string")
    for column in _PORT_COLUMNS:
        out[column] = out[column].astype("Int32")
    out["is_attack"] = out["is_attack"].astype("boolean")
    extras = [column for column in out.columns if column not in COMMON_COLUMNS]
    return out[[*COMMON_COLUMNS, *extras]]


def events_to_frame(events: Iterable[NormalizedEvent]) -> pd.DataFrame:
    """Convert validated events into a normalized frame (attributes become columns)."""
    rows: list[dict[str, Any]] = []
    for event in events:
        row: dict[str, Any] = event.model_dump(exclude={"attributes"})
        row["source"] = event.source.value
        row["event_type"] = event.event_type.value
        row.update(event.attributes)
        rows.append(row)
    return coerce_common_schema(pd.DataFrame(rows, columns=None if rows else list(COMMON_COLUMNS)))


def _to_python(value: Any) -> Any:
    """Convert a pandas/numpy cell to a plain Python value (nulls become ``None``)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item") and not isinstance(value, pd.Timestamp):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def frame_to_events(frame: pd.DataFrame) -> list[NormalizedEvent]:
    """Validate every row of a normalized frame. Intended for small frames."""
    events: list[NormalizedEvent] = []
    for record in frame.to_dict(orient="records"):
        common = {name: _to_python(record.get(name)) for name in COMMON_COLUMNS}
        attributes = {
            str(key): cleaned
            for key, value in record.items()
            if key not in COMMON_COLUMNS and (cleaned := _to_python(value)) is not None
        }
        events.append(NormalizedEvent(**common, attributes=attributes))
    return events
