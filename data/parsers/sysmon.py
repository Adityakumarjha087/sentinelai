"""Parser for Sysmon events exported as Windows Event XML.

Accepts a single ``<Event>`` or any wrapper element containing many (as produced
by ``wevtutil qe Microsoft-Windows-Sysmon/Operational /f:xml /e:Events``).

Log content is untrusted: XML is parsed with ``defusedxml`` (no entity
expansion, no external entities) and all text is length-capped.
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from data.schema import (
    COMMON_COLUMNS,
    MAX_RAW_CHARS,
    AttributeValue,
    EventSource,
    EventType,
    NormalizedEvent,
    clean_ip,
)

logger = logging.getLogger(__name__)

MAX_ATTRIBUTE_CHARS = 4_096
_EVENT_NAMESPACE = "http://schemas.microsoft.com/win/2004/08/events/event"
ET.register_namespace("", _EVENT_NAMESPACE)

EVENT_TYPES: dict[int, EventType] = {
    1: EventType.PROCESS_CREATE,
    3: EventType.NETWORK_CONNECTION,
    5: EventType.PROCESS_TERMINATE,
    7: EventType.IMAGE_LOAD,
    8: EventType.CREATE_REMOTE_THREAD,
    10: EventType.PROCESS_ACCESS,
    11: EventType.FILE_CREATE,
    12: EventType.REGISTRY_EVENT,
    13: EventType.REGISTRY_EVENT,
    14: EventType.REGISTRY_EVENT,
    22: EventType.DNS_QUERY,
}
# Fields promoted to common columns; everything else lands in ``attributes``.
_PROMOTED = {
    "SourceIp",
    "DestinationIp",
    "SourcePort",
    "DestinationPort",
    "Protocol",
    "User",
    "Image",
}


def _local(tag: str) -> str:
    """Element name without its XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local(child.tag) == name), None)


def _snake_case(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _parse_port(value: str | None) -> int | None:
    if value is None or not value.strip().isdigit():
        return None
    port = int(value)
    return port if 0 <= port <= 65535 else None


def _parse_event(event: ET.Element) -> NormalizedEvent | None:
    system = _child(event, "System")
    if system is None:
        return None
    event_id_node = _child(system, "EventID")
    time_node = _child(system, "TimeCreated")
    computer_node = _child(system, "Computer")
    record_node = _child(system, "EventRecordID")
    try:
        code = int((event_id_node.text or "").strip()) if event_id_node is not None else None
    except ValueError:
        code = None

    data: dict[str, str] = {}
    event_data = _child(event, "EventData")
    for node in event_data if event_data is not None else ():
        name = node.get("Name")
        if name and node.text and node.text.strip() not in {"", "-"}:
            data[name] = node.text.strip()

    timestamp = _parse_time(time_node.get("SystemTime") if time_node is not None else None)
    timestamp = timestamp or _parse_time(data.get("UtcTime"))
    if code is None or timestamp is None:
        return None

    raw = ET.tostring(event, encoding="unicode")[:MAX_RAW_CHARS]
    host = (computer_node.text or "").strip() or None if computer_node is not None else None
    record_id = (record_node.text or "").strip() if record_node is not None else ""
    digest = hashlib.sha256(f"{host}|{record_id}|{raw}".encode()).hexdigest()[:16]

    attributes: dict[str, AttributeValue] = {"event_code": code}
    for name, value in data.items():
        if name not in _PROMOTED:
            key = _snake_case(name)
            # e.g. the registry "EventType" field would shadow the common column.
            key = f"sysmon_{key}" if key in COMMON_COLUMNS else key
            attributes[key] = value[:MAX_ATTRIBUTE_CHARS]

    return NormalizedEvent(
        event_id=f"sysmon-{digest}",
        timestamp=timestamp,
        source=EventSource.SYSMON,
        event_type=EVENT_TYPES.get(code, EventType.OTHER),
        src_ip=clean_ip(data.get("SourceIp")),
        dst_ip=clean_ip(data.get("DestinationIp")),
        src_port=_parse_port(data.get("SourcePort")),
        dst_port=_parse_port(data.get("DestinationPort")),
        protocol=data["Protocol"].lower() if "Protocol" in data else None,
        host=host,
        process=(data.get("Image") or data.get("SourceImage") or "")[:MAX_ATTRIBUTE_CHARS] or None,
        user=(data.get("User") or "")[:MAX_ATTRIBUTE_CHARS] or None,
        raw=raw,
        attributes=attributes,
    )


def parse_sysmon_xml(text: str) -> list[NormalizedEvent]:
    """Parse Sysmon event XML into normalized events.

    Events without an event id or a usable timestamp are skipped. Malformed or
    dangerous XML (entity expansion, external entities) raises ``ValueError``.
    """
    try:
        root = SafeET.fromstring(text)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise ValueError(f"rejected Sysmon XML: {exc!r}") from exc
    nodes = [root] if _local(root.tag) == "Event" else [n for n in root if _local(n.tag) == "Event"]
    events = [event for node in nodes if (event := _parse_event(node)) is not None]
    if len(events) < len(nodes):
        logger.warning("skipped %d malformed Sysmon events", len(nodes) - len(events))
    return events


def parse_sysmon_file(path: Path) -> list[NormalizedEvent]:
    """Parse a Sysmon XML export from disk."""
    return parse_sysmon_xml(path.read_text(encoding="utf-8", errors="replace"))
