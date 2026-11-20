from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from data.parsers.sysmon import MAX_ATTRIBUTE_CHARS, parse_sysmon_file, parse_sysmon_xml
from data.schema import MAX_RAW_CHARS, EventSource, EventType

NS = 'xmlns="http://schemas.microsoft.com/win/2004/08/events/event"'


def one_event(code: str = "1", time: str = "2024-03-12T09:00:00.1234567Z", data: str = "") -> str:
    return (
        f"<Event {NS}><System><EventID>{code}</EventID>"
        f'<TimeCreated SystemTime="{time}"/><EventRecordID>7</EventRecordID>'
        f"<Computer>HOST-1</Computer></System><EventData>{data}</EventData></Event>"
    )


def test_sample_file_parses_into_expected_events(sysmon_sample: Path) -> None:
    events = parse_sysmon_file(sysmon_sample)
    assert len(events) == 13
    assert len({event.event_id for event in events}) == 13
    assert all(event.source is EventSource.SYSMON for event in events)
    assert all(event.host == "WS-FIN-012.corp.example.com" for event in events)
    assert [event.timestamp for event in events] == sorted(event.timestamp for event in events)
    counts = {kind: sum(event.event_type is kind for event in events) for kind in EventType}
    assert counts[EventType.PROCESS_CREATE] == 4
    assert counts[EventType.NETWORK_CONNECTION] == 3
    assert counts[EventType.DNS_QUERY] == 2
    assert counts[EventType.REGISTRY_EVENT] == 1
    assert counts[EventType.PROCESS_ACCESS] == 1


def test_fields_that_collide_with_common_columns_are_prefixed(sysmon_sample: Path) -> None:
    registry = next(event for event in parse_sysmon_file(sysmon_sample) if event.event_type is EventType.REGISTRY_EVENT)
    assert registry.attributes["sysmon_event_type"] == "SetValue"
    assert "CurrentVersion\\Run" in str(registry.attributes["target_object"])


def test_process_create_fields(sysmon_sample: Path) -> None:
    powershell = next(
        event
        for event in parse_sysmon_file(sysmon_sample)
        if event.event_type is EventType.PROCESS_CREATE and "powershell" in (event.process or "")
    )
    assert powershell.user == "CORP\\j.rivera"
    assert powershell.timestamp == datetime(2024, 3, 12, 9, 14, 21, 550100, tzinfo=UTC)
    assert powershell.attributes["event_code"] == 1
    assert str(powershell.attributes["parent_image"]).endswith("WINWORD.EXE")
    assert "-EncodedCommand" in str(powershell.attributes["command_line"])
    assert powershell.src_ip is None


def test_network_connection_fields(sysmon_sample: Path) -> None:
    beacon = [event for event in parse_sysmon_file(sysmon_sample) if event.event_type is EventType.NETWORK_CONNECTION][
        -1
    ]
    assert (beacon.src_ip, beacon.src_port) == ("10.20.4.112", 51902)
    assert (beacon.dst_ip, beacon.dst_port, beacon.protocol) == ("198.51.100.23", 8443, "tcp")


def test_process_access_uses_source_image(sysmon_sample: Path) -> None:
    access = next(event for event in parse_sysmon_file(sysmon_sample) if event.event_type is EventType.PROCESS_ACCESS)
    assert (access.process or "").endswith("svchost_update.exe")
    assert str(access.attributes["target_image"]).endswith("lsass.exe")


def test_injection_text_is_preserved_as_inert_data(sysmon_sample: Path) -> None:
    """The parser must neither act on nor silently drop hostile text."""
    hostile = [
        event
        for event in parse_sysmon_file(sysmon_sample)
        if "ignore all previous instructions" in str(event.attributes.get("command_line", ""))
    ]
    assert len(hostile) == 1
    assert hostile[0].event_type is EventType.PROCESS_CREATE
    assert hostile[0].is_attack is None  # a log line cannot set its own verdict


def test_single_event_without_namespace_and_unknown_code() -> None:
    text = one_event(code="255").replace(f" {NS}", "")
    (event,) = parse_sysmon_xml(text)
    assert event.event_type is EventType.OTHER
    assert event.attributes == {"event_code": 255}


def test_falls_back_to_utc_time_field() -> None:
    text = one_event(time="", data='<Data Name="UtcTime">2024-03-12 10:11:12.500</Data>')
    (event,) = parse_sysmon_xml(text)
    assert event.timestamp == datetime(2024, 3, 12, 10, 11, 12, 500000, tzinfo=UTC)


def test_malformed_events_are_skipped() -> None:
    good, undated, no_code = one_event(), one_event(time="not-a-time"), one_event(code="abc")
    events = parse_sysmon_xml(f"<Events>{good}{undated}{no_code}<Other/></Events>")
    assert len(events) == 1


def test_bad_addresses_and_ports_become_null() -> None:
    data = (
        '<Data Name="SourceIp">not-an-ip</Data><Data Name="DestinationIp">-</Data>'
        '<Data Name="SourcePort">99999</Data><Data Name="DestinationPort">https</Data>'
    )
    (event,) = parse_sysmon_xml(one_event(code="3", data=data))
    assert (event.src_ip, event.dst_ip, event.src_port, event.dst_port) == (None, None, None, None)


def test_oversized_fields_are_truncated() -> None:
    data = f'<Data Name="CommandLine">{"A" * 50_000}</Data>'
    (event,) = parse_sysmon_xml(one_event(data=data))
    assert len(str(event.attributes["command_line"])) == MAX_ATTRIBUTE_CHARS
    assert len(event.raw) == MAX_RAW_CHARS


@pytest.mark.parametrize(
    "payload",
    [
        "<Events><Event>",  # truncated document
        # billion-laughs style entity expansion
        '<!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><Events>&b;</Events>',
        # external entity (XXE) file read
        '<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><Events>&xxe;</Events>',
    ],
)
def test_dangerous_or_broken_xml_is_rejected(payload: str) -> None:
    with pytest.raises(ValueError, match="rejected Sysmon XML"):
        parse_sysmon_xml(payload)
