"""Parser for CICIDS2017 labelled flows.

Accepts both distributions of the "TrafficLabelling" data:

* the original CSVs from the Canadian Institute for Cybersecurity, and
* the parquet mirror used by :mod:`data.download`.

The original CSVs have several well-known defects this parser absorbs:

* header names carry leading spaces and one column is duplicated;
* files are cp1252 encoded (the web-attack labels contain an en dash);
* timestamps are local time (Fredericton, Canada), day-first, on a 12-hour
  clock *without* an AM/PM marker, with or without seconds;
* one file ends with hundreds of thousands of empty rows;
* rate columns contain ``Infinity`` and ``NaN``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data.schema import EventSource, EventType, coerce_common_schema

logger = logging.getLogger(__name__)

CAPTURE_TIMEZONE = "America/Moncton"
# Capture ran during working hours only, so an hour below this is an afternoon
# hour written on a 12-hour clock (e.g. "1:05" means 13:05).
_EARLIEST_CAPTURE_HOUR = 8
BENIGN_LABEL = "BENIGN"

_ID_COLUMNS = {
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
}
_REQUIRED = ("Timestamp", "Label", "Protocol", *_ID_COLUMNS)
_NON_FEATURE = {"Flow ID", *_REQUIRED}
_PROTOCOLS = {1: "icmp", 6: "tcp", 17: "udp"}
_FILE_SUFFIX = re.compile(r"\.pcap_iscx.*$", re.IGNORECASE)


def snake_case(name: str) -> str:
    """``'Flow Bytes/s'`` -> ``'flow_bytes_s'``."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def file_tag(path: Path) -> str:
    """Short, stable identifier for a dataset file, used inside event ids."""
    return snake_case(_FILE_SUFFIX.sub("", path.name)).replace("_", "-")


def parse_local_timestamps(values: pd.Series) -> pd.Series:
    """Parse the original CSV timestamp strings into UTC.

    Unparseable values become ``NaT``.
    """
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    without_seconds = pd.to_datetime(text, format="%d/%m/%Y %H:%M", errors="coerce")
    parsed = parsed.fillna(without_seconds)
    afternoon = parsed.dt.hour < _EARLIEST_CAPTURE_HOUR
    parsed = parsed.where(~afternoon, parsed + pd.Timedelta(hours=12))
    return parsed.dt.tz_localize(CAPTURE_TIMEZONE).dt.tz_convert("UTC")


def normalize_label(values: pd.Series) -> pd.Series:
    """Trim labels and replace the mis-encoded dash in the web-attack labels."""
    text = values.astype("string").str.strip()
    text = text.str.replace(r"\s*[\x96\u2013\ufffd]\s*", " - ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True)


def normalize_cicids_frame(frame: pd.DataFrame, tag: str, row_offset: int = 0) -> pd.DataFrame:
    """Normalize one chunk of CICIDS2017 rows into the common schema.

    Args:
        frame: Rows exactly as read from the source file.
        tag: Identifier of the source file (see :func:`file_tag`).
        row_offset: Position of the chunk's first row within the file, so event
            ids stay unique and stable across chunks.

    Returns:
        A normalized frame: common columns followed by one float column per flow
        statistic (snake_case names). Empty and undated rows are dropped.
    """
    frame = frame.rename(columns=lambda name: str(name).strip())
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(
            f"not a CICIDS2017 labelled-flows file, missing columns: {missing}. "
            "The 'MachineLearningCVE' variant has no IPs or timestamps; "
            "use the 'TrafficLabelling' / GeneratedLabelledFlows variant."
        )
    # pandas names a repeated header "<name>.1"; the copy carries no new information.
    frame = frame.loc[:, [c for c in frame.columns if not c.endswith(".1")]]
    frame = frame.reset_index(drop=True)
    row_number = pd.Series(np.arange(row_offset, row_offset + len(frame)), index=frame.index)

    if pd.api.types.is_datetime64_any_dtype(frame["Timestamp"]):
        timestamp = pd.to_datetime(frame["Timestamp"], utc=True)  # mirror: already UTC
    else:
        timestamp = parse_local_timestamps(frame["Timestamp"])

    keep = frame["Label"].notna() & timestamp.notna()
    dropped = int((~keep).sum())
    if dropped:
        logger.info("%s: dropped %d empty or undated rows", tag, dropped)
    frame, timestamp, row_number = frame[keep], timestamp[keep], row_number[keep]

    label = normalize_label(frame["Label"])
    protocol = pd.to_numeric(frame["Protocol"], errors="coerce")
    out = pd.DataFrame(
        {
            "event_id": f"cic17-{tag}-" + row_number.astype(str).str.zfill(7),
            "timestamp": timestamp,
            "source": EventSource.CICIDS2017.value,
            "event_type": EventType.NETWORK_FLOW.value,
            "protocol": protocol.map(_PROTOCOLS).fillna("other"),
            "label": label,
            "is_attack": label.str.upper() != BENIGN_LABEL,
            # The record as received, re-serialized in the source column order.
            "raw": frame.to_csv(header=False, index=False, lineterminator="\n").split("\n")[:-1],
        },
        index=frame.index,
    )
    for source_name, target in _ID_COLUMNS.items():
        if target.endswith("_port"):
            out[target] = pd.to_numeric(frame[source_name], errors="coerce").round()
        else:
            out[target] = frame[source_name].astype("string").str.strip()

    features = {
        snake_case(column): pd.to_numeric(frame[column], errors="coerce").astype("float64")
        for column in frame.columns
        if column not in _NON_FEATURE
    }
    out = pd.concat([out, pd.DataFrame(features, index=frame.index)], axis=1)
    return coerce_common_schema(out).reset_index(drop=True)


def _read_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    if path.suffix.lower() == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows):
            yield batch.to_pandas()
    else:
        # Read every cell as text: type inference per chunk would make ``raw`` (and
        # integer-vs-float formatting) depend on the chunk size.
        yield from pd.read_csv(
            path, chunksize=chunk_rows, encoding="cp1252", encoding_errors="replace", dtype=str
        )


def parse_cicids_file(
    path: Path, chunk_rows: int = 100_000, max_rows: int | None = None
) -> Iterator[pd.DataFrame]:
    """Yield normalized chunks of a CICIDS2017 CSV or parquet file.

    Memory use is bounded by ``chunk_rows``. ``max_rows`` stops early after that
    many *source* rows, which is useful for quick experiments.
    """
    tag = file_tag(path)
    offset = 0
    for chunk in _read_chunks(path, chunk_rows):
        if max_rows is not None:
            chunk = chunk.iloc[: max(max_rows - offset, 0)]
        if chunk.empty:
            break
        normalized = normalize_cicids_frame(chunk, tag, row_offset=offset)
        offset += len(chunk)
        if not normalized.empty:
            yield normalized
