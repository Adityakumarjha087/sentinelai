"""Write normalized events to parquet and read them back."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.parsers.cicids import parse_cicids_file
from data.parsers.sysmon import parse_sysmon_file
from data.schema import events_to_frame

logger = logging.getLogger(__name__)

_CICIDS_PATTERNS = ("*.parquet", "*.csv")


def write_normalized(chunks: Iterable[pd.DataFrame], out_path: Path) -> int:
    """Stream normalized chunks into one zstd-compressed parquet file.

    The file is written to a temporary name and renamed when complete, so readers
    never see a half-written table. Returns the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".part")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk in chunks:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no rows to write for {out_path.name}")
    partial.replace(out_path)
    return rows


def normalize_cicids_dir(
    raw_dir: Path,
    out_dir: Path,
    *,
    chunk_rows: int = 100_000,
    max_rows_per_file: int | None = None,
) -> dict[Path, int]:
    """Normalize every CICIDS2017 file in ``raw_dir``; returns ``{output: rows}``."""
    sources = sorted(path for pattern in _CICIDS_PATTERNS for path in raw_dir.glob(pattern))
    if not sources:
        raise FileNotFoundError(f"no CICIDS2017 .parquet or .csv files in {raw_dir}")
    written: dict[Path, int] = {}
    for source in sources:
        out_path = out_dir / (source.name.split(".")[0] + ".parquet")
        chunks = parse_cicids_file(source, chunk_rows=chunk_rows, max_rows=max_rows_per_file)
        written[out_path] = write_normalized(chunks, out_path)
        logger.info("normalized %s -> %s (%d rows)", source.name, out_path, written[out_path])
    return written


def normalize_sysmon_files(sources: Iterable[Path], out_path: Path) -> int:
    """Normalize Sysmon XML exports into a single parquet file."""
    events = [event for source in sources for event in parse_sysmon_file(source)]
    return write_normalized([events_to_frame(events)] if events else [], out_path)


def load_events(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a normalized parquet file, or every such file in a directory."""
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no parquet files in {path}")
    frames = [pd.read_parquet(file, columns=columns) for file in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
