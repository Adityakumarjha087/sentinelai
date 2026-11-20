"""Download the CICIDS2017 labelled flows with integrity verification.

The official host (unb.ca) now gates downloads behind a web form, so by default
files come from a community parquet mirror pinned to a specific commit. Every
file is checked against a SHA-256 recorded here; a mismatch aborts the download.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# file name -> SHA-256 of the parquet file at the pinned mirror commit.
CICIDS_FILES: dict[str, str] = {
    "Monday-WorkingHours.pcap_ISCX.csv.parquet": (
        "dfdcef4b8670e52af54dc4f82174834365a393473e877174cca46d17b12dfd02"
    ),
    "Tuesday-WorkingHours.pcap_ISCX.csv.parquet": (
        "27e83d518cb093faefd0f883cb4df3ad8b353f150934004f28d0e7962f9f31c4"
    ),
    "Wednesday-workingHours.pcap_ISCX.csv.parquet": (
        "d23a259820b16e1ad54f9f3b58d5727c5032d383015f90bc7c07cebbdf8a7140"
    ),
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet": (
        "d8110c04a7af91124ada1c5ad901c4210879df1af8882dc637767532e7165350"
    ),
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet": (
        "5da010354f0fc1040fd1fe65967096e1063475de8dd30ae4f657c07201d728a7"
    ),
    "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet": (
        "2c00236b13a69f4b1c222b8f4a89451dc2148cb04a8cd0c45c2a87af51471774"
    ),
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet": (
        "4d78cee297c27f1a9947b9384793e587a46c7a3ea89db199553dabddc9835d4a"
    ),
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet": (
        "7c5876d52189fc01af54bad6cf23afe9f7fbc0e3ca6c3595920754f0c3ba8f66"
    ),
}


class ChecksumMismatchError(RuntimeError):
    """A downloaded file does not match its expected SHA-256."""


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    client: httpx.Client | None = None,
    overwrite: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` atomically, verifying its checksum.

    An existing file with the right checksum is reused. The payload is streamed
    to a ``.part`` file and only renamed into place once verified, so a failed
    or tampered download never leaves a plausible-looking file behind.
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download: {url}")
    if dest.exists() and not overwrite:
        if expected_sha256 is None or sha256_of(dest) == expected_sha256:
            logger.info("already present: %s", dest.name)
            return dest
        logger.warning("checksum mismatch on existing %s; downloading again", dest.name)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0))
    try:
        with http.stream("GET", url) as response, partial.open("wb") as handle:
            response.raise_for_status()
            for block in response.iter_bytes(1 << 20):
                digest.update(block)
                handle.write(block)
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ChecksumMismatchError(
                f"{dest.name}: expected sha256 {expected_sha256}, got {digest.hexdigest()}"
            )
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
        if owns_client:
            http.close()
    logger.info("downloaded %s", dest.name)
    return dest


def download_cicids(
    dest_dir: Path,
    base_url: str,
    *,
    files: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Download every CICIDS2017 labelled-flows file into ``dest_dir``."""
    manifest = CICIDS_FILES if files is None else files
    return [
        download_file(
            f"{base_url.rstrip('/')}/{name}",
            dest_dir / name,
            expected_sha256=checksum,
            client=client,
        )
        for name, checksum in manifest.items()
    ]
