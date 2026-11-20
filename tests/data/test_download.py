from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from data.download import (
    CICIDS_FILES,
    ChecksumMismatchError,
    download_cicids,
    download_file,
    sha256_of,
)

PAYLOAD = b"PAR1 pretend parquet bytes" * 1000
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://mirror.example.com/flows/a.parquet"


def make_client(calls: list[str], status: int = 200, body: bytes = PAYLOAD) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_downloads_and_verifies(tmp_path: Path) -> None:
    calls: list[str] = []
    dest = download_file(
        URL, tmp_path / "raw" / "a.parquet", expected_sha256=DIGEST, client=make_client(calls)
    )
    assert dest.read_bytes() == PAYLOAD
    assert sha256_of(dest) == DIGEST
    assert calls == [URL]
    assert not list(dest.parent.glob("*.part"))


def test_checksum_mismatch_leaves_nothing_behind(tmp_path: Path) -> None:
    dest = tmp_path / "a.parquet"
    with pytest.raises(ChecksumMismatchError, match="expected sha256"):
        download_file(URL, dest, expected_sha256="0" * 64, client=make_client([]))
    assert list(tmp_path.iterdir()) == []


def test_http_error_leaves_nothing_behind(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        download_file(URL, tmp_path / "a.parquet", client=make_client([], status=404, body=b"nope"))
    assert list(tmp_path.iterdir()) == []


def test_valid_existing_file_is_not_downloaded_again(tmp_path: Path) -> None:
    dest = tmp_path / "a.parquet"
    dest.write_bytes(PAYLOAD)
    calls: list[str] = []
    download_file(URL, dest, expected_sha256=DIGEST, client=make_client(calls))
    assert calls == []


def test_corrupt_existing_file_is_replaced(tmp_path: Path) -> None:
    dest = tmp_path / "a.parquet"
    dest.write_bytes(b"truncated")
    calls: list[str] = []
    download_file(URL, dest, expected_sha256=DIGEST, client=make_client(calls))
    assert calls == [URL]
    assert dest.read_bytes() == PAYLOAD


def test_plain_http_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        download_file("http://mirror.example.com/a.parquet", tmp_path / "a.parquet")


def test_download_cicids_builds_urls_from_the_manifest(tmp_path: Path) -> None:
    calls: list[str] = []
    paths = download_cicids(
        tmp_path,
        "https://mirror.example.com/flows/",
        files={"a.parquet": DIGEST, "b.parquet": DIGEST},
        client=make_client(calls),
    )
    assert [path.name for path in paths] == ["a.parquet", "b.parquet"]
    assert calls == [URL, "https://mirror.example.com/flows/b.parquet"]


def test_manifest_covers_all_capture_days_with_valid_digests() -> None:
    assert len(CICIDS_FILES) == 8
    days = {name.split("-")[0] for name in CICIDS_FILES}
    assert days == {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in CICIDS_FILES.values())
