"""Generate a small synthetic flow file in the original CICIDS2017 CSV format.

For offline demos, CI and tests only - it is *not* real traffic and must never
be used to report detection metrics. It deliberately reproduces the format
defects of the original CSVs (see :mod:`data.parsers.cicids`) so the whole
pipeline can be exercised without the 300 MB download.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SYNTHETIC_FILE_NAME = "Synthetic-WorkingHours.pcap_ISCX.csv"
_HEADER = [
    "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port",
    " Protocol", " Timestamp", " Flow Duration", " Total Fwd Packets",
    " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets",
    "Flow Bytes/s", " Flow Packets/s", " Fwd Header Length", " SYN Flag Count",
    " ACK Flag Count", " Fwd Header Length", "Init_Win_bytes_forward", " Label",
]  # fmt: skip
_DAY = datetime(2017, 7, 7)
_ATTACKER = "172.16.0.1"
_VICTIM = "192.168.10.50"


def _stamp(moment: datetime) -> str:
    """Day-first, 12-hour clock, no AM/PM, no zero padding - like the original."""
    hour = moment.hour - 12 if moment.hour > 12 else moment.hour
    return f"{moment.day}/{moment.month}/{moment.year} {hour}:{moment.minute:02d}"


def _row(
    src: str, sport: int, dst: str, dport: int, proto: int, moment: datetime,
    duration: int, fwd: int, bwd: int, fwd_bytes: int, bwd_bytes: int, syn: int, label: str,
) -> list[object]:  # fmt: skip
    seconds = duration / 1e6
    byte_rate = (fwd_bytes + bwd_bytes) / seconds if duration else "Infinity"
    packet_rate = (fwd + bwd) / seconds if duration else "Infinity"
    header = 20 * fwd
    return [
        f"{src}-{dst}-{sport}-{dport}-{proto}", src, sport, dst, dport, proto, _stamp(moment),
        duration, fwd, bwd, fwd_bytes, bwd_bytes, byte_rate, packet_rate, header, syn,
        int(bwd > 0), header, 8192 if proto == 6 else -1, label,
    ]  # fmt: skip


def generate_cicids_like_csv(
    path: Path, n_benign: int = 2_000, n_scan: int = 300, n_ddos: int = 300, seed: int = 7
) -> Path:
    """Write a synthetic labelled-flows CSV to ``path`` and return it."""
    rng = np.random.default_rng(seed)
    timed: list[tuple[datetime, list[object]]] = []

    for _ in range(n_benign):
        moment = _DAY + timedelta(seconds=int(rng.integers(9 * 3600, 17 * 3600)))
        dport, proto = [(443, 6), (80, 6), (53, 17), (22, 6)][
            int(rng.choice(4, p=[0.5, 0.2, 0.25, 0.05]))
        ]
        fwd, bwd = int(rng.integers(1, 40)), int(rng.integers(1, 60))
        timed.append((moment,
            _row(
                f"192.168.10.{int(rng.integers(5, 26))}", int(rng.integers(49152, 65535)),
                f"192.168.10.{int(rng.integers(3, 5))}" if dport == 53 else _VICTIM, dport, proto,
                moment, int(rng.lognormal(11, 2)), fwd, bwd,
                fwd * int(rng.integers(40, 600)), bwd * int(rng.integers(40, 1400)), 1, "BENIGN",
            )
        ))  # fmt: skip

    scan_start = _DAY + timedelta(hours=13, minutes=5)
    for i in range(n_scan):  # one source sweeping many ports; some flows have zero duration
        moment = scan_start + timedelta(seconds=i * 0.4)
        timed.append((moment,
            _row(_ATTACKER, int(rng.integers(32768, 61000)), _VICTIM, 1 + i * 37 % 65000, 6,
                 moment, 0 if i % 10 == 0 else int(rng.integers(1, 80)), 1,
                 int(rng.integers(0, 2)), 0, 0, 1, "PortScan")
        ))  # fmt: skip

    ddos_start = _DAY + timedelta(hours=15, minutes=56)
    for i in range(n_ddos):  # many short HTTP floods at one server
        moment = ddos_start + timedelta(seconds=i * 0.2)
        timed.append((moment,
            _row(_ATTACKER, int(rng.integers(32768, 61000)), _VICTIM, 80, 6, moment,
                 int(rng.integers(1_000, 90_000)), int(rng.integers(3, 8)), int(rng.integers(0, 6)),
                 int(rng.integers(20, 60)), int(rng.integers(0, 11_000)), 1, "DDoS")
        ))  # fmt: skip

    timed.sort(key=lambda item: item[0])
    rows = [row for _, row in timed]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="cp1252") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerows(rows)
        writer.writerows([[""] * len(_HEADER)] * 5)  # the original has trailing empty rows
    return path
