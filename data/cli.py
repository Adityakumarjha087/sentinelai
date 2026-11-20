"""Command-line entry point for the data pipeline (``sentinel-data``)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from common.settings import get_settings
from data.download import download_cicids
from data.features import build_feature_tables
from data.normalize import load_events, normalize_cicids_dir, normalize_sysmon_files
from data.synthetic import SYNTHETIC_FILE_NAME, generate_cicids_like_csv

logger = logging.getLogger("data.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel-data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("download", help="download CICIDS2017 labelled flows (~306 MB)")
    commands.add_parser("synth", help="generate a small synthetic flow file (offline demo)")

    normalize = commands.add_parser("normalize", help="normalize raw logs into the common schema")
    normalize.add_argument("--dataset", choices=("cicids2017", "synthetic"), default="cicids2017")
    normalize.add_argument("--max-rows-per-file", type=int, default=None)
    normalize.add_argument("--chunk-rows", type=int, default=100_000)

    features = commands.add_parser("features", help="build model-ready feature tables")
    features.add_argument("--dataset", choices=("cicids2017", "synthetic"), default="cicids2017")
    features.add_argument("--fit-on", default="monday", help="file-name fragment of the fit set")

    stats = commands.add_parser("stats", help="summarize normalized events")
    stats.add_argument(
        "--dataset", choices=("cicids2017", "synthetic", "sysmon"), default="cicids2017"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()
    events_dir = settings.processed_dir / "events"
    features_dir = settings.processed_dir / "features"

    if args.command == "download":
        paths = download_cicids(settings.raw_dir / "cicids2017", settings.cicids_base_url)
        logger.info("%d files ready in %s", len(paths), paths[0].parent)
    elif args.command == "synth":
        path = generate_cicids_like_csv(settings.raw_dir / "synthetic" / SYNTHETIC_FILE_NAME)
        logger.info("wrote synthetic flows to %s", path)
    elif args.command == "normalize":
        written = normalize_cicids_dir(
            settings.raw_dir / args.dataset,
            events_dir / args.dataset,
            chunk_rows=args.chunk_rows,
            max_rows_per_file=args.max_rows_per_file,
        )
        samples: list[Path] = sorted(settings.samples_dir.glob("*.xml"))
        if samples:
            count = normalize_sysmon_files(samples, events_dir / "sysmon" / "sysmon.parquet")
            logger.info("normalized %d Sysmon events", count)
        logger.info("normalized %d flow events", sum(written.values()))
    elif args.command == "features":
        featurizer = build_feature_tables(
            events_dir / args.dataset, features_dir / args.dataset, fit_on=args.fit_on
        )
        logger.info("built %d features per flow", len(featurizer.feature_names_))
    elif args.command == "stats":
        columns = ["timestamp", "event_type", "label", "is_attack"]
        frame = load_events(events_dir / args.dataset, columns=columns)
        print(f"events: {len(frame):,}")
        print(f"time range (UTC): {frame['timestamp'].min()} -> {frame['timestamp'].max()}")
        print(frame["event_type"].value_counts().to_string())
        print(frame["label"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
