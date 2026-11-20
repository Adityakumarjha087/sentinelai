"""Turn the trained flow detector into an alert feed.

The detector scores whole capture files, not single flows: the behavioural
features count what a source did in the same minute, so a flow has to be
featurized alongside its neighbours.

The feed is a seeded sample of the flows that crossed the threshold, in time
order - the queue an analyst would actually work. Sorting by score instead
surfaces the extreme tail, which is often benign: on the CICIDS2017 DDoS
afternoon all 500 highest-scoring flows are benign, although 79% of the flows
the model flags that day are attacks.

Ground-truth labels are kept in the detections table so the dashboard can show
how the detector did, but they never reach the analyst: :func:`flow_alert`
builds what the analyst sees, and it has no label in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from data.features import FlowFeaturizer, behavioral_features
from detection.train import LoadedModel, load_model
from guardrails.sanitize import sanitize_text

FLOW_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "event_type",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
)
GROUND_TRUTH = "ground_truth"


@dataclass(frozen=True)
class FlowDetector:
    """A saved detector plus the featurizer its features came from."""

    model: LoadedModel
    featurizer: FlowFeaturizer

    @classmethod
    def load(cls, model_dir: Path, featurizer_path: Path) -> FlowDetector:
        """Load both halves and refuse a pair that was fitted on different features."""
        model = load_model(model_dir)
        featurizer = FlowFeaturizer.load(featurizer_path)
        if featurizer.feature_names_ != model.feature_names:
            raise ValueError("the featurizer and the model were fitted on different feature lists")
        return cls(model=model, featurizer=featurizer)

    @property
    def name(self) -> str:
        return str(self.model.metadata.get("model", "unknown"))

    def score(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Identifying columns, behavioural counts, score and alert decision for every flow."""
        scores = self.model.detector.score(self.featurizer.transform(flows).to_numpy())
        scored = flows[[column for column in FLOW_COLUMNS if column in flows.columns]].copy()
        scored = pd.concat([scored, behavioral_features(flows, self.featurizer.window)], axis=1)
        scored["anomaly_score"] = scores.astype("float64")
        scored["is_anomalous"] = scores >= self.model.threshold
        if "label" in flows.columns:
            scored[GROUND_TRUTH] = flows["label"]
        if "is_attack" in flows.columns:
            scored["is_attack"] = flows["is_attack"]
        return scored


@dataclass(frozen=True)
class DetectionFeed:
    """Flagged flows for analysts, plus how many were scored and flagged."""

    detections: pd.DataFrame
    summary: dict[str, Any]


def read_flows(path: Path) -> pd.DataFrame:
    """A normalized flow file without its ``raw`` column, which scoring never needs."""
    columns = [name for name in pq.ParquetFile(path).schema_arrow.names if name != "raw"]
    return pd.read_parquet(path, columns=columns)


def build_feed(
    detector: FlowDetector, flows: pd.DataFrame, *, source: str, max_alerts: int = 500, seed: int = 0
) -> DetectionFeed:
    """Score ``flows`` and keep a seeded sample of at most ``max_alerts`` flagged ones, oldest first."""
    scored = detector.score(flows)
    flagged = scored[scored["is_anomalous"]]
    sample = flagged.sample(n=min(max_alerts, len(flagged)), random_state=seed).sort_values("timestamp", kind="stable")
    summary: dict[str, Any] = {
        "detector": detector.name,
        "threshold": detector.model.threshold,
        "source": source,
        "flows_scored": len(scored),
        "flows_flagged": len(flagged),
    }
    if "is_attack" in flagged.columns and flagged["is_attack"].notna().any():
        summary["flagged_attack_share"] = round(float(flagged["is_attack"].astype("float64").mean()), 4)
    detections = sample.drop(columns=["is_anomalous", "is_attack"], errors="ignore").assign(
        detector=detector.name, threshold=detector.model.threshold
    )
    return DetectionFeed(detections=detections.reset_index(drop=True), summary=summary)


def describe_behaviour(row: pd.Series) -> str:
    """The behavioural counts behind a flow, as one sentence for the analyst."""
    return (
        f"in the same minute the source opened {int(row['src_flow_count'])} flows to "
        f"{int(row['src_unique_dst_ports'])} distinct destination ports on {int(row['src_unique_dst_ips'])} "
        f"destination hosts, and the destination received {int(row['dst_flow_count'])} flows from "
        f"{int(row['dst_unique_src_ips'])} sources"
    )


def flow_alert(row: pd.Series) -> dict[str, Any]:
    """What the analyst sees for one detection. Deliberately has no ground-truth label."""
    alert: dict[str, Any] = {
        column: sanitize_text(row[column], 100) for column in FLOW_COLUMNS if column in row and not pd.isna(row[column])
    }
    alert["detector"] = str(row["detector"])
    alert["anomaly_score"] = round(float(row["anomaly_score"]), 4)
    alert["threshold"] = round(float(row["threshold"]), 4)
    alert["summary"] = describe_behaviour(row)
    return alert
