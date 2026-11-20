from __future__ import annotations

import json

import pandas as pd
import pytest

from agent.investigator import investigate
from agent.tools import ToolBox
from data.features import FlowFeaturizer
from detection.service import GROUND_TRUTH, FlowDetector, build_feed, describe_behaviour, flow_alert, read_flows
from eval.analysts import RuleBasedAnalyst
from tests.conftest import FlowArtifacts


@pytest.fixture(scope="module")
def detector(flow_artifacts: FlowArtifacts) -> FlowDetector:
    return FlowDetector.load(flow_artifacts.model_dir, flow_artifacts.featurizer_path)


def test_scores_every_flow_and_flags_attacks_far_more_than_benign(detector: FlowDetector, flows: pd.DataFrame) -> None:
    scored = detector.score(flows)
    assert len(scored) == len(flows)
    assert {"anomaly_score", "is_anomalous", "src_unique_dst_ports", GROUND_TRUTH}.issubset(scored.columns)
    flag_rate = scored.groupby(scored[GROUND_TRUTH] == "BENIGN")["is_anomalous"].mean()
    assert flag_rate[False] > 0.8
    assert flag_rate[True] < 0.2


def test_refuses_a_featurizer_from_a_different_feature_list(
    flow_artifacts: FlowArtifacts, flows: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> None:
    other = FlowFeaturizer().fit(flows.drop(columns=["flow_duration"]))
    path = tmp_path_factory.mktemp("other") / "featurizer.json"
    other.save(path)
    with pytest.raises(ValueError, match="different feature lists"):
        FlowDetector.load(flow_artifacts.model_dir, path)


def test_feed_is_a_seeded_time_ordered_sample_of_flagged_flows(detector: FlowDetector, flows: pd.DataFrame) -> None:
    feed = build_feed(detector, flows, source="synthetic.parquet", max_alerts=25, seed=3)
    detections = feed.detections
    assert len(detections) == 25
    assert detections["timestamp"].is_monotonic_increasing
    assert (detections["anomaly_score"] >= detections["threshold"]).all()
    assert {"is_anomalous", "is_attack"}.isdisjoint(detections.columns)
    summary = feed.summary
    assert summary["source"] == "synthetic.parquet"
    assert summary["flows_scored"] == len(flows)
    assert 25 <= summary["flows_flagged"] < len(flows)
    assert 0.8 < summary["flagged_attack_share"] <= 1.0
    again = build_feed(detector, flows, source="synthetic.parquet", max_alerts=25, seed=3)
    assert again.detections["event_id"].tolist() == detections["event_id"].tolist()


def test_feed_keeps_everything_when_fewer_flows_are_flagged_than_the_cap(
    detector: FlowDetector, flows: pd.DataFrame
) -> None:
    feed = build_feed(detector, flows, source="x", max_alerts=10_000)
    assert len(feed.detections) == feed.summary["flows_flagged"]


def test_flow_alert_carries_behaviour_but_never_the_label(detector: FlowDetector, flows: pd.DataFrame) -> None:
    detections = build_feed(detector, flows, source="x").detections
    scan = detections[detections[GROUND_TRUTH] == "PortScan"].iloc[0]
    alert = flow_alert(scan)
    assert GROUND_TRUTH not in alert
    assert "PortScan" not in json.dumps(alert)
    assert alert["event_type"] == "network_flow"
    assert alert["anomaly_score"] >= alert["threshold"]
    assert "120 distinct destination ports" in alert["summary"]
    assert alert["summary"] == describe_behaviour(scan)


def test_rule_baseline_maps_a_detected_scan_but_not_a_single_port_flow(
    detector: FlowDetector, flows: pd.DataFrame
) -> None:
    scored = detector.score(flows).assign(detector="isolation_forest", threshold=detector.model.threshold)
    scan = flow_alert(scored[scored[GROUND_TRUTH] == "PortScan"].iloc[0])
    single = flow_alert(scored[scored["src_unique_dst_ports"] == 1].iloc[0])
    tools = ToolBox(flows)
    assert investigate(scan, RuleBasedAnalyst(), tools).verdict.attack_techniques == ["T1046"]
    assert "1 distinct destination ports" in single["summary"]
    assert investigate(single, RuleBasedAnalyst(), tools).verdict.is_malicious is False


def test_read_flows_skips_the_raw_column(flow_artifacts: FlowArtifacts) -> None:
    frame = read_flows(flow_artifacts.flows_path)
    assert "raw" not in frame.columns
    assert "src_ip" in frame.columns
