from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.features import (
    META_COLUMNS,
    FlowFeaturizer,
    behavioral_features,
    build_feature_tables,
    port_protocol_features,
    time_features,
)
from data.normalize import write_normalized


def tiny_flows() -> pd.DataFrame:
    """Four flows in one minute from a scanner, one from a normal client, one a minute later."""
    stamp = pd.Timestamp("2017-07-07 16:05:10", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [stamp] * 5 + [stamp + pd.Timedelta(minutes=1)],
            "src_ip": ["172.16.0.1"] * 4 + ["192.168.10.5", "172.16.0.1"],
            "dst_ip": ["192.168.10.50"] * 6,
            "src_port": pd.array([40000, 40001, 40002, 40003, 55000, 80], dtype="Int32"),
            "dst_port": pd.array([21, 22, 23, 23, 443, 60000], dtype="Int32"),
            "protocol": ["tcp", "tcp", "tcp", "tcp", "udp", "tcp"],
        }
    )


def test_behavioral_features_count_within_the_window() -> None:
    features = behavioral_features(tiny_flows())
    assert features["src_flow_count"].tolist() == [4, 4, 4, 4, 1, 1]
    assert features["src_unique_dst_ports"].tolist() == [3, 3, 3, 3, 1, 1]
    assert features["src_unique_dst_ips"].tolist() == [1, 1, 1, 1, 1, 1]
    assert features["dst_flow_count"].tolist() == [5, 5, 5, 5, 5, 1]
    assert features["dst_unique_src_ips"].tolist() == [2, 2, 2, 2, 2, 1]


def test_port_and_protocol_features() -> None:
    features = port_protocol_features(tiny_flows())
    assert features["dst_port_well_known"].tolist() == [1, 1, 1, 1, 1, 0]
    assert features["dst_port_ephemeral"].tolist() == [0, 0, 0, 0, 0, 1]
    assert features["src_port_well_known"].tolist() == [0, 0, 0, 0, 0, 1]
    assert features["proto_udp"].tolist() == [0, 0, 0, 0, 1, 0]


def test_time_features_are_on_the_unit_circle() -> None:
    features = time_features(tiny_flows())
    assert np.allclose(features["hour_sin"] ** 2 + features["hour_cos"] ** 2, 1.0)


def test_transform_is_finite_and_complete(flows: pd.DataFrame) -> None:
    assert np.isinf(flows["flow_bytes_s"]).any()  # the input really is dirty
    features = FlowFeaturizer().fit_transform(flows)
    assert features.shape[0] == len(flows)
    assert np.isfinite(features.to_numpy()).all()
    assert set(features.dtypes.astype(str)) == {"float32"}


def test_identifiers_and_clock_are_not_features(flows: pd.DataFrame) -> None:
    names = set(FlowFeaturizer().fit(flows).feature_names_)
    assert not names & {"src_ip", "dst_ip", "src_port", "dst_port", "label", "is_attack"}
    assert not names & {"hour_sin", "hour_cos"}
    assert {"hour_sin", "hour_cos"} <= set(FlowFeaturizer(include_time=True).fit(flows).feature_names_)


def test_scan_stands_out_in_behavioral_features(flows: pd.DataFrame) -> None:
    features = FlowFeaturizer().fit_transform(flows)
    scan = (flows["label"] == "PortScan").to_numpy()
    benign = (flows["label"] == "BENIGN").to_numpy()
    assert features.loc[scan, "src_unique_dst_ports"].median() > features.loc[benign, "src_unique_dst_ports"].max()


def test_heavy_tailed_columns_are_log_scaled(flows: pd.DataFrame) -> None:
    featurizer = FlowFeaturizer().fit(flows)
    assert "flow_duration" in featurizer.log_columns_
    assert "proto_tcp" not in featurizer.log_columns_
    features = featurizer.transform(flows)
    assert features["flow_duration"].max() < 40  # log1p of microseconds, not raw
    assert features["init_win_bytes_forward"].min() < 0  # signed log keeps the -1 sentinel


def test_medians_come_from_fit_data_only(flows: pd.DataFrame) -> None:
    benign = flows[flows["label"] == "BENIGN"]
    featurizer = FlowFeaturizer().fit(benign)
    expected = benign["flow_bytes_s"].replace([np.inf, -np.inf], np.nan).median()
    assert featurizer.medians_["flow_bytes_s"] == pytest.approx(expected)


def test_unseen_and_missing_columns_are_handled(flows: pd.DataFrame) -> None:
    featurizer = FlowFeaturizer().fit(flows)
    changed = flows.drop(columns=["syn_flag_count"]).assign(brand_new_stat=1.0)
    features = featurizer.transform(changed)
    assert list(features.columns) == featurizer.feature_names_
    assert (features["syn_flag_count"] == featurizer.medians_["syn_flag_count"]).all()


def test_save_and_load_round_trip(flows: pd.DataFrame, tmp_path: Path) -> None:
    featurizer = FlowFeaturizer().fit(flows)
    featurizer.save(tmp_path / "featurizer.json")
    restored = FlowFeaturizer.load(tmp_path / "featurizer.json")
    pd.testing.assert_frame_equal(restored.transform(flows), featurizer.transform(flows))


def test_unfitted_and_malformed_inputs_raise(flows: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        FlowFeaturizer().transform(flows)
    with pytest.raises(ValueError, match="missing required columns"):
        FlowFeaturizer().fit(flows.drop(columns=["src_ip"]))


def test_build_feature_tables_fits_on_the_baseline_file(flows: pd.DataFrame, tmp_path: Path) -> None:
    events, out = tmp_path / "events", tmp_path / "features"
    benign = flows[flows["label"] == "BENIGN"]
    write_normalized([benign], events / "Monday-WorkingHours.parquet")
    write_normalized([flows], events / "Friday-WorkingHours.parquet")

    featurizer = build_feature_tables(events, out)

    baseline = FlowFeaturizer().fit(benign)
    assert featurizer.medians_ == baseline.medians_
    table = pd.read_parquet(out / "Friday-WorkingHours.parquet")
    assert list(table.columns) == [*META_COLUMNS, *featurizer.feature_names_]
    assert len(table) == len(flows)
    assert FlowFeaturizer.load(out / "featurizer.json").feature_names_ == featurizer.feature_names_
