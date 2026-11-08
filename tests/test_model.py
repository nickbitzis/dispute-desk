import numpy as np
import pandas as pd
import pytest

from disputedesk import generator, model
from disputedesk.features import FEATURES, build_features


@pytest.fixture(scope="module")
def small():
    tx, d = generator.generate(n_txns=60_000, n_customers=18_000, seed=5)
    return tx, d


def test_time_split_is_ordered_and_disjoint(small):
    _, d = small
    m = model.split(d)
    r = pd.to_datetime(d["received_date"])
    assert not (m["train"] & m["calib"]).any() and not (m["calib"] & m["test"]).any()
    assert (m["train"] | m["calib"] | m["test"]).all()
    assert r[m["train"]].max() < r[m["calib"]].min() <= r[m["calib"]].max() < r[m["test"]].min()


def test_features_have_no_future_leakage(small):
    """Features for disputes before a cutoff must not change when everything after the cutoff is removed."""
    tx, d = small
    full = build_features(tx, d)
    cutoff = pd.Timestamp("2026-05-01")
    early = pd.to_datetime(d["received_date"]) < cutoff
    tx_cut = tx[tx["ts"] < cutoff].copy()
    for c in ["delivered_ts", "scan_ts", "refund_ts", "cs_ts", "cancel_ts"]:
        tx_cut.loc[tx_cut[c] >= cutoff, c] = pd.NaT  # the future did not happen yet
    d_cut = d[early].reset_index(drop=True)
    cut = build_features(tx_cut, d_cut)
    pd.testing.assert_frame_equal(full[early.to_numpy()].reset_index(drop=True), cut, check_dtype=False)


def test_feature_list_excludes_hidden_truth():
    assert not {"p_true", "won_if_fought", "is_fraud", "friendly", "pre_arb"} & set(FEATURES)


def test_model_is_calibrated_and_discriminates(built):
    _, report = built
    t = report["model"]["test"]
    # Reduced data (about 450 test disputes). The full-size run lands near 0.02 ECE; see the README.
    assert t["auc"] > 0.75
    assert t["ece"] < 0.12
    assert t["ece"] < t["ece_uncalibrated"]
    assert t["brier"] < 0.22


def test_calibration_improves_fit_on_calibration_months(built):
    _, report = built
    c = report["model"]["calib"]
    assert c["ece"] <= c["ece_uncalibrated"] + 1e-9


def test_ece_helper():
    y = np.array([0, 0, 1, 1])
    assert model.expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert model.expected_calibration_error(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)
    assert model.expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_reasons_point_the_right_way(built):
    d, _ = built
    m = model.WinModel.load(d / "model.pkl")
    tx, disp = generator.generate(n_txns=60_000, n_customers=18_000, seed=5)
    F = build_features(tx, disp)
    row = F[(disp["category"] == "not_received").to_numpy() & (F["ticket_scanned"] == 1).to_numpy()].head(1)
    rs = {r["feature"]: r for r in model.reasons(m, row, category="not_received")}
    assert rs["ticket_scanned"]["effect"] > 0
    assert "cancel_before_charge" not in rs  # not relevant to this reason


def test_policy_gate_passes_on_shared_run(built):
    from disputedesk.pipeline import gate

    ok, msg = gate(built[1])
    assert ok, msg
