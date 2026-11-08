import numpy as np
import pandas as pd
import pytest

from disputedesk.codes import HANDLING_COST, LOSS_FEE
from disputedesk.decision import breakeven_probability, compare_policies, decide, expected_value, net_recovered


def test_ev_formula():
    assert expected_value(0.6, 200.0) == pytest.approx(0.6 * 200 - HANDLING_COST - 0.4 * LOSS_FEE)


def test_ev_is_zero_at_breakeven():
    for amount in (30.0, 75.0, 400.0):
        assert expected_value(breakeven_probability(amount), amount) == pytest.approx(0.0, abs=1e-9)


def test_ev_vectorized_and_small_disputes_negative():
    ev = expected_value(np.array([0.9, 0.9]), np.array([20.0, 500.0]))
    assert ev[0] < 0 < ev[1]


def test_decide_recommends_by_ev_sign():
    assert decide(0.7, 300.0, "not_received", "none", False)["recommendation"] == "represent"
    assert decide(0.3, 40.0, "not_received", "none", False)["recommendation"] == "accept"


def test_net_recovered_counts_costs():
    r = net_recovered(np.array([1, 1, 0], bool), np.array([1, 0, 1], bool), np.array([100.0, 50.0, 80.0]))
    assert r["fought"] == 2 and r["won"] == 1 and r["accepted"] == 1
    assert r["net"] == pytest.approx(100 - 2 * HANDLING_COST - LOSS_FEE)


def tiny_frame() -> pd.DataFrame:
    """Four disputes where the right answer is obvious: fight the big likely win, accept the rest."""
    base = {
        "three_ds": 0,
        "prior_device_purchases": 0,
        "delivered_before": 0,
        "ticket_scanned": 0,
        "cs_complaint_before": 0,
        "dup_separate_order": 0,
        "refund_before": 0,
        "product_type": 1,
        "cancel_before_charge": 0,
    }
    rows = [
        {
            "amount": 300.0,
            "won_if_fought": True,
            "category_name": "not_received",
            "fight_model": True,
            "delivered_before": 1,
        },
        {
            "amount": 20.0,
            "won_if_fought": True,
            "category_name": "not_received",
            "fight_model": False,
            "delivered_before": 1,
        },
        {"amount": 150.0, "won_if_fought": False, "category_name": "not_received", "fight_model": False},
        {"amount": 60.0, "won_if_fought": False, "category_name": "duplicate", "fight_model": False},
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_policy_comparison_on_tiny_dataset():
    rows = {r["policy"]: r for r in compare_policies(tiny_frame())}
    h, f = HANDLING_COST, LOSS_FEE
    assert rows["fight_everything"]["net"] == pytest.approx(320 - 4 * h - 2 * f)
    assert rows["fight_nothing"]["net"] == 0
    assert rows["model_ev"]["net"] == pytest.approx(300 - h)
    assert rows["oracle"]["net"] == pytest.approx(320 - 2 * h)
    assert rows["amount_over_75"]["fought"] == 2  # 300 and 150
    assert rows["rule_of_thumb"]["fought"] == 1  # only the $300 has delivery proof and is >= $40
    assert rows["model_ev"]["vs_fight_everything"] == pytest.approx(
        rows["model_ev"]["net"] - rows["fight_everything"]["net"]
    )
