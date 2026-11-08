import pytest

from disputedesk.decision import decide
from disputedesk.liability import liability_shift


def test_3ds_authenticated_fraud_shifts():
    r = liability_shift("fraud", "authenticated")
    assert r.applies and r.rule_id == "LS-3DS-AUTH" and r.p_floor == 0.95


def test_3ds_attempted_fraud_shifts():
    r = liability_shift("fraud", "attempted")
    assert r.applies and r.rule_id == "LS-3DS-ATTEMPT"


def test_no_3ds_fraud_no_shift():
    assert not liability_shift("fraud", "none").applies


@pytest.mark.parametrize("cat", ["not_received", "not_as_described", "duplicate", "canceled_recurring"])
def test_3ds_does_not_cover_non_fraud(cat):
    r = liability_shift(cat, "authenticated")
    assert not r.applies and "fraud only" in r.explanation


def test_refund_before_dispute_applies_to_any_category():
    assert liability_shift("not_received", "none", refund_before_dispute=True).rule_id == "CR-REFUNDED"


def test_shift_forces_fight_and_floors_probability():
    d = decide(p_model=0.05, amount=100.0, category="fraud", three_ds="authenticated", refund_before=False)
    assert d["p_win"] == 0.95 and d["recommendation"] == "represent"
    d2 = decide(p_model=0.05, amount=100.0, category="fraud", three_ds="none", refund_before=False)
    assert d2["recommendation"] == "accept"


def test_shift_does_not_force_a_fight_that_costs_more_than_it_recovers():
    d = decide(p_model=0.05, amount=9.99, category="fraud", three_ds="authenticated", refund_before=False)
    assert d["liability"]["applies"] and d["expected_value"] < 0 and d["recommendation"] == "accept"
