import pandas as pd

from disputedesk import generator
from disputedesk.codes import CATEGORIES, CODE_TO_CATEGORY, REASON_CODES, codes_for


def test_seeded_and_deterministic():
    a_tx, a_d = generator.generate(n_txns=20_000, n_customers=6_000, seed=3)
    b_tx, b_d = generator.generate(n_txns=20_000, n_customers=6_000, seed=3)
    pd.testing.assert_frame_equal(a_d, b_d)
    assert a_tx["amount"].sum() == b_tx["amount"].sum()


def test_disputes_are_consistent_with_orders():
    tx, d = generator.generate(n_txns=40_000, n_customers=12_000, seed=4)
    assert set(d["category"]) <= set(CATEGORIES)
    assert (d["reason_code"].map(CODE_TO_CATEGORY) == d["category"]).all()
    t = tx.set_index("txn_id").loc[d["txn_id"]]
    assert (t["amount"].to_numpy() == d["amount"].to_numpy()).all()
    assert (pd.to_datetime(d["received_date"]).to_numpy() > t["ts"].to_numpy()).all()
    assert (pd.to_datetime(d["received_date"]).dt.date <= generator.AS_OF).all()
    assert d["dispute_id"].is_unique
    assert d["p_true"].between(0, 1).all()


def test_every_category_has_a_code_on_both_networks():
    for c in CATEGORIES:
        assert codes_for(c, "A") and codes_for(c, "B")
    assert len({code for _, code, _, _ in REASON_CODES}) == len(REASON_CODES)


def test_hidden_model_rewards_the_right_evidence():
    tx, d = generator.generate(n_txns=120_000, n_customers=36_000, seed=8)
    t = tx.set_index("txn_id").loc[d["txn_id"]]
    nr = (d["category"] == "not_received").to_numpy()
    scanned = t["scan_ts"].notna().to_numpy()
    assert d["p_true"][nr & scanned].mean() > d["p_true"][nr & ~scanned].mean() + 0.2
    dup = (d["category"] == "duplicate").to_numpy()
    retry = (t["dup_of"] > 0).to_numpy()
    assert d["p_true"][dup & ~retry].mean() > d["p_true"][dup & retry].mean() + 0.4
