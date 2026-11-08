"""Expected value of fighting, the decision for one dispute, and the policy comparison.

The amount is already debited when a dispute arrives, and the dispute fee is charged either way, so accepting is
worth $0 from here. Fighting is worth:

    EV = P(win) x amount - handling cost - (1 - P(win)) x loss fee

Net recovered dollars for a policy, scored on true outcomes: for each dispute fought, add the amount if it was won,
subtract the handling cost, and subtract the loss fee if it was lost.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .codes import HANDLING_COST, LOSS_FEE
from .liability import liability_shift


def expected_value(p_win, amount, handling: float = HANDLING_COST, loss_fee: float = LOSS_FEE):
    p_win = np.asarray(p_win, dtype=float)
    amount = np.asarray(amount, dtype=float)
    return p_win * amount - handling - (1.0 - p_win) * loss_fee


def breakeven_probability(amount: float, handling: float = HANDLING_COST, loss_fee: float = LOSS_FEE) -> float:
    """Smallest P(win) at which fighting pays: p*A - h - (1-p)*f = 0  =>  p = (h + f) / (A + f)."""
    return (handling + loss_fee) / (amount + loss_fee)


def decide(p_model: float, amount: float, category: str, three_ds: str, refund_before: bool) -> dict:
    ls = liability_shift(category, three_ds, refund_before)
    p = max(p_model, ls.p_floor) if ls.applies else p_model
    ev = float(expected_value(p, amount))
    fight = ev > 0  # a liability shift raises P(win) through the floor; a tiny amount can still not be worth it
    return {
        "p_model": round(float(p_model), 4),
        "p_win": round(float(p), 4),
        "expected_value": round(ev, 2),
        "breakeven_p": round(breakeven_probability(amount), 4),
        "recommendation": "represent" if fight else "accept",
        "liability": ls.as_dict(),
    }


def net_recovered(fight: np.ndarray, won: np.ndarray, amount: np.ndarray) -> dict:
    fight = np.asarray(fight, dtype=bool)
    won = np.asarray(won, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    recovered = float(amount[fight & won].sum())
    handling = float(fight.sum() * HANDLING_COST)
    fees = float((fight & ~won).sum() * LOSS_FEE)
    return {
        "fought": int(fight.sum()),
        "accepted": int((~fight).sum()),
        "won": int((fight & won).sum()),
        "win_rate": float((fight & won).sum() / fight.sum()) if fight.any() else 0.0,
        "recovered": round(recovered, 2),
        "handling_cost": round(handling, 2),
        "loss_fees": round(fees, 2),
        "net": round(recovered - handling - fees, 2),
    }


def has_required_evidence(df: pd.DataFrame) -> np.ndarray:
    """The rule-of-thumb view of 'we have the key document' per category, from features."""
    cat = df["category_name"].to_numpy()
    out = np.zeros(len(df), dtype=bool)
    out |= (cat == "fraud") & ((df["three_ds"] >= 1) | (df["prior_device_purchases"] >= 2)).to_numpy()
    out |= (cat == "not_received") & ((df["delivered_before"] == 1) | (df["ticket_scanned"] == 1)).to_numpy()
    out |= (cat == "not_as_described") & (df["cs_complaint_before"] == 0).to_numpy()
    out |= (cat == "duplicate") & (df["dup_separate_order"] == 1).to_numpy()
    out |= (cat == "credit_not_processed") & ((df["refund_before"] == 1) | (df["product_type"] == 0)).to_numpy()
    out |= (cat == "canceled_recurring") & (df["cancel_before_charge"] == 0).to_numpy()
    return out


POLICIES: dict[str, tuple[str, Callable[[pd.DataFrame], np.ndarray]]] = {
    "fight_everything": ("Represent every dispute", lambda d: np.ones(len(d), dtype=bool)),
    "fight_nothing": ("Accept every dispute", lambda d: np.zeros(len(d), dtype=bool)),
    "amount_over_75": ("Represent when the amount is $75 or more", lambda d: (d["amount"] >= 75).to_numpy()),
    "rule_of_thumb": (
        "Represent when the key document exists and the amount is $40 or more",
        lambda d: has_required_evidence(d) & (d["amount"] >= 40).to_numpy(),
    ),
    "model_ev": (
        "Represent when expected value is positive (liability shift rules floor P(win))",
        lambda d: (d["fight_model"]).to_numpy().astype(bool),
    ),
}


def compare_policies(df: pd.DataFrame) -> list[dict]:
    """df needs: amount, won_if_fought, category_name, the feature columns, and fight_model."""
    won = df["won_if_fought"].to_numpy()
    amount = df["amount"].to_numpy()
    rows = []
    for key, (label, rule) in POLICIES.items():
        r = net_recovered(rule(df), won, amount)
        rows.append({"policy": key, "label": label, **r})
    oracle = net_recovered(won, won, amount)
    rows.append({"policy": "oracle", "label": "Hindsight: represent only the disputes that win", **oracle})
    base = next(r["net"] for r in rows if r["policy"] == "fight_everything")
    for r in rows:
        r["vs_fight_everything"] = round(r["net"] - base, 2)
    return rows
