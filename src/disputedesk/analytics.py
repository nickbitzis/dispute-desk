"""Prevention analytics: where disputes come from, what 3-D Secure would prevent, repeat disputers, and monitoring
against the network's monthly dispute-rate threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .codes import CATEGORY_LABELS, DISPUTE_FEE, EARLY_WARNING, HANDLING_COST, NETWORK_THRESHOLD

ABANDON_RATE = 0.035  # share of good customers who drop out when challenged with 3-D Secure
MARGIN = 0.22  # contribution margin on a lost sale
PREVENTION = 0.85  # share of stolen-card fraud stopped (or shifted to the issuer) by a challenge


def rates_by(tx: pd.DataFrame, d: pd.DataFrame, col: str) -> list[dict]:
    dd = d.merge(tx[["txn_id", col]], on="txn_id", how="left") if col not in d else d
    n_tx = tx.groupby(col).size()
    amt_tx = tx.groupby(col)["amount"].sum()
    n_d = dd.groupby(col).size().reindex(n_tx.index, fill_value=0)
    amt_d = dd.groupby(col)["amount"].sum().reindex(n_tx.index, fill_value=0)
    return [
        {
            col: str(k),
            "transactions": int(n_tx[k]),
            "disputes": int(n_d[k]),
            "rate": float(n_d[k] / n_tx[k]),
            "amount_rate": float(amt_d[k] / amt_tx[k]) if amt_tx[k] else 0.0,
        }
        for k in n_tx.index
    ]


def by_reason(d: pd.DataFrame, n_tx: int) -> list[dict]:
    g = d.groupby("category").agg(disputes=("dispute_id", "size"), amount=("amount", "sum"))
    return [
        {
            "category": k,
            "label": CATEGORY_LABELS[k],
            "disputes": int(r.disputes),
            "amount": round(float(r.amount), 2),
            "rate": float(r.disputes / n_tx),
        }
        for k, r in g.sort_values("disputes", ascending=False).iterrows()
    ]


def monthly_monitor(
    tx: pd.DataFrame, d: pd.DataFrame, threshold: float = NETWORK_THRESHOLD, warning: float = EARLY_WARNING
) -> dict:
    """Count-based monthly dispute rate: disputes received in a month / sales in the month before.

    Disputes trail sales by weeks, so the prior month's sales are the fairer denominator (one of the two common
    network conventions). The first month has no prior month and is skipped. Status per month: ok, warning (at or
    above the early warning level, or the three-month trend projects a breach next month), or breach.
    """
    tm = tx["ts"].dt.to_period("M")
    dm = pd.to_datetime(d["received_date"]).dt.to_period("M")
    sales = tm.value_counts().sort_index()
    disp = dm.value_counts()
    months = []
    rates: list[float] = []
    for prev, m in zip(sales.index[:-1], sales.index[1:], strict=True):
        n = int(disp.get(m, 0))
        rate = float(n / sales[prev])
        rates.append(rate)
        proj = None
        if len(rates) >= 3:
            slope = float(np.polyfit(np.arange(3), rates[-3:], 1)[0])
            proj = rates[-1] + slope
        if rate >= threshold:
            status = "breach"
        elif rate >= warning or (proj is not None and proj >= threshold):
            status = "warning"
        else:
            status = "ok"
        months.append(
            {
                "month": str(m),
                "sales_prior_month": int(sales[prev]),
                "disputes": n,
                "rate": rate,
                "projected_next": proj,
                "status": status,
            }
        )
    first_warning = next((m["month"] for m in months if m["status"] != "ok"), None)
    first_breach = next((m["month"] for m in months if m["status"] == "breach"), None)
    return {
        "threshold": threshold,
        "early_warning": warning,
        "months": months,
        "first_warning": first_warning,
        "first_breach": first_breach,
        "latest": months[-1] if months else None,
    }


def three_ds_thresholds(tx: pd.DataFrame, d: pd.DataFrame, thresholds=(0, 50, 100, 150, 200, 250, 400)) -> list[dict]:
    """What requiring a 3-D Secure challenge above an amount would have done over the year.

    Prevented: fraud disputes on orders at or above the threshold that were not already authenticated, times the
    prevention rate. Saved per prevented dispute: the amount, the dispute fee, and handling. Cost: good customers who
    abandon at the challenge, valued at the lost margin.
    """
    fraud = d[d["category"] == "fraud"].merge(tx[["txn_id", "three_ds"]], on="txn_id")
    rows = []
    unauth_tx = tx[tx["three_ds"] == "none"]
    for t in thresholds:
        f = fraud[(fraud["amount"] >= t) & (fraud["three_ds"] == "none")]
        prevented = len(f) * PREVENTION
        saved = float(f["amount"].sum() + len(f) * (DISPUTE_FEE + HANDLING_COST)) * PREVENTION
        challenged = unauth_tx[(unauth_tx["amount"] >= t) & (~unauth_tx["is_fraud"])]
        lost_sales = float(challenged["amount"].sum() * ABANDON_RATE)
        cost = lost_sales * MARGIN
        rows.append(
            {
                "threshold": t,
                "challenged_orders": int(len(challenged)),
                "fraud_disputes_prevented": round(prevented, 1),
                "share_of_fraud_disputes": float(prevented / len(fraud)) if len(fraud) else 0.0,
                "dispute_losses_avoided": round(saved, 2),
                "lost_sales": round(lost_sales, 2),
                "lost_margin": round(cost, 2),
                "net_benefit": round(saved - cost, 2),
            }
        )
    return rows


def friendly_fraud(d: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Repeat disputers and the evidence that the customer received what they paid for."""
    dd = d.reset_index(drop=True).join(
        features.reset_index(drop=True)[["ticket_scanned", "delivered_before", "prior_device_purchases"]]
    )
    per = dd.groupby("customer_id").size()
    repeat_ids = per[per >= 2].index
    rep = dd[dd["customer_id"].isin(repeat_ids)]
    received_signal = (rep["ticket_scanned"] == 1) | (rep["delivered_before"] == 1)
    top = (
        rep.groupby("customer_id")
        .agg(disputes=("dispute_id", "size"), amount=("amount", "sum"))
        .sort_values(["disputes", "amount"], ascending=False)
        .head(10)
    )
    return {
        "repeat_customers": int(len(repeat_ids)),
        "disputes_from_repeat": int(len(rep)),
        "share_of_disputes": float(len(rep) / len(dd)) if len(dd) else 0.0,
        "share_with_proof_of_receipt": float(received_signal.mean()) if len(rep) else 0.0,
        "win_rate_if_fought": float(rep["won_if_fought"].mean()) if len(rep) else 0.0,
        "top": [
            {"customer_id": int(k), "disputes": int(r.disputes), "amount": round(float(r.amount), 2)}
            for k, r in top.iterrows()
        ],
    }


def build(tx: pd.DataFrame, d: pd.DataFrame, features: pd.DataFrame) -> dict:
    return {
        "totals": {
            "transactions": int(len(tx)),
            "sales": round(float(tx["amount"].sum()), 2),
            "disputes": int(len(d)),
            "disputed_amount": round(float(d["amount"].sum()), 2),
            "dispute_rate": float(len(d) / len(tx)),
        },
        "by_product": rates_by(tx, d, "product"),
        "by_product_type": rates_by(tx, d, "product_type"),
        "by_channel": rates_by(tx, d, "channel"),
        "by_reason": by_reason(d, len(tx)),
        "monitor": monthly_monitor(tx, d),
        "three_ds": three_ds_thresholds(tx, d),
        "friendly_fraud": friendly_fraud(d, features),
    }
