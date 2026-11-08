"""Evidence availability features per dispute, computed as of the date the dispute was received.

Every timestamped fact (venue scan, delivery, refund, customer service contact, earlier disputes) counts only if it
happened before the dispute arrived. The leakage test truncates the data at a cutoff and checks that features for
earlier disputes do not change.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .codes import CATEGORIES

THREE_DS = {"none": 0, "attempted": 1, "authenticated": 2}
AVS = {"no_match": 0, "partial": 1, "match": 2}
TRACKING = {"none": 0, "in_transit": 1, "delivered": 2, "delivered_signature": 3}
PRODUCTS = {"ticket": 0, "merch": 1, "membership": 2}
CHANNELS = {"web": 0, "app": 1, "phone": 2}

FEATURES = [
    "category",
    "amount",
    "product_type",
    "channel",
    "network_a",
    "three_ds",
    "avs",
    "cvv_match",
    "ip_distance_km",
    "ip_foreign",
    "ship_bill_match",
    "tracking",
    "delivered_before",
    "ticket_scanned",
    "event_passed",
    "prior_device_purchases",
    "customer_prior_disputes",
    "tenure_days",
    "cs_contact_before",
    "cs_complaint_before",
    "refund_before",
    "cancel_before_charge",
    "dup_retry_charge",
    "dup_separate_order",
    "days_to_dispute",
]
CATEGORICAL = ["category", "product_type", "channel"]


def _before(col: pd.Series, rec: np.ndarray) -> np.ndarray:
    v = col.to_numpy()
    return (col.notna().to_numpy()) & (v < rec)


def build_features(tx: pd.DataFrame, disputes: pd.DataFrame) -> pd.DataFrame:
    disputes = disputes.reset_index(drop=True)
    txi = tx.set_index("txn_id")
    t = txi.loc[disputes["txn_id"].to_numpy()]
    rec = pd.to_datetime(disputes["received_date"]).to_numpy()
    tts = t["ts"].to_numpy()

    f = pd.DataFrame(index=disputes.index)
    f["category"] = disputes["category"].map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy()
    f["amount"] = disputes["amount"].to_numpy()
    f["product_type"] = t["product_type"].map(PRODUCTS).to_numpy()
    f["channel"] = t["channel"].map(CHANNELS).to_numpy()
    f["network_a"] = (disputes["network"] == "A").astype(int).to_numpy()
    f["three_ds"] = t["three_ds"].map(THREE_DS).to_numpy()
    f["avs"] = t["avs"].map(AVS).to_numpy()
    f["cvv_match"] = t["cvv_match"].astype(int).to_numpy()
    f["ip_distance_km"] = t["ip_distance_km"].to_numpy()
    f["ip_foreign"] = (t["ip_country"] != "US").astype(int).to_numpy()
    f["ship_bill_match"] = t["ship_bill_match"].astype(int).to_numpy()
    delivered = _before(t["delivered_ts"], rec)
    f["tracking"] = np.where(
        delivered, t["tracking"].map(TRACKING).to_numpy(), np.minimum(t["tracking"].map(TRACKING), 1)
    )
    f["delivered_before"] = delivered.astype(int)
    f["ticket_scanned"] = _before(t["scan_ts"], rec).astype(int)
    f["event_passed"] = _before(t["event_ts"], rec).astype(int)
    f["cs_contact_before"] = _before(t["cs_ts"], rec).astype(int)
    f["cs_complaint_before"] = (_before(t["cs_ts"], rec) & t["cs_complaint"].to_numpy().astype(bool)).astype(int)
    f["refund_before"] = _before(t["refund_ts"], rec).astype(int)
    f["cancel_before_charge"] = (t["cancel_ts"].notna().to_numpy() & (t["cancel_ts"].to_numpy() < tts)).astype(int)
    f["dup_retry_charge"] = (t["dup_of"].to_numpy() > 0).astype(int)
    f["dup_separate_order"] = (t["dup_of"].to_numpy() < 0).astype(int)
    f["tenure_days"] = ((tts - t["customer_signup"].to_numpy()) / np.timedelta64(1, "D")).astype(int)
    f["days_to_dispute"] = ((rec - tts) / np.timedelta64(1, "D")).astype(int)
    f["prior_device_purchases"] = prior_undisputed_device_purchases(tx, disputes)
    f["customer_prior_disputes"] = customer_prior_disputes(disputes)
    return f[FEATURES]


def prior_undisputed_device_purchases(tx: pd.DataFrame, disputes: pd.DataFrame) -> np.ndarray:
    """Orders on the same device before this order that had not been disputed by the time this dispute arrived.

    This is the core of compelling evidence for fraud disputes: the same device has bought before without trouble.
    """
    txi = tx.set_index("txn_id")
    dd = disputes.assign(
        device=txi.loc[disputes["txn_id"].to_numpy(), "device_id"].to_numpy(),
        tts=txi.loc[disputes["txn_id"].to_numpy(), "ts"].to_numpy(),
        rec=pd.to_datetime(disputes["received_date"]).to_numpy(),
    )
    devs = set(dd["device"])
    sub = tx[tx["device_id"].isin(devs)]
    by_dev = {k: np.sort(g.to_numpy()) for k, g in sub.groupby("device_id")["ts"]}
    disputed: dict[str, list[tuple]] = defaultdict(list)
    for dev, tts, rec in zip(dd["device"], dd["tts"], dd["rec"], strict=True):
        disputed[dev].append((tts, rec))
    out = np.zeros(len(dd), dtype=int)
    for i, (dev, tts, rec) in enumerate(zip(dd["device"], dd["tts"], dd["rec"], strict=True)):
        prior = int(np.searchsorted(by_dev[dev], tts, side="left"))
        flagged = sum(1 for (ots, orec) in disputed[dev] if ots < tts and orec < rec)
        out[i] = max(prior - flagged, 0)
    return out


def customer_prior_disputes(disputes: pd.DataFrame) -> np.ndarray:
    disputes = disputes.reset_index(drop=True)
    rec = pd.to_datetime(disputes["received_date"])
    out = np.zeros(len(disputes), dtype=int)
    for _, g in disputes.assign(_rec=rec).groupby("customer_id"):
        if len(g) == 1:
            continue
        r = g["_rec"].to_numpy()
        out[g.index.to_numpy()] = [(r < x).sum() for x in r]
    return out
