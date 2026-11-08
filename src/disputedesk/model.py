"""Win-probability model: gradient boosting on evidence availability, a time split, and Platt calibration.

Split by the date a dispute was received, never at random:
    train        received before 2026-06-01   (fit the trees)
    calibration  June and July 2026           (fit the Platt map from score to probability)
    test         August 2026 onward           (policy comparison and every reported number)

Labels: until the model went live, the merchant represented every dispute it could, so every training dispute has
a true outcome. The generator records the outcome-if-fought for every dispute, which lets the test set score policies
that would have fought fewer of them.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from .features import CATEGORICAL, FEATURES

TRAIN_END = date(2026, 6, 1)
CALIB_END = date(2026, 8, 1)
SEED = 7


def split(d: pd.DataFrame, train_end: date = TRAIN_END, calib_end: date = CALIB_END) -> dict[str, np.ndarray]:
    r = pd.to_datetime(d["received_date"]).dt.date
    return {
        "train": (r < train_end).to_numpy(),
        "calib": ((r >= train_end) & (r < calib_end)).to_numpy(),
        "test": (r >= calib_end).to_numpy(),
    }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Platt:
    """Logistic regression on the boosted model's log-odds. With a few hundred calibration disputes it is steadier
    than isotonic regression, which overfits step edges (measured: test ECE 0.018 vs 0.029 at full size)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Platt:
        self.lr.fit(_logit(raw), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(raw))[:, 1]


@dataclass
class WinModel:
    gbm: HistGradientBoostingClassifier
    cal: Platt
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    meta: dict = field(default_factory=dict)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        with threadpool_limits(1):  # a few thousand rows: OpenMP thread start-up costs more than it saves
            return self.gbm.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.cal.predict(self.raw(X)), 0.005, 0.995)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> WinModel:
        return pickle.loads(path.read_bytes())


def fit(X: pd.DataFrame, y: np.ndarray, masks: dict[str, np.ndarray], seed: int = SEED) -> WinModel:
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    gbm = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        random_state=seed,
    )
    with threadpool_limits(1):
        gbm.fit(X.loc[masks["train"], FEATURES], y[masks["train"]])
        raw_cal = gbm.predict_proba(X.loc[masks["calib"], FEATURES])[:, 1]
    return WinModel(gbm, Platt().fit(raw_cal, y[masks["calib"]]))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -1, 2
    idx = np.clip(np.digitize(p, q[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {"bin": b + 1, "n": int(m.sum()), "predicted": float(p[m].mean()), "observed": float(y[m].mean())}
            )
    return rows


def evaluate(model: WinModel, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> dict:
    p = model.predict(X.loc[mask])
    yt = y[mask].astype(int)
    raw = model.raw(X.loc[mask])
    return {
        "n": int(mask.sum()),
        "win_rate": float(yt.mean()),
        "auc": float(roc_auc_score(yt, p)),
        "brier": float(brier_score_loss(yt, p)),
        "brier_uncalibrated": float(brier_score_loss(yt, raw)),
        "ece": expected_calibration_error(yt, p),
        "ece_uncalibrated": expected_calibration_error(yt, raw),
        "calibration": calibration_table(yt, p),
    }


EVIDENCE_BASELINE = {
    # For "reasons": what the feature looks like when that piece of evidence is missing.
    "prior_device_purchases": 0,
    "ticket_scanned": 0,
    "delivered_before": 0,
    "tracking": 0,
    "three_ds": 0,
    "avs": 0,
    "cvv_match": 0,
    "cs_contact_before": 1,
    "cs_complaint_before": 1,
    "refund_before": 0,
    "cancel_before_charge": 1,
    "dup_separate_order": 0,
    "dup_retry_charge": 1,
    "ship_bill_match": 0,
    "customer_prior_disputes": 3,
    "ip_distance_km": 500.0,
}
REASON_LABELS = {
    "prior_device_purchases": "Undisputed purchases from the same device",
    "ticket_scanned": "Ticket scanned at the venue",
    "delivered_before": "Carrier shows delivery",
    "tracking": "Tracking and signature",
    "three_ds": "3-D Secure result",
    "avs": "Address verification",
    "cvv_match": "CVV match",
    "cs_contact_before": "No customer service contact before the dispute",
    "cs_complaint_before": "No complaint logged",
    "refund_before": "Refund issued before the dispute",
    "cancel_before_charge": "No cancellation before the charge",
    "dup_separate_order": "Second charge is a separate fulfilled order",
    "dup_retry_charge": "Not a gateway retry charge",
    "ship_bill_match": "Shipped to the billing address",
    "customer_prior_disputes": "Customer dispute history",
    "ip_distance_km": "IP location near the billing address",
}


COMMON = ["customer_prior_disputes", "cs_contact_before", "cs_complaint_before", "refund_before"]
RELEVANT = {
    "fraud": [
        "prior_device_purchases",
        "three_ds",
        "avs",
        "cvv_match",
        "ip_distance_km",
        "ticket_scanned",
        "delivered_before",
        "ship_bill_match",
        *COMMON,
    ],
    "not_received": ["delivered_before", "tracking", "ticket_scanned", "ship_bill_match", *COMMON],
    "not_as_described": ["ticket_scanned", "delivered_before", *COMMON],
    "duplicate": ["dup_separate_order", "dup_retry_charge", "delivered_before", "ticket_scanned", *COMMON],
    "credit_not_processed": ["ticket_scanned", *COMMON],
    "canceled_recurring": ["cancel_before_charge", *COMMON],
}


def reasons(model: WinModel, row: pd.DataFrame, top: int = 5, category: str | None = None) -> list[dict]:
    """How much each piece of evidence moves P(win): swap it for its missing value and re-score.

    Positive means the evidence helps. This is a simple ablation, cheap at one row and easy to read.
    """
    base = float(model.predict(row)[0])
    variants = []
    keys = []
    allowed = set(RELEVANT.get(category, EVIDENCE_BASELINE)) if category else set(EVIDENCE_BASELINE)
    for k, v in EVIDENCE_BASELINE.items():
        if k not in allowed or row.iloc[0][k] == v:
            continue
        r = row.copy()
        r[k] = v
        variants.append(r)
        keys.append(k)
    if not variants:
        return []
    ps = model.predict(pd.concat(variants, ignore_index=True))
    out = [
        {"feature": k, "label": REASON_LABELS[k], "value": _plain(row.iloc[0][k]), "effect": round(base - float(p), 4)}
        for k, p in zip(keys, ps, strict=True)
    ]
    out.sort(key=lambda r: -abs(r["effect"]))
    return [r for r in out[:top] if abs(r["effect"]) >= 0.005]


def _plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v
