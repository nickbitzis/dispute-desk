"""End-to-end batch job: generate, build features, train and calibrate, score, compare policies, analytics, and
seed the operational store with each dispute's lifecycle up to the demo date.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, decision, generator, model, store
from .bizdays import add_business_days
from .codes import CATEGORIES
from .features import build_features
from .lifecycle import DisputeState, new_dispute, transition

ANALYSTS = ["r.moreno", "k.osei", "t.lindqvist"]


def score(m: model.WinModel, d: pd.DataFrame, F: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    p = m.predict(F)
    txi = tx.set_index("txn_id")
    three = txi.loc[d["txn_id"].to_numpy(), "three_ds"].to_numpy()
    out = [
        decision.decide(float(pi), float(a), c, t, bool(rb))
        for pi, a, c, t, rb in zip(p, d["amount"], d["category"], three, F["refund_before"], strict=True)
    ]
    d = d.copy()
    d["p_model"] = [o["p_model"] for o in out]
    d["p_win"] = [o["p_win"] for o in out]
    d["ev"] = [o["expected_value"] for o in out]
    d["recommendation"] = [o["recommendation"] for o in out]
    d["liability_rule"] = [o["liability"]["rule_id"] for o in out]
    return d


def policy_frame(d: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([d.reset_index(drop=True), F.reset_index(drop=True).drop(columns=["amount", "category"])], axis=1)
    df["category_name"] = df["category"]
    df["fight_model"] = df["recommendation"] == "represent"
    return df


def seed_lifecycle(d: pd.DataFrame, as_of: date, go_live: date, seed: int = 11):
    """Replay each dispute's history to the demo date: the old policy (represent everything) before go-live, the
    model's recommendation after it. Returns states and audit events."""
    rng = np.random.default_rng(seed)
    states: list[DisputeState] = []
    events: list[tuple] = []

    def ev(s_from, s_to, day, actor, action, note=None):
        events.append((s_to.dispute_id, f"{day.isoformat()}T10:00:00", actor, action, s_from, s_to.status, note))

    for r in d.itertuples(index=False):
        rec = r.received_date
        s = new_dispute(r.dispute_id, rec)
        events.append(
            (r.dispute_id, f"{rec.isoformat()}T08:00:00", "network", "received", None, "received", r.reason_code)
        )
        analyst = ANALYSTS[int(rng.integers(0, len(ANALYSTS)))]
        fight = True if rec < go_live else r.recommendation == "represent"
        if s.deadline >= as_of:  # still inside the response window: open work
            review_day = add_business_days(rec, 1)
            if review_day <= as_of and rng.random() < 0.5:
                s2 = transition(s, "under_review", review_day)
                ev("received", s2, review_day, analyst, "assign")
                s = s2
            states.append(s)
            continue
        if not fight:
            day = add_business_days(rec, 1)
            s2 = transition(s, "accepted", day)
            ev("received", s2, day, "system", "auto_accept", "negative_ev")
            states.append(s2)
            continue
        day = add_business_days(rec, 2)
        s = transition(s, "under_review", day)
        ev("received", s, day, analyst, "assign")
        s.packet_ready = True
        sub = add_business_days(rec, int(rng.integers(4, 15)))
        s2 = transition(s, "represented", sub)
        ev("under_review", s2, sub, analyst, "represent", "packet submitted")
        s = s2
        decided = add_business_days(sub, int(rng.integers(8, 28)))
        if decided > as_of:
            states.append(s)
            continue
        if r.pre_arb:
            s2 = transition(s, "pre_arbitration", decided)
            ev("represented", s2, decided, "issuer", "pre_arbitration")
            s = s2
            decided = add_business_days(decided, int(rng.integers(3, 9)))
            if decided > as_of:
                states.append(s)
                continue
        final = "won" if r.won_if_fought else "lost"
        prev = s.status
        s2 = transition(s, final, decided)
        ev(prev, s2, decided, "issuer", final)
        states.append(s2)
    return states, events


def run(data_dir: Path, n_txns: int = 300_000, n_customers: int = 90_000, seed: int = 42, log=print) -> dict:
    t0 = time.time()
    data_dir.mkdir(parents=True, exist_ok=True)
    tx, d = generator.generate(n_txns=n_txns, n_customers=n_customers, seed=seed)
    log(f"generated {len(tx):,} transactions and {len(d):,} disputes in {time.time() - t0:.1f}s")
    F = build_features(tx, d)
    masks = model.split(d)
    y = d["won_if_fought"].to_numpy().astype(int)
    m = model.fit(F, y, masks)
    metrics = {k: model.evaluate(m, F, y, masks[k]) for k in ("calib", "test")}
    metrics["train_n"] = int(masks["train"].sum())
    m.meta = {"seed": seed, "train_end": str(model.TRAIN_END), "calib_end": str(model.CALIB_END), "metrics": metrics}
    m.save(data_dir / "model.pkl")
    log(
        f"model test AUC {metrics['test']['auc']:.3f}, Brier {metrics['test']['brier']:.4f}, "
        f"ECE {metrics['test']['ece']:.4f}"
    )

    d = score(m, d, F, tx)
    d["split"] = np.select([masks["train"], masks["calib"]], ["train", "calib"], "test")
    pf = policy_frame(d, F)
    test = pf[masks["test"]]
    policies = decision.compare_policies(test)
    by_cat = []
    for c in CATEGORIES:
        sub = test[test["category_name"] == c]
        if len(sub):
            rows = {r["policy"]: r for r in decision.compare_policies(sub)}
            by_cat.append(
                {
                    "category": c,
                    "n": int(len(sub)),
                    "fight_everything": rows["fight_everything"]["net"],
                    "rule_of_thumb": rows["rule_of_thumb"]["net"],
                    "model_ev": rows["model_ev"]["net"],
                    "model_fought": rows["model_ev"]["fought"],
                }
            )
    for p in policies:
        log(f"  {p['policy']:<18} fought {p['fought']:>5}  net ${p['net']:>12,.2f}")

    ana = analytics.build(tx, d, F)
    store.write_all(data_dir / "disputedesk.db", tx, d, F)
    states, events = seed_lifecycle(d, generator.AS_OF, model.CALIB_END)
    with store.session(data_dir / "disputedesk.db") as con:
        store.write_states(con, states, events)

    report = {
        "generated": {
            "transactions": int(len(tx)),
            "disputes": int(len(d)),
            "seed": seed,
            "as_of": str(generator.AS_OF),
        },
        "split": {k: int(v.sum()) for k, v in masks.items()},
        "model": metrics,
        "policies": policies,
        "policies_by_category": by_cat,
        "liability_shift_disputes_test": int(test["liability_rule"].notna().sum()),
        "analytics": ana,
        "seconds": round(time.time() - t0, 1),
    }
    (data_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"done in {report['seconds']}s, wrote {data_dir}")
    return report


def gate(report: dict) -> tuple[bool, str]:
    p = {r["policy"]: r for r in report["policies"]}
    model_net, all_net = p["model_ev"]["net"], p["fight_everything"]["net"]
    ok = model_net > all_net
    return ok, f"model policy net ${model_net:,.2f} vs fight-everything ${all_net:,.2f}: {'PASS' if ok else 'FAIL'}"
