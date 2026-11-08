"""Seeded synthetic data for Brightgate Tickets, a made-up online seller of event tickets, merchandise, and a fan club
membership.

One year of card-not-present orders with the facts a dispute team actually uses (device, IP geolocation, AVS and CVV,
3-D Secure, shipping and tracking, venue scans, refunds, customer service contacts, membership cancellations), then
the disputes those orders produce: stolen cards, friendly fraud, real fulfillment failures, duplicate charges, missed
refunds, and charges after a cancellation.

A hidden evidence model decides how likely each dispute is to win if represented. The win model in model.py never
sees it; it only sees which evidence exists. Every dispute gets a true outcome if represented, so any fight policy can
be scored on the same disputes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .codes import CATEGORIES, codes_for

START = pd.Timestamp("2025-11-01")
END = pd.Timestamp("2026-10-31 23:59:59")
AS_OF = date(2026, 11, 6)  # the demo's "today": a Friday a week after the last order

TICKET_CATS = ["concert", "sports", "theater", "festival"]
MERCH_CATS = ["apparel", "vinyl", "posters", "accessories"]
GATES = ["North A", "North B", "South A", "South B", "East", "West", "VIP"]
TRACK = ["none", "in_transit", "delivered", "delivered_signature"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_txns: int = 300_000, n_customers: int = 90_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # Customers
    cust_signup = START - pd.to_timedelta(rng.integers(0, 4 * 365, n_customers), unit="D")
    repeat_disputer = rng.random(n_customers) < 0.012
    n_dev = rng.integers(1, 3, n_customers)

    # Orders over the year with seasonality: summer festivals and December gifts.
    day = np.arange(365)
    month_of_day = (START + pd.to_timedelta(day, unit="D")).month.to_numpy()
    season = np.ones(365)
    season[np.isin(month_of_day, [6, 7, 8])] = 1.35
    season[month_of_day == 12] = 1.25
    season[month_of_day == 1] = 0.92
    days = rng.choice(day, size=n_txns, p=season / season.sum())
    secs = rng.integers(7 * 3600, 24 * 3600, n_txns)
    ts = (START + pd.to_timedelta(days, unit="D") + pd.to_timedelta(secs, unit="s")).to_numpy()
    order = np.argsort(ts, kind="stable")
    ts = ts[order]

    cust = rng.integers(0, n_customers, n_txns)
    fraud = rng.random(n_txns) < 0.004  # stolen card used by someone else

    ptype = rng.choice(["ticket", "merch", "membership"], size=n_txns, p=[0.64, 0.29, 0.07])
    is_ticket, is_merch, is_member = ptype == "ticket", ptype == "merch", ptype == "membership"
    product = np.where(
        is_ticket,
        rng.choice(TICKET_CATS, n_txns, p=[0.42, 0.28, 0.14, 0.16]),
        np.where(is_merch, rng.choice(MERCH_CATS, n_txns), "fan_club"),
    )
    channel = rng.choice(["web", "app", "phone"], size=n_txns, p=[0.56, 0.36, 0.08])

    amount = np.where(
        is_ticket,
        np.clip(rng.lognormal(np.log(120), 0.65, n_txns), 25, 1200),
        np.where(is_merch, np.clip(rng.lognormal(np.log(42), 0.55, n_txns), 12, 320), 0),
    )
    amount = np.where(is_member, rng.choice([9.99, 14.99], n_txns), amount)
    amount = np.where(fraud & ~is_member, amount * rng.uniform(1.2, 2.2, n_txns), amount)  # fraudsters buy big
    amount = np.round(amount, 2)

    # Device and network signals
    dev_slot = rng.integers(0, 2, n_txns) % n_dev[cust]
    known_device = ~fraud & (rng.random(n_txns) < 0.9)
    device = np.where(known_device, [f"D{c}-{s}" for c, s in zip(cust, dev_slot, strict=True)], "")
    new_dev_ids = np.array([f"N{i}" for i in range(n_txns)])
    device = np.where(device == "", new_dev_ids, device)
    # A fraud ring reuses a few hundred devices.
    ring_dev = np.array([f"R{i}" for i in rng.integers(0, 300, n_txns)])
    device = np.where(fraud & (rng.random(n_txns) < 0.4), ring_dev, device)

    ip_km = np.where(fraud, rng.lognormal(np.log(900), 0.9, n_txns), rng.lognormal(np.log(18), 1.0, n_txns))
    ip_km = np.round(np.clip(ip_km, 0, 12000), 1)
    ip_country = np.where(fraud & (rng.random(n_txns) < 0.35), rng.choice(["CA", "GB", "NG", "BR", "RO"], n_txns), "US")

    # 3-D Secure: the merchant challenges most orders above $250 already; fraudsters favor unchallenged ones.
    p3 = np.where(amount >= 250, 0.8, 0.28)
    p3 = np.where(fraud, p3 * 0.35, p3)
    u = rng.random(n_txns)
    three_ds = np.where(u < p3, "authenticated", np.where(u < p3 + 0.05, "attempted", "none"))
    avs = np.where(
        fraud,
        rng.choice(["match", "partial", "no_match"], n_txns, p=[0.35, 0.25, 0.40]),
        rng.choice(["match", "partial", "no_match"], n_txns, p=[0.88, 0.08, 0.04]),
    )
    cvv = np.where(fraud, rng.random(n_txns) < 0.8, rng.random(n_txns) < 0.985)

    # Shipping for merchandise
    ship_match = np.where(is_merch, np.where(fraud, rng.random(n_txns) < 0.3, rng.random(n_txns) < 0.9), True)
    tr_u = rng.random(n_txns)
    tracking = np.where(
        ~is_merch,
        "none",
        np.where(
            tr_u < 0.03,
            "none",
            np.where(tr_u < 0.07, "in_transit", np.where(amount >= 120, "delivered_signature", "delivered")),
        ),
    )
    tracking = np.where(
        is_merch & (tracking == "delivered_signature") & (rng.random(n_txns) < 0.35), "delivered", tracking
    )
    ship_days = rng.integers(2, 9, n_txns)
    delivered_ts = np.where(
        np.isin(tracking, ["delivered", "delivered_signature"]),
        ts + pd.to_timedelta(ship_days, unit="D").to_numpy(),
        np.datetime64("NaT", "ns"),
    )

    # Tickets: event date and venue scan
    event_ts = np.where(
        is_ticket,
        ts + pd.to_timedelta(rng.integers(3, 120, n_txns), unit="D").to_numpy(),
        np.datetime64("NaT", "ns"),
    )
    event_ts = pd.to_datetime(event_ts).normalize() + pd.Timedelta(hours=19)
    p_scan = np.where(fraud, 0.55, 0.93)
    scanned = is_ticket & (rng.random(n_txns) < p_scan)
    scan_ts = np.where(
        scanned,
        event_ts.to_numpy() - pd.to_timedelta(rng.integers(5, 90, n_txns), unit="m").to_numpy(),
        np.datetime64("NaT", "ns"),
    )
    gate = np.where(scanned, rng.choice(GATES, n_txns), "")

    # Refunds and customer service
    refunded = rng.random(n_txns) < 0.03
    refund_ts = np.where(
        refunded, ts + pd.to_timedelta(rng.integers(1, 30, n_txns), unit="D").to_numpy(), np.datetime64("NaT", "ns")
    )
    p_cs = 0.06 + 0.25 * (tracking == "in_transit") + 0.15 * refunded
    cs_contacts = np.where(rng.random(n_txns) < p_cs, rng.integers(1, 4, n_txns), 0)
    cs_complaint = (cs_contacts > 0) & (rng.random(n_txns) < 0.45)
    cs_ts = np.where(
        cs_contacts > 0,
        ts + pd.to_timedelta(rng.integers(1, 20, n_txns), unit="D").to_numpy(),
        np.datetime64("NaT", "ns"),
    )

    # Membership: some members cancel and get charged again anyway (a billing bug).
    canceled_before = is_member & (rng.random(n_txns) < 0.015)
    cancel_ts = np.where(
        canceled_before,
        ts - pd.to_timedelta(rng.integers(1, 20, n_txns), unit="D").to_numpy(),
        np.datetime64("NaT", "ns"),
    )

    tx = pd.DataFrame(
        {
            "txn_id": np.arange(1, n_txns + 1),
            "ts": ts,
            "customer_id": cust,
            "customer_signup": cust_signup[cust].to_numpy(),
            "product_type": ptype,
            "product": product,
            "channel": channel,
            "amount": amount,
            "device_id": device,
            "ip_country": ip_country,
            "ip_distance_km": ip_km,
            "three_ds": three_ds,
            "avs": avs,
            "cvv_match": cvv,
            "ship_bill_match": ship_match,
            "tracking": tracking,
            "delivered_ts": delivered_ts,
            "event_ts": event_ts.to_numpy(),
            "scan_ts": scan_ts,
            "gate": gate,
            "refund_ts": refund_ts,
            "cs_contacts": cs_contacts,
            "cs_complaint": cs_complaint,
            "cs_ts": cs_ts,
            "cancel_ts": cancel_ts,
            "is_fraud": fraud,
            "dup_of": 0,
        }
    )

    # Duplicate processing: a gateway retry charges a small share of orders twice within two minutes. A different,
    # legitimate case is a customer who places two separate orders for the same amount on the same day.
    dup_src = tx.sample(frac=0.0012, random_state=seed)
    dups = dup_src.copy()
    dups["ts"] = dups["ts"] + pd.to_timedelta(rng.integers(20, 120, len(dups)), unit="s")
    dups["dup_of"] = dup_src["txn_id"].to_numpy()
    for col in ["delivered_ts", "scan_ts", "refund_ts", "cs_ts"]:
        dups[col] = pd.NaT
    dups["tracking"] = np.where(dups["product_type"] == "merch", "none", dups["tracking"])
    dups["gate"] = ""
    twin_src = tx[tx["product_type"] != "membership"].sample(frac=0.0012, random_state=seed + 1)
    twins = twin_src.copy()
    twins["ts"] = twins["ts"] + pd.to_timedelta(rng.integers(2, 9, len(twins)), unit="h")
    twins["dup_of"] = -twin_src["txn_id"].to_numpy()  # negative: a separate, fulfilled order (not a duplicate)
    tx = pd.concat([tx, dups, twins], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    tx = tx[tx["ts"] <= END].reset_index(drop=True)
    tx["txn_id"] = np.arange(1, len(tx) + 1)
    # re-point dup_of to the new ids
    tx["_src"] = 0
    tx = _relink_duplicates(tx, dup_src, twin_src)

    disputes = _make_disputes(tx, repeat_disputer, rng)
    return tx, disputes


def _relink_duplicates(tx: pd.DataFrame, dup_src: pd.DataFrame, twin_src: pd.DataFrame) -> pd.DataFrame:
    """After re-numbering, point each duplicate or twin at its source order.

    Match on customer and amount, taking the nearest earlier order.
    """
    tx = tx.drop(columns="_src")
    flag = tx["dup_of"] != 0
    sign = np.sign(tx.loc[flag, "dup_of"]).to_numpy()
    base = tx[~flag][["txn_id", "customer_id", "amount", "ts"]].sort_values("ts")
    probe = tx[flag][["txn_id", "customer_id", "amount", "ts"]].sort_values("ts")
    m = pd.merge_asof(
        probe,
        base.rename(columns={"txn_id": "src_id", "ts": "src_ts"}),
        left_on="ts",
        right_on="src_ts",
        by=["customer_id", "amount"],
        direction="backward",
    )
    m = m.set_index("txn_id")["src_id"]
    ids = tx.loc[flag, "txn_id"].to_numpy()
    tx.loc[flag, "dup_of"] = (m.reindex(ids).fillna(0).to_numpy() * sign).astype(int)
    return tx


def _make_disputes(tx: pd.DataFrame, repeat_disputer: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(tx)
    ts = tx["ts"]
    month = ts.dt.month.to_numpy()
    is_merch = (tx["product_type"] == "merch").to_numpy()
    is_member = (tx["product_type"] == "membership").to_numpy()
    fraud = tx["is_fraud"].to_numpy()
    rep = repeat_disputer[tx["customer_id"].to_numpy()]

    cat = np.full(n, "", dtype=object)
    # Stolen cards: the real cardholder disputes most of them, fewer when the issuer itself authenticated.
    p_f = np.where(tx["three_ds"].to_numpy() == "authenticated", 0.35, 0.8)
    cat[fraud & (rng.random(n) < p_f)] = "fraud"

    # Real fulfillment failures and billing errors.
    lost = is_merch & np.isin(tx["tracking"].to_numpy(), ["none", "in_transit"]) & (tx["dup_of"].to_numpy() <= 0)
    cat[(cat == "") & lost & (rng.random(n) < 0.06)] = "not_received"
    true_dup = tx["dup_of"].to_numpy() > 0
    cat[(cat == "") & true_dup & (rng.random(n) < 0.6)] = "duplicate"
    twin = tx["dup_of"].to_numpy() < 0
    cat[(cat == "") & twin & (rng.random(n) < 0.3)] = "duplicate"
    after_cancel = tx["cancel_ts"].notna().to_numpy()
    cat[(cat == "") & after_cancel & (rng.random(n) < 0.55)] = "canceled_recurring"
    refunded = tx["refund_ts"].notna().to_numpy()
    cat[(cat == "") & refunded & (rng.random(n) < 0.02)] = "credit_not_processed"

    # Friendly fraud: legitimate orders disputed anyway. Higher for repeat disputers, and a wave of festival ticket
    # disputes in the last three months so the dispute-rate monitor has something to warn about.
    p_ff = np.where(rep, 0.06, 0.0032)
    p_ff = p_ff * np.where(is_member, 0.5, 1.0) * np.where(tx["channel"].to_numpy() == "phone", 1.4, 1.0)
    wave = np.isin(month, [8, 9, 10]) & (tx["product"].to_numpy() == "festival")
    p_ff = np.where(wave, p_ff * 6.0, p_ff)
    p_ff = p_ff * np.where(np.isin(month, [8, 9, 10]), 1.25, 1.0)
    ff = (cat == "") & ~fraud & (rng.random(n) < p_ff)
    k = ff.sum()
    ff_cat = rng.choice(
        ["fraud", "not_received", "not_as_described", "credit_not_processed"], size=k, p=[0.5, 0.2, 0.22, 0.08]
    )
    ff_cat = np.where(is_member[ff], "canceled_recurring", ff_cat)
    cat[ff] = ff_cat

    idx = np.flatnonzero(cat != "")
    d = tx.iloc[idx][["txn_id", "ts", "amount", "customer_id"]].copy()
    d["category"] = cat[idx]
    d["friendly"] = ff[idx]
    lag = np.where(d["category"].to_numpy() == "fraud", rng.integers(4, 60, len(d)), rng.integers(10, 75, len(d)))
    received = (d["ts"] + pd.to_timedelta(lag, unit="D")).dt.normalize()
    received = received + pd.to_timedelta((received.dt.day_of_week >= 5) * (7 - received.dt.day_of_week), unit="D")
    d["received_date"] = received.dt.date
    d = d[d["received_date"] <= AS_OF].copy()
    d["network"] = rng.choice(["A", "B"], len(d), p=[0.6, 0.4])
    d["reason_code"] = [rng.choice(codes_for(c, net)) for c, net in zip(d["category"], d["network"], strict=True)]
    d = d.sort_values(["received_date", "txn_id"]).reset_index(drop=True)
    d.insert(0, "dispute_id", [f"DP-{i + 100001}" for i in range(len(d))])
    d = d.drop(columns=["ts"])

    p = true_win_probability(d, tx.set_index("txn_id"), rng)
    d["p_true"] = np.round(p, 4)
    d["won_if_fought"] = rng.random(len(d)) < p
    # Some first-cycle wins are escalated by the issuer to pre-arbitration. The final outcome already accounts for it.
    d["pre_arb"] = d["won_if_fought"] & (rng.random(len(d)) < 0.12)
    return d


def true_win_probability(d: pd.DataFrame, txi: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """The hidden evidence model: what makes each reason category winnable. The win model never reads this."""
    t = txi.loc[d["txn_id"].to_numpy()]
    rec = pd.to_datetime(d["received_date"]).to_numpy()
    cat = d["category"].to_numpy()
    scanned = (t["scan_ts"].notna() & (t["scan_ts"].to_numpy() < rec)).to_numpy()
    delivered = np.isin(t["tracking"].to_numpy(), ["delivered", "delivered_signature"])
    signature = t["tracking"].to_numpy() == "delivered_signature"
    lost = np.isin(t["tracking"].to_numpy(), ["none", "in_transit"]) & (t["product_type"].to_numpy() == "merch")
    complaint = (t["cs_complaint"].to_numpy()) & (t["cs_ts"].to_numpy() < rec)
    no_contact = ~(t["cs_ts"].notna().to_numpy() & (t["cs_ts"].to_numpy() < rec))
    avs_cvv = (t["avs"].to_numpy() == "match") & t["cvv_match"].to_numpy().astype(bool)
    near = t["ip_distance_km"].to_numpy() < 60
    known_dev = ~pd.Series(t["device_id"].to_numpy()).str.match(r"^[NR]").to_numpy()
    refund_before = (t["refund_ts"].notna() & (t["refund_ts"].to_numpy() < rec)).to_numpy()
    cancel_before = t["cancel_ts"].notna().to_numpy()
    true_dup = t["dup_of"].to_numpy() > 0
    is_ticket = t["product_type"].to_numpy() == "ticket"
    friendly = d["friendly"].to_numpy()
    three = t["three_ds"].to_numpy()

    z = np.zeros(len(d))
    f = cat == "fraud"
    z[f] = (
        -1.5
        + 1.7 * known_dev[f]
        + 0.6 * avs_cvv[f]
        + 0.9 * (scanned[f] & is_ticket[f])
        + 0.7 * (delivered[f] & t["ship_bill_match"].to_numpy()[f])
        + 0.4 * near[f]
        + 0.5 * friendly[f]
    )
    nr = cat == "not_received"
    z[nr] = -1.7 + 2.6 * signature[nr] + 1.5 * (delivered[nr] & ~signature[nr]) + 2.8 * scanned[nr] - 1.4 * lost[nr]
    z[nr] += 0.3 * no_contact[nr]
    nad = cat == "not_as_described"
    z[nad] = -1.3 + 0.9 * no_contact[nad] + 0.8 * scanned[nad] - 0.9 * complaint[nad] + 0.4 * is_ticket[nad]
    dup = cat == "duplicate"
    z[dup] = np.where(true_dup[dup], -3.0, 1.6)
    cnp = cat == "credit_not_processed"
    z[cnp] = -1.0 + 0.9 * is_ticket[cnp] + 0.5 * no_contact[cnp] - 0.5 * complaint[cnp]
    cr = cat == "canceled_recurring"
    z[cr] = np.where(cancel_before[cr], -2.8, 0.9 + 0.4 * no_contact[cr])

    z = z + 0.15 * np.log1p(d["amount"].to_numpy()) - 0.6 + rng.normal(0, 0.35, len(d))
    p = _sigmoid(z)
    p = np.where(refund_before, 0.93, p)
    p = np.where(f & (three == "authenticated"), np.maximum(p, 0.96), p)
    p = np.where(f & (three == "attempted"), np.maximum(p, 0.88), p)
    return np.clip(p, 0.01, 0.99)


CATEGORY_ORDER = CATEGORIES
