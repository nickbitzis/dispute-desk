from __future__ import annotations

from pathlib import Path

import pytest

from disputedesk import pipeline


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> tuple[Path, dict]:
    """One pipeline run at reduced size (about 1,200 disputes), shared by the model, analytics, and API tests."""
    d = tmp_path_factory.mktemp("data")
    report = pipeline.run(d, n_txns=120_000, n_customers=36_000, seed=42, log=lambda m: None)
    return d, report


def make_txn(**kw) -> dict:
    base = {
        "txn_id": 500,
        "ts": "2026-03-01 12:00:00",
        "customer_id": 9,
        "product_type": "ticket",
        "product": "concert",
        "channel": "web",
        "amount": 180.0,
        "device_id": "D9-0",
        "ip_country": "US",
        "ip_distance_km": 12.0,
        "three_ds": "none",
        "avs": "match",
        "cvv_match": 1,
        "ship_bill_match": 1,
        "tracking": "none",
        "delivered_ts": None,
        "event_ts": "2026-03-20 19:00:00",
        "scan_ts": None,
        "gate": "",
        "refund_ts": None,
        "cs_contacts": 0,
        "cs_complaint": 0,
        "cs_ts": None,
        "cancel_ts": None,
        "dup_of": 0,
    }
    base.update(kw)
    return base


def make_context(category: str, received: str = "2026-04-10", history: int = 0, **txn) -> dict:
    from datetime import date

    t = make_txn(**txn)
    return {
        "dispute": {
            "dispute_id": "DP-1",
            "category": category,
            "reason_code": "A-0",
            "network": "A",
            "amount": t["amount"],
        },
        "txn": t,
        "received": date.fromisoformat(received),
        "device_history": [{"txn_id": 100 + i, "ts": "2025-12-01", "amount": 50.0} for i in range(history)],
    }
