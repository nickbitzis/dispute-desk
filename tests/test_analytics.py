import pandas as pd

from disputedesk import analytics


def frames(monthly_sales, monthly_disputes):
    rows, drows, i = [], [], 0
    for m, n in monthly_sales.items():
        for _ in range(n):
            i += 1
            rows.append({"txn_id": i, "ts": pd.Timestamp(f"{m}-10"), "amount": 50.0})
    for m, n in monthly_disputes.items():
        for k in range(n):
            drows.append({"dispute_id": f"D{m}{k}", "received_date": pd.Timestamp(f"{m}-15").date(), "amount": 50.0})
    return pd.DataFrame(rows), pd.DataFrame(drows)


def test_monitor_uses_prior_month_and_flags_breach():
    tx, d = frames({"2026-01": 1000, "2026-02": 1000, "2026-03": 1000}, {"2026-02": 5, "2026-03": 20})
    mon = analytics.monthly_monitor(tx, d, threshold=0.015, warning=0.01)
    by = {m["month"]: m for m in mon["months"]}
    assert "2026-01" not in by  # no prior month
    assert by["2026-02"]["rate"] == 0.005 and by["2026-02"]["status"] == "ok"
    assert by["2026-03"]["status"] == "breach" and mon["first_breach"] == "2026-03"


def test_monitor_early_warning_level():
    tx, d = frames({"2026-01": 1000, "2026-02": 1000}, {"2026-02": 12})
    mon = analytics.monthly_monitor(tx, d, threshold=0.015, warning=0.01)
    assert mon["latest"]["status"] == "warning" and mon["first_warning"] == "2026-02"


def test_monitor_trend_projection_warns_before_breach():
    sales = {f"2026-0{m}": 1000 for m in range(1, 6)}
    tx, d = frames(sales, {"2026-02": 4, "2026-03": 7, "2026-04": 10})
    mon = analytics.monthly_monitor(tx, d, threshold=0.012, warning=0.011)
    apr = next(m for m in mon["months"] if m["month"] == "2026-04")
    assert apr["rate"] < 0.011 and apr["projected_next"] >= 0.012 and apr["status"] == "warning"


def test_three_ds_threshold_tradeoff(built):
    _, report = built
    rows = report["analytics"]["three_ds"]
    challenged = [r["challenged_orders"] for r in rows]
    prevented = [r["fraud_disputes_prevented"] for r in rows]
    assert challenged == sorted(challenged, reverse=True)  # higher threshold, fewer challenges
    assert prevented == sorted(prevented, reverse=True)
    assert all(0 <= r["share_of_fraud_disputes"] <= 1 for r in rows)


def test_rates_and_friendly_fraud(built):
    _, report = built
    a = report["analytics"]
    assert {r["product_type"] for r in a["by_product_type"]} == {"ticket", "merch", "membership"}
    assert sum(r["disputes"] for r in a["by_reason"]) == a["totals"]["disputes"]
    ff = a["friendly_fraud"]
    assert ff["repeat_customers"] > 0 and 0 < ff["share_of_disputes"] < 1
    assert a["monitor"]["latest"]["status"] in {"warning", "breach"}  # the autumn festival wave
