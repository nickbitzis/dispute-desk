from disputedesk.evidence import build_packet, completeness, gather, render_packet_html
from tests.conftest import make_context


def by_key(items):
    return {i["key"]: i for i in items}


def test_fraud_gathers_device_history_and_3ds():
    items = by_key(gather(make_context("fraud", history=3, three_ds="authenticated")))
    assert items["prior_device_purchases"]["available"]
    assert "3 earlier purchases" in items["prior_device_purchases"]["fact"]
    assert items["three_ds_result"]["available"]
    assert "delivery_proof" not in items  # merch-only item is skipped for a ticket


def test_fraud_with_one_prior_purchase_is_not_compelling():
    items = by_key(gather(make_context("fraud", history=1)))
    assert not items["prior_device_purchases"]["available"]
    assert not completeness("fraud", list(items.values()))["required_met"]


def test_not_received_ticket_uses_venue_scan_time_and_gate():
    items = by_key(gather(make_context("not_received", scan_ts="2026-03-20 18:20:00", gate="North A")))
    scan = items["venue_scan"]
    assert scan["available"] and "North A" in scan["fact"] and "2026-03-20 18:20" in scan["fact"]


def test_scan_after_dispute_date_does_not_count():
    items = by_key(gather(make_context("not_received", received="2026-03-15", scan_ts="2026-03-20 18:20:00")))
    assert not items["venue_scan"]["available"]


def test_not_received_merch_uses_delivery_proof():
    ctx = make_context(
        "not_received",
        product_type="merch",
        product="vinyl",
        tracking="delivered_signature",
        delivered_ts="2026-03-05 14:00:00",
    )
    items = by_key(gather(ctx))
    assert items["delivery_proof"]["available"] and "signature" in items["delivery_proof"]["fact"]
    assert "venue_scan" not in items
    assert completeness("not_received", list(items.values()))["required_met"]


def test_not_as_described_reads_customer_service_log():
    items = by_key(gather(make_context("not_as_described", cs_contacts=2, cs_complaint=1, cs_ts="2026-03-05 10:00:00")))
    assert not items["cs_log"]["available"] and "complaint" in items["cs_log"]["fact"]


def test_duplicate_separate_order_vs_retry():
    ok = by_key(gather(make_context("duplicate", dup_of=-499)))
    assert ok["separate_order"]["available"] and "499" in ok["separate_order"]["fact"]
    bad = by_key(gather(make_context("duplicate", dup_of=499)))
    assert not bad["separate_order"]["available"]


def test_credit_not_processed_refund_record():
    items = by_key(gather(make_context("credit_not_processed", refund_ts="2026-03-10 09:00:00")))
    assert items["refund_record"]["available"]


def test_canceled_recurring_cancel_before_charge_is_missing_evidence():
    ctx = make_context(
        "canceled_recurring",
        product_type="membership",
        product="fan_club",
        amount=9.99,
        cancel_ts="2026-02-20 09:00:00",
    )
    items = by_key(gather(ctx))
    assert not items["cancellation_log"]["available"]
    assert "membership_terms" in items


def test_item_ids_are_sequential_and_packet_html_escapes():
    ctx = make_context("fraud", history=2, gate="<b>x</b>")
    ctx["dispute"]["reason_code"] = "<script>"
    p = build_packet(ctx)
    assert [i["id"] for i in p["items"]] == [f"E{n}" for n in range(1, len(p["items"]) + 1)]
    html = render_packet_html(p)
    assert "<script>" not in html and "&lt;script&gt;" in html
