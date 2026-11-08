"""Evidence templates per reason category, fact gathering from order data, and the evidence packet.

A template lists what each category needs: required items (without them the dispute is rarely worth fighting) and
helpful items. The gatherer checks each one against the order data as of the dispute date and writes a plain
sentence of fact for every item it finds. The packet is JSON (what is submitted and stored) plus an HTML page.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import datetime

import pandas as pd

from .codes import CATEGORY_LABELS

# key -> (label, applies to which product types)
ITEMS: dict[str, tuple[str, set[str]]] = {
    "order_receipt": ("Order receipt and terms accepted at checkout", {"ticket", "merch", "membership"}),
    "three_ds_result": ("3-D Secure authentication result", {"ticket", "merch", "membership"}),
    "avs_cvv": ("Address verification and CVV result", {"ticket", "merch", "membership"}),
    "prior_device_purchases": ("Earlier undisputed purchases from the same device", {"ticket", "merch", "membership"}),
    "ip_geolocation": ("IP geolocation near the billing address", {"ticket", "merch", "membership"}),
    "venue_scan": ("Ticket scanned at the venue gate", {"ticket"}),
    "delivery_proof": ("Carrier delivery confirmation", {"merch"}),
    "shipping_address": ("Shipped to the billing address", {"merch"}),
    "product_description": ("Product description shown at purchase", {"ticket", "merch"}),
    "cs_log": ("Customer service log", {"ticket", "merch", "membership"}),
    "separate_order": ("Second charge is a separate order that was fulfilled", {"ticket", "merch"}),
    "refund_record": ("Refund issued before the dispute", {"ticket", "merch", "membership"}),
    "refund_policy": ("No-refund policy for event tickets, accepted at checkout", {"ticket"}),
    "cancellation_log": ("Membership cancellation log", {"membership"}),
    "membership_terms": ("Membership renewal terms accepted at signup", {"membership"}),
}

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "fraud": {
        "required": ["three_ds_result", "prior_device_purchases"],
        "helpful": ["avs_cvv", "ip_geolocation", "venue_scan", "delivery_proof", "shipping_address", "order_receipt"],
        "any_of_required": ["three_ds_result", "prior_device_purchases"],
    },
    "not_received": {
        "required": ["delivery_proof", "venue_scan"],
        "helpful": ["shipping_address", "cs_log", "order_receipt"],
        "any_of_required": ["delivery_proof", "venue_scan"],
    },
    "not_as_described": {
        "required": ["product_description"],
        "helpful": ["cs_log", "venue_scan", "delivery_proof", "order_receipt"],
    },
    "duplicate": {"required": ["separate_order"], "helpful": ["order_receipt", "delivery_proof", "venue_scan"]},
    "credit_not_processed": {
        "required": ["refund_record", "refund_policy"],
        "helpful": ["cs_log", "order_receipt", "venue_scan"],
        "any_of_required": ["refund_record", "refund_policy"],
    },
    "canceled_recurring": {
        "required": ["cancellation_log"],
        "helpful": ["membership_terms", "cs_log", "order_receipt"],
    },
}


def _ts(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")


def _before(v, received: datetime) -> bool:
    s = _ts(v)
    return s is not None and pd.Timestamp(v) < pd.Timestamp(received)


def _money(v: float) -> str:
    return f"${v:,.2f}"


Gatherer = Callable[[dict], tuple[bool, str, dict]]


def _g_order(c):
    t = c["txn"]
    return (
        True,
        (
            f"Order {t['txn_id']} on {_ts(t['ts'])} for {_money(t['amount'])}: {t['product'].replace('_', ' ')} "
            f"({t['product_type']}) via {t['channel']}. Terms of sale accepted at checkout."
        ),
        {"txn_id": t["txn_id"], "amount": t["amount"]},
    )


def _g_3ds(c):
    v = c["txn"]["three_ds"]
    if v == "none":
        return False, "No 3-D Secure authentication on this order.", {"three_ds": v}
    word = "fully authenticated by the issuer" if v == "authenticated" else "attempted; the issuer did not take part"
    return True, f"3-D Secure was {word} at purchase.", {"three_ds": v}


def _g_avs(c):
    t = c["txn"]
    ok = t["avs"] == "match" and bool(t["cvv_match"])
    return (
        ok,
        f"AVS result {t['avs'].replace('_', ' ')}, CVV {'match' if t['cvv_match'] else 'no match'}.",
        {
            "avs": t["avs"],
            "cvv_match": bool(t["cvv_match"]),
        },
    )


def _g_device(c):
    hist = c.get("device_history", [])
    if len(hist) < 2:
        return (
            False,
            f"{len(hist)} earlier undisputed purchases from this device (two or more are needed).",
            {
                "device_id": c["txn"]["device_id"],
                "orders": hist,
            },
        )
    ids = ", ".join(str(h["txn_id"]) for h in hist[:5])
    return (
        True,
        (
            f"Device {c['txn']['device_id']} made {len(hist)} earlier purchases on this account that were never "
            f"disputed (orders {ids})."
        ),
        {"device_id": c["txn"]["device_id"], "orders": hist[:10]},
    )


def _g_ip(c):
    t = c["txn"]
    ok = t["ip_distance_km"] < 60 and t["ip_country"] == "US"
    return (
        ok,
        f"IP geolocated {t['ip_distance_km']:,.0f} km from the billing address, country {t['ip_country']}.",
        {
            "ip_distance_km": t["ip_distance_km"],
            "ip_country": t["ip_country"],
        },
    )


def _g_scan(c):
    t = c["txn"]
    if not _before(t["scan_ts"], c["received"]):
        return False, "No venue scan recorded before the dispute.", {}
    return (
        True,
        f"Ticket scanned at gate {t['gate']} on {_ts(t['scan_ts'])}, before the event at {_ts(t['event_ts'])}.",
        {
            "scan_ts": _ts(t["scan_ts"]),
            "gate": t["gate"],
        },
    )


def _g_delivery(c):
    t = c["txn"]
    if not _before(t["delivered_ts"], c["received"]):
        return (
            False,
            f"Carrier status: {t['tracking'].replace('_', ' ')}. No delivery confirmation.",
            {"tracking": t["tracking"]},
        )
    sig = " with signature" if t["tracking"] == "delivered_signature" else ""
    return (
        True,
        f"Carrier confirms delivery{sig} on {_ts(t['delivered_ts'])}.",
        {
            "tracking": t["tracking"],
            "delivered_ts": _ts(t["delivered_ts"]),
        },
    )


def _g_ship(c):
    ok = bool(c["txn"]["ship_bill_match"])
    return ok, "Shipped to the billing address." if ok else "Shipping address differs from the billing address.", {}


def _g_desc(c):
    return (
        True,
        f"Listing for {c['txn']['product'].replace('_', ' ')} as displayed at checkout, archived with the order.",
        {},
    )


def _g_cs(c):
    t = c["txn"]
    if not _before(t["cs_ts"], c["received"]):
        return True, "No customer service contact before the dispute was raised.", {"contacts": 0}
    if t["cs_complaint"]:
        return (
            False,
            f"{t['cs_contacts']} contact(s); a complaint was logged on {_ts(t['cs_ts'])}.",
            {
                "contacts": int(t["cs_contacts"]),
                "complaint": True,
            },
        )
    return (
        True,
        f"{t['cs_contacts']} contact(s) starting {_ts(t['cs_ts'])}, no complaint about the order.",
        {
            "contacts": int(t["cs_contacts"]),
            "complaint": False,
        },
    )


def _g_sep(c):
    t = c["txn"]
    if t["dup_of"] < 0:
        return (
            True,
            f"Charge {t['txn_id']} is its own order, placed separately from order {-t['dup_of']} and fulfilled.",
            {"other_order": int(-t["dup_of"])},
        )
    if t["dup_of"] > 0:
        return False, f"Charge {t['txn_id']} repeats order {t['dup_of']} within minutes with no second fulfillment.", {}
    return False, "No second charge found for the same amount.", {}


def _g_refund(c):
    t = c["txn"]
    if _before(t["refund_ts"], c["received"]):
        return (
            True,
            f"Refund of {_money(t['amount'])} issued on {_ts(t['refund_ts'])}.",
            {"refund_ts": _ts(t["refund_ts"])},
        )
    return False, "No refund issued before the dispute.", {}


def _g_policy(c):
    ok = c["txn"]["product_type"] == "ticket"
    return ok, "Event tickets are non-refundable except for canceled events; accepted at checkout.", {}


def _g_cancel(c):
    t = c["txn"]
    if _ts(t["cancel_ts"]) and pd.Timestamp(t["cancel_ts"]) < pd.Timestamp(t["ts"]):
        return (
            False,
            f"Member canceled on {_ts(t['cancel_ts'])}, before this charge.",
            {"cancel_ts": _ts(t["cancel_ts"])},
        )
    return True, "No cancellation request on file before this renewal charge.", {}


def _g_terms(c):
    return True, "Monthly auto-renewal terms accepted at signup, with a cancel link in every renewal email.", {}


GATHERERS: dict[str, Gatherer] = {
    "order_receipt": _g_order,
    "three_ds_result": _g_3ds,
    "avs_cvv": _g_avs,
    "prior_device_purchases": _g_device,
    "ip_geolocation": _g_ip,
    "venue_scan": _g_scan,
    "delivery_proof": _g_delivery,
    "shipping_address": _g_ship,
    "product_description": _g_desc,
    "cs_log": _g_cs,
    "separate_order": _g_sep,
    "refund_record": _g_refund,
    "refund_policy": _g_policy,
    "cancellation_log": _g_cancel,
    "membership_terms": _g_terms,
}


def gather(context: dict) -> list[dict]:
    """context: dispute (dict), txn (dict), received (date), device_history (list). Returns checklist items."""
    cat = context["dispute"]["category"]
    ptype = context["txn"]["product_type"]
    tpl = TEMPLATES[cat]
    items = []
    n = 0
    for role in ("required", "helpful"):
        for key in tpl[role]:
            label, applies = ITEMS[key]
            if ptype not in applies:
                continue
            n += 1
            available, fact, data = GATHERERS[key](context)
            items.append(
                {
                    "id": f"E{n}",
                    "key": key,
                    "label": label,
                    "role": role,
                    "available": bool(available),
                    "fact": fact,
                    "data": data,
                }
            )
    return items


def completeness(category: str, items: list[dict]) -> dict:
    tpl = TEMPLATES[category]
    req = [i for i in items if i["role"] == "required"]
    if tpl.get("any_of_required"):
        ok = any(i["available"] for i in req)
    else:
        ok = all(i["available"] for i in req)
    return {
        "required_met": ok,
        "required_total": len(req),
        "required_available": sum(i["available"] for i in req),
        "helpful_available": sum(i["available"] for i in items if i["role"] == "helpful"),
        "rule": "any one required item" if tpl.get("any_of_required") else "all required items",
    }


def build_packet(context: dict, decision: dict | None = None, letter: dict | None = None) -> dict:
    d = context["dispute"]
    items = gather(context)
    return {
        "dispute_id": d["dispute_id"],
        "reason_code": d["reason_code"],
        "network": d["network"],
        "category": d["category"],
        "category_label": CATEGORY_LABELS[d["category"]],
        "amount": d["amount"],
        "txn_id": int(context["txn"]["txn_id"]),
        "received_date": str(context["received"]),
        "items": items,
        "completeness": completeness(d["category"], items),
        "decision": decision,
        "letter": letter,
    }


PACKET_CSS = """:root { --bg:#fff; --text:#1c2330; --muted:#5d6878; --border:#dde2e9; --ok:#11804f; --no:#b42318;
  color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg:#11151c; --text:#e6eaf0; --muted:#9aa5b5; --border:#2a3340;
  --ok:#4ccb8f; --no:#ff8a7a; color-scheme: dark; } }
body { font-family: Georgia, serif; max-width: 820px; margin: 24px auto; padding: 0 16px; background: var(--bg);
  color: var(--text); }
h1 { font-size: 22px; } h2 { font-size: 16px; margin-top: 28px; }
.meta { color: var(--muted); font-family: system-ui, sans-serif; font-size: 13px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-family: system-ui, sans-serif; font-size: 13px; }
td, th { border-top: 1px solid var(--border); padding: 6px; text-align: left; vertical-align: top; }
.ok { color: var(--ok); font-weight: 600; } .no { color: var(--no); font-weight: 600; }
.role { color: var(--muted); font-size: 11px; }
.cite { color: var(--muted); font-family: system-ui, sans-serif; font-size: 12px; }
"""


def render_packet_html(packet: dict) -> str:
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(i['id'])}</td><td>{e(i['label'])}<div class='role'>{e(i['role'])}</div></td>"
        f"<td class='{'ok' if i['available'] else 'no'}'>{'Available' if i['available'] else 'Missing'}</td>"
        f"<td>{e(i['fact'])}</td></tr>"
        for i in packet["items"]
    )
    letter = ""
    if packet.get("letter"):
        paras = "".join(
            f"<p>{e(p['text'])} <span class='cite'>[{', '.join(e(c) for c in p['cites'])}]</span></p>"
            for p in packet["letter"]["paragraphs"]
        )
        letter = f"<h2>Rebuttal letter</h2><div class='letter'>{paras}</div>"
    c = packet["completeness"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evidence packet {e(packet["dispute_id"])}</title>
<style>
{PACKET_CSS}</style></head><body>
<h1>Representment evidence packet</h1>
<div class="meta">Brightgate Tickets (synthetic merchant). Dispute {e(packet["dispute_id"])}, reason code
{e(packet["reason_code"])} ({e(packet["category_label"])}), network {e(packet["network"])}, order {packet["txn_id"]},
amount ${packet["amount"]:,.2f}, received {e(packet["received_date"])}.</div>
<div class="meta">Required evidence: {c["required_available"]} of {c["required_total"]} available
({e(c["rule"])}): {"met" if c["required_met"] else "not met"}.</div>
<h2>Evidence</h2><div class="wrap"><table>
<tr><th>ID</th><th>Item</th><th>Status</th><th>Fact</th></tr>{rows}</table></div>
{letter}
</body></html>"""
