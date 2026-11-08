import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from disputedesk.app import create_app
from disputedesk.generator import AS_OF

H = {"x-analyst": "k.osei"}


@pytest.fixture(scope="module")
def client(built):
    data_dir, _ = built
    return TestClient(create_app(data_dir, today=AS_OF))


def open_ids(client, n=50):
    return [r["dispute_id"] for r in client.get(f"/api/queue?limit={n}").json()["items"]]


def test_healthz_and_index(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["disputes"] > 500
    r = client.get("/")
    assert r.status_code == 200 and "Dispute Desk" in r.text


def test_healthz_not_ready_without_data(tmp_path):
    assert TestClient(create_app(tmp_path)).get("/healthz").status_code == 503


def test_queue_sorted_by_deadline_then_ev(client):
    items = client.get("/api/queue?limit=200").json()["items"]
    assert items and all(i["status"] in ("received", "under_review") for i in items)
    keys = [(i["deadline"], -i["ev"]) for i in items]
    assert keys == sorted(keys)


def test_queue_sort_by_ev_and_filters(client):
    items = client.get("/api/queue?sort=ev&status=all&limit=100").json()["items"]
    evs = [i["ev"] for i in items]
    assert evs == sorted(evs, reverse=True)
    fr = client.get("/api/queue?status=all&category=fraud&limit=20").json()["items"]
    assert fr and all(i["category"] == "fraud" for i in fr)
    assert client.get("/api/queue?status=bogus").status_code == 400
    assert client.get("/api/queue?sort=amount").status_code == 400


def test_detail_hides_truth_and_has_decision(client):
    did = open_ids(client)[0]
    d = client.get(f"/api/disputes/{did}").json()
    blob = json.dumps(d)
    assert "won_if_fought" not in blob and "p_true" not in blob and "is_fraud" not in blob
    dec = d["decision"]
    assert 0 <= dec["p_win"] <= 1 and dec["recommendation"] in ("represent", "accept")
    assert d["evidence"] and d["evidence"][0]["id"] == "E1"
    assert d["audit"][0]["action"] == "received"


def test_unknown_dispute_404(client):
    assert client.get("/api/disputes/DP-0").status_code == 404


def test_packet_json_and_html(client):
    did = open_ids(client)[1]
    p = client.get(f"/api/disputes/{did}/packet").json()
    assert p["dispute_id"] == did and "completeness" in p and "decision" in p
    h = client.get(f"/api/disputes/{did}/packet.html")
    assert h.status_code == 200 and "evidence packet" in h.text.lower()


def test_letter_then_represent_flow_with_audit(client):
    did = next(i for i in open_ids(client, 200) if client.get(f"/api/disputes/{i}").json()["state"]["days_left"] >= 2)
    st = client.get(f"/api/disputes/{did}").json()["state"]
    r = client.post(f"/api/disputes/{did}/transition", json={"to": "represented"}, headers=H)
    assert r.status_code == 409  # no packet yet, or not under review
    if st["status"] == "received":
        assert client.post(f"/api/disputes/{did}/transition", json={"to": "under_review"}, headers=H).status_code == 200
    r = client.post(f"/api/disputes/{did}/letter", json={"provider": "template"}, headers=H)
    assert r.status_code == 200 and r.json()["packet_ready"]
    r = client.post(f"/api/disputes/{did}/transition", json={"to": "represented", "note": "submitted"}, headers=H)
    assert r.status_code == 200 and r.json()["status"] == "represented"
    ev = client.get(f"/api/disputes/{did}/audit").json()["events"]
    assert [e["action"] for e in ev][-2:] == ["letter_drafted", "transition"]
    assert ev[-1]["actor"] == "k.osei" and ev[-1]["to_status"] == "represented"
    p = client.get(f"/api/disputes/{did}/packet").json()
    assert p["letter"]["cited"]


def test_illegal_transition_409(client):
    did = client.get("/api/queue?status=won&limit=1").json()["items"][0]["dispute_id"]
    r = client.post(f"/api/disputes/{did}/transition", json={"to": "under_review"}, headers=H)
    assert r.status_code == 409 and "cannot move" in r.json()["error"]


def test_claude_provider_with_fake_client(built):
    data_dir, _ = built

    class Msgs:
        def create(self, **kw):
            evidence = json.loads(kw["messages"][0]["content"].split("\n", 1)[1])["evidence"]
            missing = [e["id"] for e in evidence if e["status"] == "missing"]
            text = json.dumps({"paragraphs": [{"text": "Cites a missing item.", "cites": missing[:1] or ["E99"]}]})
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn", usage=None
            )

    c = TestClient(create_app(data_dir, today=AS_OF, letter_client=SimpleNamespace(messages=Msgs())))
    did = open_ids(c)[3]
    r = c.post(f"/api/disputes/{did}/letter", json={"provider": "claude"}, headers=H)
    assert r.status_code == 422 and r.json()["errors"]
    assert c.get(f"/api/disputes/{did}/audit").json()["events"][-1]["action"] == "letter_rejected"
    assert c.post(f"/api/disputes/{did}/letter", json={"provider": "nope"}).status_code == 400


def test_sweep_dry_run_then_apply(client):
    dry = client.post("/api/sweep", json={"dry_run": True}).json()
    assert dry["dry_run"] and dry["actions"]
    before = client.get("/api/queue?status=accepted&limit=1").json()["total"]
    applied = client.post("/api/sweep", json={}).json()
    auto = sum(v for k, v in applied["summary"].items() if k.startswith("auto_accept"))
    after = client.get("/api/queue?status=accepted&limit=1").json()["total"]
    assert after == before + auto
    neg = [a for a in applied["actions"] if a["reason"] == "negative_ev"]
    if neg:
        d = client.get(f"/api/disputes/{neg[0]['dispute_id']}").json()
        assert d["state"]["status"] == "accepted" and d["audit"][-1]["actor"] == "system"


def test_audit_trail_is_append_only(built):
    con = sqlite3.connect(built[0] / "disputedesk.db")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE events SET actor='x' WHERE id=1")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM events WHERE id=1")
    con.close()


def test_analytics_policy_model_and_codes(client):
    a = client.get("/api/analytics").json()
    assert a["monitor"]["months"] and a["three_ds"]
    p = client.get("/api/policy").json()
    assert {r["policy"] for r in p["policies"]} >= {"fight_everything", "fight_nothing", "rule_of_thumb", "model_ev"}
    assert client.get("/api/model").json()["metrics"]["test"]["auc"] > 0.5
    codes = client.get("/api/reason-codes").json()
    assert len(codes) == 13


def test_metrics_prometheus_format(client):
    client.get("/healthz")
    text = client.get("/metrics").text
    assert "# TYPE dd_http_requests_total counter" in text
    assert "dd_disputes_received" in text and "dd_dispute_rate_threshold 0.0125" in text
    assert 'dd_http_request_duration_seconds_bucket{le="+Inf"}' in text
