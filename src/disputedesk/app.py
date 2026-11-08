"""FastAPI service: dispute queue, detail, decision, evidence packet, letters, transitions, sweep, analytics.

The data is synthetic and ends on a fixed date, so the service runs on a fixed clock (DISPUTEDESK_TODAY, default
the generator's as-of date). Every write goes to the append-only audit trail with the actor from x-analyst.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from . import evidence, letter, store
from .bizdays import business_days_between
from .codes import CATEGORY_LABELS, REASON_CODES
from .decision import decide
from .generator import AS_OF
from .lifecycle import OPEN, STATES, TERMINAL, TransitionError, days_left, sweep_one, transition
from .logs import get_logger, log_event
from .metrics import Metrics
from .model import WinModel, reasons

STATIC = Path(__file__).parent / "static"


def create_app(data_dir: Path | str | None = None, today: date | None = None, letter_client=None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("DISPUTEDESK_DATA_DIR", "data"))
    db = data_dir / "disputedesk.db"
    clock = today or date.fromisoformat(os.environ.get("DISPUTEDESK_TODAY", AS_OF.isoformat()))
    app = FastAPI(title="Dispute Desk", version="1.0.0")
    logger = get_logger()
    metrics = Metrics()
    cache: dict = {}

    def model() -> WinModel:
        if "model" not in cache:
            cache["model"] = WinModel.load(data_dir / "model.pkl")
        return cache["model"]

    def report() -> dict:
        if "report" not in cache:
            cache["report"] = json.loads((data_dir / "report.json").read_text())
        return cache["report"]

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log_event(logger, "unhandled error", request_id=rid, path=request.url.path)
            raise
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.observe_request(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if not path.startswith(("/static", "/metrics", "/healthz")):
            log_event(
                logger,
                "request",
                request_id=rid,
                method=request.method,
                route=path,
                status=response.status_code,
                ms=round(dt * 1000, 1),
            )
        return response

    def ready() -> bool:
        return db.exists() and (data_dir / "model.pkl").exists()

    @app.get("/healthz")
    def healthz():
        if not ready():
            return JSONResponse(
                {"status": "not_ready", "reason": f"run the pipeline first: no data in {data_dir}"}, 503
            )
        with store.session(db) as con:
            n = con.execute("SELECT COUNT(*) FROM disputes").fetchone()[0]
        return {"status": "ok", "disputes": n, "today": clock.isoformat()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        if ready():
            with store.session(db) as con:
                for s in STATES:
                    n = con.execute("SELECT COUNT(*) FROM state WHERE status=?", (s,)).fetchone()[0]
                    metrics.set_gauge(f"dd_disputes_{s}", f"Disputes currently in state {s}.", n)
            mon = report()["analytics"]["monitor"]["latest"]
            metrics.set_gauge("dd_dispute_rate_latest_month", "Latest monthly dispute rate (count based).", mon["rate"])
            metrics.set_gauge(
                "dd_dispute_rate_threshold",
                "Network dispute-rate threshold.",
                report()["analytics"]["monitor"]["threshold"],
            )
        return metrics.render()

    def need(con, dispute_id: str) -> dict:
        d = store.dispute_row(con, dispute_id)
        if d is None:
            raise HTTPException(404, f"dispute {dispute_id} not found")
        return d

    def context(con, d: dict) -> dict:
        txn = store.txn_row(con, d["txn_id"])
        return {
            "dispute": d,
            "txn": txn,
            "received": date.fromisoformat(d["received_date"]),
            "device_history": store.device_history(con, txn, d["received_date"]),
        }

    def decision_for(con, d: dict) -> dict:
        f = store.feature_row(con, d["dispute_id"])
        row = pd.DataFrame([{k: v for k, v in f.items() if k != "dispute_id"}])
        m = model()
        p = float(m.predict(row)[0])
        txn = store.txn_row(con, d["txn_id"])
        out = decide(p, d["amount"], d["category"], txn["three_ds"], bool(f["refund_before"]))
        out["reasons"] = reasons(m, row, category=d["category"])
        return out

    def state_view(con, dispute_id: str) -> dict:
        s = store.load_state(con, dispute_id)
        v = s.as_dict()
        v["days_left"] = days_left(s, clock)
        return v

    @app.get("/api/queue")
    def queue(status: str = "open", sort: str = "deadline", limit: int = 50, offset: int = 0, category: str = ""):
        limit = max(1, min(limit, 500))
        where, args = [], []
        if status == "open":
            where.append(f"s.status IN ({','.join('?' * len(OPEN))})")
            args += sorted(OPEN)
        elif status in STATES:
            where.append("s.status = ?")
            args.append(status)
        elif status != "all":
            raise HTTPException(400, f"unknown status {status!r}")
        if category:
            where.append("d.category = ?")
            args.append(category)
        order = {"deadline": "s.deadline IS NULL, s.deadline ASC, d.ev DESC", "ev": "d.ev DESC"}.get(sort)
        if order is None:
            raise HTTPException(400, "sort must be deadline or ev")
        sql_where = ("WHERE " + " AND ".join(where)) if where else ""
        with store.session(db) as con:
            total = con.execute(
                f"SELECT COUNT(*) FROM disputes d JOIN state s USING (dispute_id) {sql_where}", args
            ).fetchone()[0]
            rows = con.execute(
                f"""SELECT {store.PUBLIC_DISPUTE_COLS}, s.status, s.deadline, s.packet_ready, t.product, t.product_type
                    FROM disputes d JOIN state s USING (dispute_id) JOIN transactions t ON t.txn_id = d.txn_id
                    {sql_where} ORDER BY {order} LIMIT ? OFFSET ?""",
                [*args, limit, offset],
            ).fetchall()
        items = []
        for r in rows:
            r = dict(r)
            r["days_left"] = business_days_between(clock, date.fromisoformat(r["deadline"])) if r["deadline"] else None
            r["category_label"] = CATEGORY_LABELS[r["category"]]
            items.append(r)
        return {"today": clock.isoformat(), "total": total, "items": items}

    @app.get("/api/disputes/{dispute_id}")
    def detail(dispute_id: str):
        with store.session(db) as con:
            d = need(con, dispute_id)
            ctx = context(con, d)
            dec = decision_for(con, d)
            items = evidence.gather(ctx)
            t = ctx["txn"]
            return {
                "dispute": {**d, "category_label": CATEGORY_LABELS[d["category"]]},
                "order": {
                    k: t[k]
                    for k in (
                        "txn_id",
                        "ts",
                        "product_type",
                        "product",
                        "channel",
                        "amount",
                        "device_id",
                        "ip_country",
                        "ip_distance_km",
                        "three_ds",
                        "avs",
                        "cvv_match",
                        "ship_bill_match",
                        "tracking",
                        "event_ts",
                        "gate",
                    )
                },
                "state": state_view(con, dispute_id),
                "decision": dec,
                "evidence": items,
                "completeness": evidence.completeness(d["category"], items),
                "letter": store.latest_letter(con, dispute_id),
                "audit": store.events(con, dispute_id),
            }

    @app.get("/api/disputes/{dispute_id}/decision")
    def get_decision(dispute_id: str):
        with store.session(db) as con:
            return decision_for(con, need(con, dispute_id))

    def packet(con, dispute_id: str) -> dict:
        d = need(con, dispute_id)
        return evidence.build_packet(context(con, d), decision_for(con, d), store.latest_letter(con, dispute_id))

    @app.get("/api/disputes/{dispute_id}/packet")
    def get_packet(dispute_id: str):
        with store.session(db) as con:
            return packet(con, dispute_id)

    @app.get("/api/disputes/{dispute_id}/packet.html", response_class=HTMLResponse)
    def get_packet_html(dispute_id: str):
        with store.session(db) as con:
            return evidence.render_packet_html(packet(con, dispute_id))

    @app.post("/api/disputes/{dispute_id}/letter")
    def draft_letter(dispute_id: str, request: Request, body: dict = Body(default={})):  # noqa: B008
        provider_name = body.get("provider", "template")
        actor = request.headers.get("x-analyst", "anonymous")
        with store.session(db) as con:
            p = packet(con, dispute_id)
            try:
                prov = letter.make_provider(provider_name, client=letter_client)
                draft = prov.draft(p)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except letter.LetterRejected as e:
                metrics.letter(provider_name, "rejected")
                store.audit(con, dispute_id, actor, "letter_rejected", note="; ".join(e.errors))
                return JSONResponse({"error": "letter rejected by validator", "errors": e.errors}, 422)
            store.save_letter(con, dispute_id, draft)
            s = store.load_state(con, dispute_id)
            s.packet_ready = True  # a validated letter plus the packet is what gets submitted
            store.save_state(con, s)
            store.audit(
                con, dispute_id, actor, "letter_drafted", note=f"{draft['provider']}, cites {', '.join(draft['cited'])}"
            )
            metrics.letter(provider_name, "ok")
            return {"letter": draft, "packet_ready": s.packet_ready, "completeness": p["completeness"]}

    @app.post("/api/disputes/{dispute_id}/transition")
    def do_transition(dispute_id: str, request: Request, body: dict = Body(...)):  # noqa: B008
        to = body.get("to")
        actor = request.headers.get("x-analyst") or body.get("actor") or "anonymous"
        with store.session(db) as con:
            need(con, dispute_id)
            s = store.load_state(con, dispute_id)
            try:
                s2 = transition(s, to, clock)
            except TransitionError as e:
                metrics.transition(str(to), "rejected")
                return JSONResponse({"error": str(e), "status": s.status}, 409)
            store.save_state(con, s2)
            store.audit(con, dispute_id, actor, "transition", s.status, s2.status, body.get("note"))
            metrics.transition(to, "ok")
            log_event(logger, "transition", dispute_id=dispute_id, actor=actor, from_status=s.status, to_status=to)
            return state_view(con, dispute_id)

    @app.get("/api/disputes/{dispute_id}/audit")
    def get_audit(dispute_id: str):
        with store.session(db) as con:
            need(con, dispute_id)
            return {"events": store.events(con, dispute_id)}

    @app.post("/api/sweep")
    def sweep(body: dict = Body(default={})):  # noqa: B008
        """Daily job: reminders, auto-accepts, and issuer timeouts. dry_run lists actions without applying them."""
        dry = bool(body.get("dry_run", False))
        actions = []
        with store.session(db) as con:
            ids = [
                r[0]
                for r in con.execute(
                    "SELECT dispute_id FROM state WHERE status NOT IN ({})".format(",".join("?" * len(TERMINAL))),
                    sorted(TERMINAL),
                ).fetchall()
            ]
            for did in ids:
                d = store.dispute_row(con, did)
                s = store.load_state(con, did)
                a = sweep_one(s, clock, d["ev"])
                if a is None:
                    continue
                actions.append(a.__dict__)
                if dry:
                    continue
                if a.to:
                    s2 = transition(s, a.to, clock)
                    store.save_state(con, s2)
                    store.audit(con, did, "system", a.action, s.status, s2.status, a.reason)
                    metrics.transition(a.to, "sweep")
                else:
                    store.audit(con, did, "system", a.action, note=a.reason)
        summary: dict[str, int] = {}
        for a in actions:
            key = f"{a['action']}:{a['reason']}" if a["action"] != "remind" else "remind"
            summary[key] = summary.get(key, 0) + 1
        log_event(logger, "sweep", actions=len(actions), dry_run=dry)
        return {"today": clock.isoformat(), "dry_run": dry, "summary": summary, "actions": actions[:200]}

    @app.get("/api/analytics")
    def get_analytics():
        return report()["analytics"]

    @app.get("/api/policy")
    def get_policy():
        r = report()
        return {"policies": r["policies"], "by_category": r["policies_by_category"], "split": r["split"]}

    @app.get("/api/model")
    def get_model():
        r = report()
        return {"metrics": r["model"], "split": r["split"]}

    @app.get("/api/reason-codes")
    def reason_codes():
        return [{"network": n, "code": c, "description": desc, "category": cat} for n, c, desc, cat in REASON_CODES]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def main_app() -> FastAPI:  # uvicorn factory
    return create_app()
