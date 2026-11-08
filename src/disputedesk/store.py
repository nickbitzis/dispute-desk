"""SQLite store: orders, disputes, features, lifecycle state, letters, and an append-only audit trail.

WAL mode, one connection per request. Hidden truth columns (p_true, won_if_fought) stay in the table for evaluation
but the API never returns them.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .lifecycle import DisputeState

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    dispute_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage_start TEXT NOT NULL, deadline TEXT,
    packet_ready INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, dispute_id TEXT NOT NULL, ts TEXT NOT NULL, actor TEXT NOT NULL,
    action TEXT NOT NULL, from_status TEXT, to_status TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_dispute ON events(dispute_id, id);
CREATE TABLE IF NOT EXISTS letters (
    id INTEGER PRIMARY KEY AUTOINCREMENT, dispute_id TEXT NOT NULL, provider TEXT NOT NULL, model TEXT,
    created TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'audit trail is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'audit trail is append-only'); END;
"""

TS_COLS = ["ts", "customer_signup", "delivered_ts", "event_ts", "scan_ts", "refund_ts", "cs_ts", "cancel_ts"]


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextmanager
def session(path: Path):
    con = connect(path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def write_all(path: Path, tx: pd.DataFrame, disputes: pd.DataFrame, features: pd.DataFrame) -> None:
    if path.exists():
        path.unlink()
    con = connect(path)
    try:
        t = tx.copy()
        for c in TS_COLS:
            t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
        t.to_sql("transactions", con, index=False, chunksize=50_000)
        con.execute("CREATE UNIQUE INDEX ix_tx_id ON transactions(txn_id)")
        con.execute("CREATE INDEX ix_tx_device ON transactions(device_id, ts)")
        d = disputes.copy()
        d["received_date"] = d["received_date"].astype(str)
        d.to_sql("disputes", con, index=False)
        con.execute("CREATE UNIQUE INDEX ix_d_id ON disputes(dispute_id)")
        f = features.copy()
        f.insert(0, "dispute_id", disputes["dispute_id"].to_numpy())
        f.to_sql("features", con, index=False)
        con.execute("CREATE UNIQUE INDEX ix_f_id ON features(dispute_id)")
        con.executescript(SCHEMA)
        con.commit()
    finally:
        con.close()


def write_states(con: sqlite3.Connection, states: list[DisputeState], events: list[tuple]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR REPLACE INTO state VALUES (?,?,?,?,?,?)",
        [
            (
                s.dispute_id,
                s.status,
                s.stage_start.isoformat(),
                s.deadline.isoformat() if s.deadline else None,
                int(s.packet_ready),
                now,
            )
            for s in states
        ],
    )
    con.executemany(
        "INSERT INTO events (dispute_id, ts, actor, action, from_status, to_status, note) VALUES (?,?,?,?,?,?,?)",
        events,
    )


def load_state(con: sqlite3.Connection, dispute_id: str) -> DisputeState | None:
    r = con.execute(
        "SELECT s.*, d.received_date FROM state s JOIN disputes d USING (dispute_id) WHERE dispute_id = ?",
        (dispute_id,),
    ).fetchone()
    if r is None:
        return None
    return DisputeState(
        r["dispute_id"],
        r["status"],
        date.fromisoformat(r["received_date"]),
        date.fromisoformat(r["stage_start"]),
        date.fromisoformat(r["deadline"]) if r["deadline"] else None,
        bool(r["packet_ready"]),
    )


def save_state(con: sqlite3.Connection, s: DisputeState) -> None:
    con.execute(
        "UPDATE state SET status=?, stage_start=?, deadline=?, packet_ready=?, updated_at=? WHERE dispute_id=?",
        (
            s.status,
            s.stage_start.isoformat(),
            s.deadline.isoformat() if s.deadline else None,
            int(s.packet_ready),
            datetime.now().isoformat(timespec="seconds"),
            s.dispute_id,
        ),
    )


def audit(con, dispute_id: str, actor: str, action: str, from_status=None, to_status=None, note=None, ts=None) -> None:
    con.execute(
        "INSERT INTO events (dispute_id, ts, actor, action, from_status, to_status, note) VALUES (?,?,?,?,?,?,?)",
        (dispute_id, ts or datetime.now().isoformat(timespec="seconds"), actor, action, from_status, to_status, note),
    )


def events(con, dispute_id: str) -> list[dict]:
    rows = con.execute("SELECT * FROM events WHERE dispute_id=? ORDER BY id", (dispute_id,)).fetchall()
    return [dict(r) for r in rows]


PUBLIC_DISPUTE_COLS = (
    "d.dispute_id, d.txn_id, d.customer_id, d.amount, d.category, d.reason_code, d.network, d.received_date, "
    "d.split, d.p_model, d.p_win, d.ev, d.recommendation, d.liability_rule"
)


def dispute_row(con, dispute_id: str) -> dict | None:
    r = con.execute(f"SELECT {PUBLIC_DISPUTE_COLS} FROM disputes d WHERE dispute_id=?", (dispute_id,)).fetchone()
    return dict(r) if r else None


def txn_row(con, txn_id: int) -> dict:
    r = dict(con.execute("SELECT * FROM transactions WHERE txn_id=?", (txn_id,)).fetchone())
    r.pop("is_fraud", None)  # hidden truth
    for c in TS_COLS:
        if r.get(c) is None:
            r[c] = None
    return r


def feature_row(con, dispute_id: str) -> dict:
    return dict(con.execute("SELECT * FROM features WHERE dispute_id=?", (dispute_id,)).fetchone())


def device_history(con, txn: dict, received: str) -> list[dict]:
    """Earlier orders on the same device that had not been disputed by the time this dispute arrived."""
    rows = con.execute(
        """SELECT t.txn_id, t.ts, t.amount, t.product FROM transactions t
           WHERE t.device_id = ? AND t.ts < ?
             AND NOT EXISTS (SELECT 1 FROM disputes d WHERE d.txn_id = t.txn_id AND d.received_date < ?)
           ORDER BY t.ts DESC LIMIT 25""",
        (txn["device_id"], txn["ts"], received),
    ).fetchall()
    return [dict(r) for r in rows]


def latest_letter(con, dispute_id: str) -> dict | None:
    r = con.execute("SELECT body FROM letters WHERE dispute_id=? ORDER BY id DESC LIMIT 1", (dispute_id,)).fetchone()
    return json.loads(r["body"]) if r else None


def save_letter(con, dispute_id: str, letter: dict) -> None:
    con.execute(
        "INSERT INTO letters (dispute_id, provider, model, created, body) VALUES (?,?,?,?,?)",
        (
            dispute_id,
            letter["provider"],
            letter.get("model"),
            datetime.now().isoformat(timespec="seconds"),
            json.dumps(letter),
        ),
    )
