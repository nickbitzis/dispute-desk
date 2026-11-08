"""Dispute lifecycle as an explicit state machine with network deadlines.

    received ──▶ under_review ──▶ represented ──▶ won
       │              │               │  └──────▶ lost
       │              │               ▼
       └──────────────┴──▶ accepted   pre_arbitration ──▶ won | lost | accepted

Each open stage carries a deadline in business days. Pure functions here; the store persists the result and the
audit trail. Illegal moves raise TransitionError, so the API can return 409 with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .bizdays import add_business_days, business_days_between
from .codes import MIN_PREP_DAYS, REMINDER_DAYS, RESPONSE_DAYS

STATES = ["received", "under_review", "accepted", "represented", "pre_arbitration", "won", "lost"]
TERMINAL = {"accepted", "won", "lost"}
OPEN = {"received", "under_review"}  # the merchant owes a response
TRANSITIONS: dict[str, set[str]] = {
    "received": {"under_review", "accepted"},
    "under_review": {"accepted", "represented"},
    "represented": {"won", "lost", "pre_arbitration"},
    "pre_arbitration": {"accepted", "won", "lost"},
    "accepted": set(),
    "won": set(),
    "lost": set(),
}


class TransitionError(Exception):
    pass


@dataclass
class DisputeState:
    dispute_id: str
    status: str
    received_date: date
    stage_start: date
    deadline: date | None
    packet_ready: bool = False

    def as_dict(self) -> dict:
        return {
            "dispute_id": self.dispute_id,
            "status": self.status,
            "received_date": self.received_date.isoformat(),
            "stage_start": self.stage_start.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "packet_ready": self.packet_ready,
        }


def response_deadline(received: date) -> date:
    return add_business_days(received, RESPONSE_DAYS["received"])


def new_dispute(dispute_id: str, received: date) -> DisputeState:
    return DisputeState(dispute_id, "received", received, received, response_deadline(received))


def transition(s: DisputeState, to: str, today: date) -> DisputeState:
    """Return the new state. Guards: legal edge, merchant deadlines, and a ready packet to represent."""
    if to not in STATES:
        raise TransitionError(f"unknown state {to!r}")
    if to not in TRANSITIONS[s.status]:
        raise TransitionError(f"cannot move from {s.status} to {to}")
    if today < s.stage_start:
        raise TransitionError("action dated before the current stage started")
    if to == "represented":
        if not s.packet_ready:
            raise TransitionError("evidence packet is not ready")
        if s.deadline and today > s.deadline:
            raise TransitionError(f"response deadline {s.deadline} has passed")
    if s.status == "pre_arbitration" and to == "won" and s.deadline and today > s.deadline:
        raise TransitionError("pre-arbitration response deadline has passed")
    if to == "under_review":
        deadline = s.deadline  # review does not reset the network clock
    elif to == "represented":
        deadline = add_business_days(today, RESPONSE_DAYS["represented"])  # the issuer's clock
    elif to == "pre_arbitration":
        deadline = add_business_days(today, RESPONSE_DAYS["pre_arbitration"])
    else:
        deadline = None
    return DisputeState(s.dispute_id, to, s.received_date, today, deadline, s.packet_ready)


def days_left(s: DisputeState, today: date) -> int | None:
    if s.deadline is None:
        return None
    return business_days_between(today, s.deadline)


@dataclass
class SweepAction:
    dispute_id: str
    action: str  # remind | auto_accept | issuer_timeout
    reason: str
    to: str | None = None


def sweep_one(s: DisputeState, today: date, ev: float | None) -> SweepAction | None:
    """Daily job for one dispute: reminders, and auto-accept when fighting is not worth it or cannot be done in time.

    - The merchant must answer before the deadline. With less than MIN_PREP_DAYS business days left and no
      packet ready, the dispute is accepted now rather than lost by default later.
    - When the expected value of fighting is negative, accept. EV already includes any liability shift floor.
    - When the issuer lets its own clock run out on a represented dispute, the merchant wins.
    - When the merchant lets a pre-arbitration deadline pass, the case is accepted.
    """
    left = days_left(s, today)
    if s.status in OPEN:
        if left is not None and (left < 0 or (left < MIN_PREP_DAYS and not s.packet_ready)):
            return SweepAction(s.dispute_id, "auto_accept", "deadline", "accepted")
        if ev is not None and ev < 0:
            return SweepAction(s.dispute_id, "auto_accept", "negative_ev", "accepted")
        if left in REMINDER_DAYS:
            return SweepAction(s.dispute_id, "remind", f"{left} business days left")
        return None
    if s.status == "represented" and left is not None and left < 0:
        return SweepAction(s.dispute_id, "issuer_timeout", "issuer did not respond in time", "won")
    if s.status == "pre_arbitration" and left is not None and left < 0:
        return SweepAction(s.dispute_id, "auto_accept", "pre_arbitration deadline passed", "accepted")
    return None
