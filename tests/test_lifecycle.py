from datetime import date

import pytest

from disputedesk.bizdays import add_business_days
from disputedesk.lifecycle import (
    STATES,
    TRANSITIONS,
    DisputeState,
    TransitionError,
    new_dispute,
    sweep_one,
    transition,
)

REC = date(2026, 10, 1)


def state(status: str, ready: bool = True, deadline: date | None = None) -> DisputeState:
    return DisputeState("DP-1", status, REC, REC, deadline or add_business_days(REC, 20), ready)


LEGAL = [(a, b) for a, nxt in TRANSITIONS.items() for b in sorted(nxt)]
ILLEGAL = [(a, b) for a in STATES for b in STATES if b not in TRANSITIONS[a]]


@pytest.mark.parametrize(("frm", "to"), LEGAL)
def test_every_legal_transition(frm, to):
    s2 = transition(state(frm), to, date(2026, 10, 5))
    assert s2.status == to


@pytest.mark.parametrize(("frm", "to"), ILLEGAL)
def test_every_illegal_transition_is_rejected(frm, to):
    with pytest.raises(TransitionError):
        transition(state(frm), to, date(2026, 10, 5))


def test_new_dispute_gets_20_business_day_deadline():
    s = new_dispute("DP-9", REC)
    assert s.status == "received" and s.deadline == add_business_days(REC, 20)


def test_review_keeps_network_clock():
    s = transition(state("received"), "under_review", date(2026, 10, 5))
    assert s.deadline == add_business_days(REC, 20)


def test_represent_needs_packet():
    with pytest.raises(TransitionError, match="packet"):
        transition(state("under_review", ready=False), "represented", date(2026, 10, 5))


def test_represent_after_deadline_rejected():
    with pytest.raises(TransitionError, match="deadline"):
        transition(state("under_review", deadline=date(2026, 10, 9)), "represented", date(2026, 10, 12))


def test_represented_starts_issuer_clock_and_pre_arb_clock():
    s = transition(state("under_review"), "represented", date(2026, 10, 5))
    assert s.deadline == add_business_days(date(2026, 10, 5), 30)
    p = transition(s, "pre_arbitration", date(2026, 11, 2))
    assert p.deadline == add_business_days(date(2026, 11, 2), 10)


def test_terminal_states_have_no_deadline():
    s = transition(state("received"), "accepted", date(2026, 10, 2))
    assert s.deadline is None


def test_unknown_state_and_backdated_action():
    with pytest.raises(TransitionError):
        transition(state("received"), "escalated", date(2026, 10, 2))
    with pytest.raises(TransitionError, match="before"):
        transition(state("received"), "accepted", date(2026, 9, 1))


def test_sweep_auto_accepts_negative_ev():
    a = sweep_one(state("received", ready=False), date(2026, 10, 5), ev=-12.0)
    assert a.action == "auto_accept" and a.reason == "negative_ev"


def test_sweep_positive_ev_with_time_left_is_left_alone():
    assert sweep_one(state("received", ready=False), date(2026, 10, 5), ev=12.0) is None


def test_sweep_accepts_when_deadline_would_be_missed():
    s = state("under_review", ready=False, deadline=date(2026, 10, 9))
    a = sweep_one(s, date(2026, 10, 9), ev=50.0)
    assert a.reason == "deadline" and a.to == "accepted"


def test_sweep_ready_packet_on_deadline_day_is_left_alone():
    s = state("under_review", ready=True, deadline=date(2026, 10, 9))
    assert sweep_one(s, date(2026, 10, 9), ev=50.0) is None


def test_sweep_reminder_at_five_days():
    s = state("received", deadline=add_business_days(date(2026, 10, 5), 5))
    a = sweep_one(s, date(2026, 10, 5), ev=40.0)
    assert a.action == "remind"


def test_sweep_issuer_timeout_wins():
    s = DisputeState("DP-1", "represented", REC, REC, date(2026, 10, 20), True)
    a = sweep_one(s, date(2026, 10, 21), ev=None)
    assert a.to == "won"


def test_sweep_pre_arbitration_deadline_passed_accepts():
    s = DisputeState("DP-1", "pre_arbitration", REC, REC, date(2026, 10, 20), True)
    assert sweep_one(s, date(2026, 10, 21), ev=None).to == "accepted"
