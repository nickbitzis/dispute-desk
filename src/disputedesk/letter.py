"""Rebuttal letters behind one interface: an offline template (the default) and Claude.

Both return paragraphs, each with the evidence item ids it relies on. The validator rejects any letter that cites an
id not in the packet, cites an item marked missing, has a paragraph with no citation, or mentions an id in the text
that it does not list. A rejected letter never reaches a packet.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .codes import CATEGORY_LABELS

DEFAULT_MODEL = "claude-opus-5-5"
CITE_RE = re.compile(r"\[?\b(E\d+)\b\]?")


class LetterRejected(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class LetterProvider(Protocol):
    name: str

    def draft(self, packet: dict) -> dict: ...


def validate_letter(letter: dict, packet: dict) -> list[str]:
    items = {i["id"]: i for i in packet["items"]}
    errors: list[str] = []
    paras = letter.get("paragraphs") or []
    if not paras:
        errors.append("letter has no paragraphs")
    for n, p in enumerate(paras, start=1):
        cites = p.get("cites") or []
        text = p.get("text") or ""
        if not text.strip():
            errors.append(f"paragraph {n} is empty")
        if not cites:
            errors.append(f"paragraph {n} cites no evidence")
        for c in cites:
            if c not in items:
                errors.append(f"paragraph {n} cites {c}, which is not in the packet")
            elif not items[c]["available"]:
                errors.append(f"paragraph {n} cites {c} ({items[c]['label']}), which is missing")
        for c in CITE_RE.findall(text):
            if c not in cites:
                errors.append(f"paragraph {n} mentions {c} in the text but does not list it as a citation")
    return errors


def checked(letter: dict, packet: dict) -> dict:
    errors = validate_letter(letter, packet)
    if errors:
        raise LetterRejected(errors)
    letter["cited"] = sorted({c for p in letter["paragraphs"] for c in p["cites"]}, key=lambda s: int(s[1:]))
    return letter


class TemplateLetter:
    """Deterministic letter from the available evidence. Works offline; the demo and tests use it."""

    name = "template"

    def draft(self, packet: dict) -> dict:
        avail = [i for i in packet["items"] if i["available"]]
        if not avail:
            raise LetterRejected(["no available evidence to cite"])
        cat = CATEGORY_LABELS[packet["category"]].lower()
        first = avail[0]
        paras = [
            {
                "text": (
                    f"Brightgate Tickets responds to dispute {packet['dispute_id']} "
                    f"(reason code {packet['reason_code']}, "
                    f"{cat}) for ${packet['amount']:,.2f} and asks that it be reversed. {first['fact']}"
                ),
                "cites": [first["id"]],
            }
        ]
        req = [i for i in avail[1:] if i["role"] == "required"]
        helpful = [i for i in avail[1:] if i["role"] == "helpful"]
        for i in req:
            paras.append({"text": f"{i['label']}: {i['fact']}", "cites": [i["id"]]})
        if helpful:
            paras.append(
                {
                    "text": "Supporting facts: " + " ".join(i["fact"] for i in helpful),
                    "cites": [i["id"] for i in helpful],
                }
            )
        paras.append(
            {
                "text": "The attached records show the charge was authorized and fulfilled as agreed. "
                "We request the dispute be decided in the merchant's favor.",
                "cites": [i["id"] for i in avail],
            }
        )
        return checked({"provider": self.name, "model": None, "paragraphs": paras}, packet)


SYSTEM_PROMPT = (
    "You write chargeback rebuttal letters for an online ticket and merchandise seller. Use only the evidence "
    "items given. Never cite an item whose status is missing. Every paragraph must list the ids of the evidence "
    "items it relies on in 'cites'. Be factual and brief: four paragraphs at most, no speculation about the "
    "cardholder."
)

LETTER_SCHEMA = {
    "type": "object",
    "properties": {
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}}},
                "required": ["text", "cites"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["paragraphs"],
    "additionalProperties": False,
}


class ClaudeLetter:
    """Claude with structured output. The anthropic SDK is imported only when this provider is built."""

    name = "claude"

    def __init__(self, client=None, model: str | None = None, max_tokens: int = 4000):
        if client is None:
            import anthropic  # lazy: the app runs without the extra installed

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("DISPUTEDESK_CLAUDE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.last_usage: dict = {}

    def draft(self, packet: dict) -> dict:
        payload = {
            "dispute_id": packet["dispute_id"],
            "reason_code": packet["reason_code"],
            "category": packet["category_label"],
            "amount": packet["amount"],
            "evidence": [
                {
                    "id": i["id"],
                    "label": i["label"],
                    "status": "available" if i["available"] else "missing",
                    "fact": i["fact"],
                }
                for i in packet["items"]
            ],
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": LETTER_SCHEMA}},
            messages=[{"role": "user", "content": "Dispute and evidence:\n" + json.dumps(payload)}],
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LetterRejected(["the model declined to draft this letter"])
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LetterRejected([f"model reply was not valid JSON: {e}"]) from e
        paras = [
            {"text": str(p.get("text", "")), "cites": [str(c) for c in p.get("cites", [])]}
            for p in data.get("paragraphs", [])
        ]
        return checked({"provider": self.name, "model": self.model, "paragraphs": paras}, packet)


def make_provider(name: str | None = None, client=None) -> LetterProvider:
    name = (name or os.environ.get("DISPUTEDESK_LETTER_PROVIDER", "template")).lower()
    if name == "template":
        return TemplateLetter()
    if name == "claude":
        return ClaudeLetter(client=client)
    raise ValueError(f"unknown letter provider {name!r}")
