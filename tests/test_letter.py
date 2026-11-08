import json
import sys
from types import SimpleNamespace

import pytest

from disputedesk.evidence import build_packet
from disputedesk.letter import ClaudeLetter, LetterRejected, TemplateLetter, make_provider, validate_letter
from tests.conftest import make_context


def packet():
    return build_packet(make_context("fraud", history=3, three_ds="none"))


def ids(p, available=True):
    return [i["id"] for i in p["items"] if i["available"] == available]


def test_template_letter_passes_validator_and_cites_only_available():
    p = packet()
    letter = TemplateLetter().draft(p)
    assert validate_letter(letter, p) == []
    assert set(letter["cited"]) <= set(ids(p))


def test_validator_rejects_missing_evidence():
    p = packet()
    missing = ids(p, available=False)[0]
    errs = validate_letter({"paragraphs": [{"text": "x", "cites": [missing]}]}, p)
    assert any("missing" in e for e in errs)


def test_validator_rejects_unknown_id():
    errs = validate_letter({"paragraphs": [{"text": "x", "cites": ["E99"]}]}, packet())
    assert any("not in the packet" in e for e in errs)


def test_validator_rejects_uncited_paragraph_and_empty_letter():
    assert any(
        "cites no evidence" in e for e in validate_letter({"paragraphs": [{"text": "x", "cites": []}]}, packet())
    )
    assert validate_letter({"paragraphs": []}, packet()) == ["letter has no paragraphs"]


def test_validator_rejects_inline_id_not_listed():
    p = packet()
    a = ids(p)[0]
    other = [i["id"] for i in p["items"] if i["id"] != a][0]
    errs = validate_letter({"paragraphs": [{"text": f"See {other}.", "cites": [a]}]}, p)
    assert any("does not list" in e for e in errs)


class FakeMessages:
    def __init__(self, reply: str, stop_reason: str = "end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=900, output_tokens=200),
        )


def fake(reply: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(messages=FakeMessages(reply, stop_reason))


def test_claude_letter_uses_structured_output_and_validates():
    p = packet()
    good = ids(p)[:2]
    client = fake(json.dumps({"paragraphs": [{"text": "The device has history.", "cites": good}]}))
    letter = ClaudeLetter(client=client).draft(p)
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert letter["provider"] == "claude" and letter["cited"] == sorted(good, key=lambda s: int(s[1:]))


def test_claude_letter_citing_missing_evidence_is_rejected():
    p = packet()
    bad = ids(p, available=False)[0]
    with pytest.raises(LetterRejected):
        ClaudeLetter(client=fake(json.dumps({"paragraphs": [{"text": "x", "cites": [bad]}]}))).draft(p)


def test_claude_bad_json_and_refusal_are_rejected():
    with pytest.raises(LetterRejected, match="JSON"):
        ClaudeLetter(client=fake("not json")).draft(packet())
    with pytest.raises(LetterRejected, match="declined"):
        ClaudeLetter(client=fake("", stop_reason="refusal")).draft(packet())


def test_provider_factory_and_lazy_import():
    sys.modules.pop("anthropic", None)
    assert make_provider("template").name == "template"
    assert "anthropic" not in sys.modules
    with pytest.raises(ValueError):
        make_provider("gpt")
