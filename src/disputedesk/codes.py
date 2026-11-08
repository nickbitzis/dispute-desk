"""Reason categories, the fictional network-agnostic reason codes, and the money constants.

Card networks each publish their own reason codes. Back offices map them into a small set of categories so one
evidence template and one model can serve every network. The codes below are made up (network "A" and "B"), but the
shape of the mapping table is the real thing.
"""

from __future__ import annotations

CATEGORIES = [
    "fraud",
    "not_received",
    "not_as_described",
    "duplicate",
    "credit_not_processed",
    "canceled_recurring",
]

CATEGORY_LABELS = {
    "fraud": "Fraud, card not present",
    "not_received": "Item or ticket not received",
    "not_as_described": "Not as described",
    "duplicate": "Duplicate processing",
    "credit_not_processed": "Credit not processed",
    "canceled_recurring": "Canceled recurring",
}

# network, code, description, category
REASON_CODES: list[tuple[str, str, str, str]] = [
    ("A", "A-10.4", "Card absent fraud", "fraud"),
    ("A", "A-10.5", "Account takeover", "fraud"),
    ("A", "A-13.1", "Merchandise or services not received", "not_received"),
    ("A", "A-13.3", "Not as described or defective", "not_as_described"),
    ("A", "A-12.6", "Duplicate processing", "duplicate"),
    ("A", "A-13.6", "Credit not processed", "credit_not_processed"),
    ("A", "A-13.2", "Canceled recurring transaction", "canceled_recurring"),
    ("B", "B-4837", "No cardholder authorization", "fraud"),
    ("B", "B-4855", "Goods or services not provided", "not_received"),
    ("B", "B-4853", "Cardholder dispute, not as described", "not_as_described"),
    ("B", "B-4834", "Point of interaction error, duplicate", "duplicate"),
    ("B", "B-4860", "Credit not processed", "credit_not_processed"),
    ("B", "B-4841", "Canceled recurring transaction", "canceled_recurring"),
]

CODE_TO_CATEGORY = {code: cat for _, code, _, cat in REASON_CODES}


def codes_for(category: str, network: str) -> list[str]:
    return [code for net, code, _, cat in REASON_CODES if cat == category and net == network]


# Business days the merchant has to act at each stage, network style.
RESPONSE_DAYS = {
    "received": 20,  # respond (accept or represent) within 20 business days of the dispute date
    "represented": 30,  # issuer has 30 business days to accept the evidence or escalate
    "pre_arbitration": 10,  # merchant has 10 business days to answer a pre-arbitration case
}
REMINDER_DAYS = (5, 2)  # remind when this many business days are left
MIN_PREP_DAYS = 1  # an evidence packet needs at least one business day to assemble and submit

# Money. Every dispute already debited the amount; the question is whether fighting earns it back.
HANDLING_COST = 25.0  # analyst time and tooling to assemble and submit a representment
LOSS_FEE = 15.0  # extra fee the merchant pays when a represented dispute is lost (network and acquirer)
DISPUTE_FEE = 15.0  # charged on every dispute, fought or not, so it never changes the decision

# Network monitoring program (fictional), count based: disputes in a month / sales in the month before.
NETWORK_THRESHOLD = 0.0125
EARLY_WARNING = 0.0110
