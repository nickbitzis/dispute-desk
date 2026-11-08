"""Liability shift rules, kept explicit and out of the model.

A model learns that 3-D Secure authenticated fraud disputes tend to win, but a rule from the network should not be
left to a model to rediscover each retrain. These rules run before the model and set a floor on the win
probability, and they force a fight even when the model alone would accept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LiabilityResult:
    applies: bool
    rule_id: str | None
    p_floor: float
    explanation: str

    def as_dict(self) -> dict:
        return asdict(self)


def liability_shift(category: str, three_ds: str, refund_before_dispute: bool = False) -> LiabilityResult:
    if refund_before_dispute:
        return LiabilityResult(
            True,
            "CR-REFUNDED",
            0.90,
            "The order was refunded before the dispute was raised. A credit already processed defense applies to "
            "every reason category.",
        )
    if category == "fraud":
        if three_ds == "authenticated":
            return LiabilityResult(
                True,
                "LS-3DS-AUTH",
                0.95,
                "Fraud dispute on a 3-D Secure authenticated transaction. Liability shifts to the issuer.",
            )
        if three_ds == "attempted":
            return LiabilityResult(
                True,
                "LS-3DS-ATTEMPT",
                0.85,
                "3-D Secure was attempted and the issuer did not take part. Liability shifts to the issuer.",
            )
        return LiabilityResult(False, None, 0.0, "No 3-D Secure authentication, so the merchant carries fraud risk.")
    if three_ds == "authenticated":
        return LiabilityResult(
            False,
            None,
            0.0,
            "3-D Secure shifts liability for fraud only. It does not cover this reason category.",
        )
    return LiabilityResult(False, None, 0.0, "No liability shift rule applies.")
