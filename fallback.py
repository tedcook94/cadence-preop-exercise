"""Asymmetric low-confidence fallback for extracted facts.

A low-confidence value that would *clear* a check is replaced with the value
that fails it, so uncertainty can add issues but never remove one. Values that
raise issues (e.g. an ACTIVE anticoagulant or a high note vital) are kept.
"""

import logging

from schemas import CaseFacts

logger = logging.getLogger(__name__)

# Not tuned: kept as a conservative POC default. See docs/implementation_plan.md,
# "Threshold rationale", before changing it.
CONFIDENCE_THRESHOLD = 0.8

# (section, field) -> {clearing value: failing replacement}
DEMOTIONS: dict[tuple[str, str], dict[object, object]] = {
    ("documents", "doc_kind"): {"HP_PREOP": "HP_OTHER", "CONSENT": "OTHER", "ANTICOAG_PLAN": "OTHER"},
    ("documents", "consent_status"): {"SIGNED": "UNSIGNED"},
    ("documents", "consent_matches_procedure"): {"YES": "NO"},
    ("labs", "test"): {"CBC": "OTHER", "CMP": "OTHER"},
    ("anticoagulants", "status"): {"INACTIVE": "UNCERTAIN"},
    ("anticoagulants", "plan_has_hold"): {True: False},
    ("anticoagulants", "plan_has_resume"): {True: False},
    ("anticoagulants", "plan_matches_drug"): {True: False},
}


def apply_confidence_fallback(
    facts: CaseFacts,
    confidences: dict[str, float],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[CaseFacts, list[str]]:
    """Return facts with low-confidence clearing values demoted, plus the demoted paths."""

    if not confidences:
        logger.warning("No logprob confidences available; extracted facts used as-is")
        return facts, []

    demoted: list[str] = []
    sections: dict[str, list] = {}
    for section in ("documents", "labs", "note_vitals", "anticoagulants"):
        items = []
        for index, item in enumerate(getattr(facts, section)):
            updates = {}
            for (target, field), mapping in DEMOTIONS.items():
                if target != section:
                    continue
                path = f"{section}[{index}].{field}"
                value = getattr(item, field)
                if value in mapping and confidences.get(path, 1.0) < threshold:
                    updates[field] = mapping[value]
                    demoted.append(path)
                    logger.info("Low confidence %.3f on %s=%r; treated as %r",
                                confidences[path], path, value, mapping[value])
            items.append(item.model_copy(update=updates) if updates else item)
        sections[section] = items
    return CaseFacts(**sections), demoted
