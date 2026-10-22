import logging

import pytest

from fallback import apply_confidence_fallback
from schemas import AnticoagulantFacts, CaseFacts, DocumentFacts, LabFacts

FACTS = CaseFacts(
    documents=[
        DocumentFacts(index=0, doc_kind="CONSENT", consent_status="SIGNED",
                      consent_matches_procedure="YES", evidence="signed"),
    ],
    labs=[LabFacts(index=0, test="CBC", evidence="CBC")],
    note_vitals=[],
    anticoagulants=[
        AnticoagulantFacts(drug="warfarin", status="INACTIVE", plan_doc_index=None,
                           plan_has_hold=True, plan_has_resume=True,
                           plan_matches_drug=True, evidence="stopped"),
    ],
)


def test_low_confidence_clearing_values_are_demoted() -> None:
    confidences = {
        "documents[0].consent_status": 0.4,
        "labs[0].test": 0.5,
        "anticoagulants[0].status": 0.3,
        "anticoagulants[0].plan_has_resume": 0.6,
    }

    facts, demoted = apply_confidence_fallback(FACTS, confidences, threshold=0.8)

    assert facts.documents[0].consent_status == "UNSIGNED"
    assert facts.documents[0].doc_kind == "CONSENT"
    assert facts.labs[0].test == "OTHER"
    assert facts.anticoagulants[0].status == "UNCERTAIN"
    assert facts.anticoagulants[0].plan_has_resume is False
    assert facts.anticoagulants[0].plan_has_hold is True
    assert demoted == list(confidences)


def test_high_confidence_and_failing_values_are_kept() -> None:
    active = FACTS.model_copy(update={
        "anticoagulants": [FACTS.anticoagulants[0].model_copy(update={"status": "ACTIVE"})]
    })

    facts, demoted = apply_confidence_fallback(
        active, {"documents[0].consent_status": 0.95, "anticoagulants[0].status": 0.1}, threshold=0.8
    )

    assert facts == active
    assert demoted == []


def test_no_confidences_leaves_facts_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        facts, demoted = apply_confidence_fallback(FACTS, {})

    assert (facts, demoted) == (FACTS, [])
    assert "No logprob confidences" in caplog.text
