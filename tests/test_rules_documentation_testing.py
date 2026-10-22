from collections.abc import Sequence
from typing import Any

import pytest

from rules import (
    RuleContext,
    check_history_and_physical,
    check_required_testing,
    check_surgical_consent,
)
from schemas import CaseFacts, ConsentMatch, ConsentStatus, DocKind, DocumentFacts, LabFacts


def _doc(
    index: int, kind: DocKind, status: ConsentStatus = "N/A", match: ConsentMatch = "N/A"
) -> DocumentFacts:
    return DocumentFacts(
        index=index,
        doc_kind=kind,
        consent_status=status,
        consent_matches_procedure=match,
        evidence=f"documents[{index}] evidence",
    )


def _ctx(
    *,
    doc_dates: Sequence[str] = (),
    doc_facts: Sequence[DocumentFacts] = (),
    labs: Sequence[dict[str, Any]] = (),
    lab_facts: Sequence[LabFacts] = (),
    risk: str | None = "LOW",
    procedure_date: str | None = "2026-03-31",
) -> RuleContext:
    submission = {
        "procedure": {"procedure_risk": risk, "procedure_date": procedure_date},
        "documents": [{"date": d} for d in doc_dates],
        "labs": list(labs),
    }
    facts = CaseFacts(
        documents=list(doc_facts), labs=list(lab_facts), note_vitals=[], anticoagulants=[]
    )
    return RuleContext(submission=submission, facts=facts)


def _lab(code: str, effective_at: str, status: str = "final") -> dict[str, Any]:
    return {"code": code, "effective_at": effective_at, "status": status}


# --- H&P --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hp_date", "expected"),
    [("2026-03-01", []), ("2026-02-28", ["Pre-operative H&P is outdated"])],
)
def test_hp_window_is_inclusive(hp_date: str, expected: list[str]) -> None:
    issues = check_history_and_physical(
        _ctx(doc_dates=[hp_date], doc_facts=[_doc(0, "HP_PREOP")])
    )

    assert [i.description for i in issues] == expected


def test_non_preop_hp_does_not_count() -> None:
    issues = check_history_and_physical(
        _ctx(doc_dates=["2026-03-20"], doc_facts=[_doc(0, "HP_OTHER")])
    )

    assert [(i.category, i.description) for i in issues] == [
        ("REQUIRED_DOCUMENTATION", "Missing pre-operative History and Physical")
    ]


def test_most_recent_hp_is_used_and_post_procedure_ignored() -> None:
    ctx = _ctx(
        doc_dates=["2025-06-01", "2026-03-15", "2026-04-02"],
        doc_facts=[_doc(0, "HP_PREOP"), _doc(1, "HP_PREOP"), _doc(2, "HP_PREOP")],
    )

    assert check_history_and_physical(ctx) == []


def test_hp_only_after_procedure_is_missing() -> None:
    issues = check_history_and_physical(
        _ctx(doc_dates=["2026-04-02"], doc_facts=[_doc(0, "HP_PREOP")])
    )

    assert [i.description for i in issues] == ["Missing pre-operative History and Physical"]


def test_hp_window_skipped_without_procedure_date() -> None:
    ctx = _ctx(doc_dates=["2025-01-01"], doc_facts=[_doc(0, "HP_PREOP")], procedure_date=None)

    assert check_history_and_physical(ctx) == []


# --- Consent ----------------------------------------------------------------


def test_later_signed_consent_supersedes_pending() -> None:
    ctx = _ctx(
        doc_dates=["2026-03-01", "2026-03-10"],
        doc_facts=[_doc(0, "CONSENT", "PENDING", "YES"), _doc(1, "CONSENT", "SIGNED", "YES")],
    )

    assert check_surgical_consent(ctx) == []


@pytest.mark.parametrize(
    ("status", "match", "description"),
    [
        ("PENDING", "YES", "Surgical consent not signed by patient"),
        ("UNSIGNED", "YES", "Surgical consent not signed by patient"),
        ("SIGNED", "NO", "Signed consent does not cover the scheduled procedure"),
    ],
)
def test_unusable_consent(status: ConsentStatus, match: ConsentMatch, description: str) -> None:
    issues = check_surgical_consent(
        _ctx(doc_dates=["2026-03-10"], doc_facts=[_doc(0, "CONSENT", status, match)])
    )

    assert [(i.description, i.evidence.source) for i in issues] == [(description, "documents[0]")]
    assert issues[0].evidence.details == "documents[0] evidence"


def test_missing_consent() -> None:
    issues = check_surgical_consent(_ctx())

    assert [i.description for i in issues] == ["Missing surgical consent"]


# --- Testing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("risk", "cbc_date", "expected"),
    [
        ("LOW", "2026-03-01", []),
        ("MODERATE", "2026-02-28", ["CBC outside required window"]),
        ("HIGH", "2026-03-17", []),
        ("HIGH", "2026-03-16", ["CBC outside required window"]),
    ],
)
def test_cbc_window_by_risk(risk: str, cbc_date: str, expected: list[str]) -> None:
    labs = [_lab("CBC", f"{cbc_date}T08:00:00Z"), _lab("CMP", "2026-03-30T08:00:00Z")]

    issues = check_required_testing(_ctx(labs=labs, risk=risk))

    assert [i.description for i in issues] == expected


def test_high_risk_bmp_is_not_cmp() -> None:
    labs = [_lab("CBC", "2026-03-30"), _lab("BMP", "2026-03-30")]

    issues = check_required_testing(_ctx(labs=labs, risk="HIGH"))

    assert [(i.category, i.description) for i in issues] == [("REQUIRED_TESTING", "Missing CMP")]


def test_entered_in_error_falls_back_to_prior_result() -> None:
    labs = [_lab("CBC", "2026-02-01"), _lab("CBC", "2026-03-30", status="entered-in-error")]

    issues = check_required_testing(_ctx(labs=labs))

    assert [(i.description, i.evidence.source) for i in issues] == [
        ("CBC outside required window", "labs[0]")
    ]


def test_extracted_lab_type_overrides_code() -> None:
    labs = [_lab("HEME-PANEL-7", "2026-03-30")]

    assert check_required_testing(_ctx(labs=labs)) != []
    assert check_required_testing(
        _ctx(labs=labs, lab_facts=[LabFacts(index=0, test="CBC", evidence="labs[0] is a CBC")])
    ) == []


@pytest.mark.parametrize(("risk", "procedure_date"), [(None, "2026-03-31"), ("LOW", None)])
def test_testing_skipped_without_risk_or_date(risk: str | None, procedure_date: str | None) -> None:
    assert check_required_testing(_ctx(risk=risk, procedure_date=procedure_date)) == []
