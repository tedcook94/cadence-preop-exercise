from collections.abc import Sequence
from typing import Any

import pytest

from rules import RuleContext, check_acute_safety, check_anticoagulation, check_missing_data
from schemas import AnticoagulantFacts, AnticoagulantStatus, CaseFacts, NoteVital, TempUnit


def _ctx(
    *,
    doc_dates: Sequence[str] = (),
    vitals: Sequence[dict[str, Any]] = (),
    note_vitals: Sequence[NoteVital] = (),
    anticoagulants: Sequence[AnticoagulantFacts] = (),
    procedure_date: str | None = "2026-03-31",
) -> RuleContext:
    submission = {
        "procedure": {"procedure_risk": "LOW", "procedure_date": procedure_date},
        "documents": [{"date": d} for d in doc_dates],
        "vitals": list(vitals),
    }
    facts = CaseFacts(
        documents=[],
        labs=[],
        note_vitals=list(note_vitals),
        anticoagulants=list(anticoagulants),
    )
    return RuleContext(submission=submission, facts=facts)


def _drug(
    *,
    status: AnticoagulantStatus = "ACTIVE",
    plan: int | None = 0,
    hold: bool = True,
    resume: bool = True,
    matches: bool = True,
) -> AnticoagulantFacts:
    return AnticoagulantFacts(
        drug="apixaban",
        status=status,
        plan_doc_index=plan,
        plan_has_hold=hold,
        plan_has_resume=resume,
        plan_matches_drug=matches,
        evidence="drug evidence",
    )


def _bp(systolic: float, diastolic: float, at: str) -> dict[str, Any]:
    return {"type": "blood_pressure", "systolic": systolic, "diastolic": diastolic, "date": at}


def _temp(value_f: float, at: str) -> dict[str, Any]:
    return {"type": "temperature", "value_f": value_f, "date": at}


def _note_bp(systolic: float, diastolic: float, timestamp: str | None, doc: int = 0) -> NoteVital:
    return NoteVital(
        doc_index=doc,
        kind="BP",
        systolic=systolic,
        diastolic=diastolic,
        value=None,
        unit=None,
        timestamp=timestamp,
        evidence="note BP",
    )


def _note_temp(value: float, unit: TempUnit, timestamp: str | None, doc: int = 0) -> NoteVital:
    return NoteVital(
        doc_index=doc,
        kind="TEMP",
        systolic=None,
        diastolic=None,
        value=value,
        unit=unit,
        timestamp=timestamp,
        evidence="note temp",
    )


# --- Anticoagulation ----------------------------------------------------------


def test_complete_plan_passes() -> None:
    assert check_anticoagulation(_ctx(doc_dates=["2026-03-20"], anticoagulants=[_drug()])) == []


def test_inactive_and_uncertain_drugs_are_skipped() -> None:
    drugs = [_drug(status="INACTIVE", plan=None), _drug(status="UNCERTAIN", plan=None)]

    assert check_anticoagulation(_ctx(anticoagulants=drugs)) == []


@pytest.mark.parametrize(
    ("drug", "doc_dates", "description", "source"),
    [
        (_drug(plan=None), [], "no perioperative plan documented", "medications"),
        (_drug(), ["2026-04-02"], "no perioperative plan documented", "medications"),
        (_drug(matches=False), ["2026-03-20"], "plan does not address this drug", "documents[0]"),
        (_drug(resume=False), ["2026-03-20"], "plan missing resume instructions", "documents[0]"),
        (_drug(hold=False), ["2026-03-20"], "plan missing hold instructions", "documents[0]"),
        (
            _drug(hold=False, resume=False),
            ["2026-03-20"],
            "plan missing hold and resume instructions",
            "documents[0]",
        ),
    ],
)
def test_incomplete_plan(
    drug: AnticoagulantFacts, doc_dates: list[str], description: str, source: str
) -> None:
    issues = check_anticoagulation(_ctx(doc_dates=doc_dates, anticoagulants=[drug]))

    assert [(i.category, i.description, i.evidence.source) for i in issues] == [
        ("ANTICOAGULATION_MANAGEMENT", f"Anticoagulant apixaban: {description}", source)
    ]
    assert issues[0].evidence.details == "drug evidence"


# --- Acute safety -------------------------------------------------------------


@pytest.mark.parametrize(
    ("systolic", "diastolic", "flagged"),
    [(179, 109, False), (180, 80, True), (120, 110, True)],
)
def test_bp_thresholds_are_inclusive(systolic: float, diastolic: float, flagged: bool) -> None:
    issues = check_acute_safety(_ctx(vitals=[_bp(systolic, diastolic, "2026-03-30T08:00:00Z")]))

    assert bool(issues) is flagged


@pytest.mark.parametrize(
    ("value", "unit", "flagged"),
    [(100.4, "F", False), (100.5, "F", True), (38.0, "C", False), (38.1, "C", True)],
)
def test_temperature_threshold_is_exclusive(value: float, unit: TempUnit, flagged: bool) -> None:
    ctx = _ctx(doc_dates=["2026-03-30"], note_vitals=[_note_temp(value, unit, "2026-03-30T08:00:00Z")])

    assert bool(check_acute_safety(ctx)) is flagged


def test_later_normal_recheck_overrides_high_reading() -> None:
    ctx = _ctx(
        doc_dates=["2026-03-30"],
        note_vitals=[
            _note_bp(190, 112, "2026-03-30T08:06:00Z"),
            _note_bp(137, 78, "2026-03-30T08:30:00Z"),
        ],
    )

    assert check_acute_safety(ctx) == []


def test_later_note_reading_overrides_structured_vitals() -> None:
    ctx = _ctx(
        doc_dates=["2026-03-30"],
        vitals=[_bp(119, 78, "2026-03-28T09:00:00Z")],
        note_vitals=[_note_bp(157, 117, "2026-03-30T07:00:00Z")],
    )

    issues = check_acute_safety(ctx)

    assert [(i.category, i.evidence.source, i.evidence.details) for i in issues] == [
        ("ACUTE_SAFETY_EXCLUSION", "documents[0]", "note BP")
    ]


def test_date_only_note_reading_uses_document_date_at_midnight() -> None:
    high_note = _note_bp(185, 90, None)

    earlier_day = _ctx(doc_dates=["2026-03-30"], vitals=[_bp(120, 80, "2026-03-29T09:00:00Z")], note_vitals=[high_note])
    same_day = _ctx(doc_dates=["2026-03-30"], vitals=[_bp(120, 80, "2026-03-30T09:00:00Z")], note_vitals=[high_note])

    assert len(check_acute_safety(earlier_day)) == 1
    assert check_acute_safety(same_day) == []


def test_bp_and_temperature_report_separately() -> None:
    vitals = [_bp(185, 115, "2026-03-30T08:00:00Z"), _temp(101.8, "2026-03-30T08:00:00Z")]

    issues = check_acute_safety(_ctx(vitals=vitals))

    assert [i.evidence.source for i in issues] == ["vitals[0]", "vitals[1]"]
    assert "systolic 185 >= 180" in issues[0].description
    assert "diastolic 115 >= 110" in issues[0].description


def test_readings_after_procedure_are_ignored() -> None:
    issues = check_acute_safety(_ctx(vitals=[_bp(200, 120, "2026-04-01T08:00:00Z")]))

    assert issues == []


# --- Antiplatelets ------------------------------------------------------------


@pytest.mark.parametrize("name", ["aspirin", "Aspirin 81 mg PO daily", "ASA", "Plavix", "clopidogrel"])
def test_antiplatelets_never_raise_anticoagulation_issues(name: str) -> None:
    drug = _drug(plan=None).model_copy(update={"drug": name})

    assert check_anticoagulation(_ctx(anticoagulants=[drug])) == []


def test_antiplatelet_uncertain_status_is_not_missing_data() -> None:
    drug = _drug(status="UNCERTAIN", plan=None).model_copy(update={"drug": "aspirin"})
    ctx = _ctx(
        anticoagulants=[drug],
        vitals=[_bp(120, 80, "2026-03-30T08:00:00Z"), _temp(98.6, "2026-03-30T08:00:00Z")],
    )

    assert check_missing_data(ctx) == []


@pytest.mark.parametrize("name", ["apixaban", "warfarin", "basaprin"])
def test_non_antiplatelets_are_still_checked(name: str) -> None:
    drug = _drug(plan=None).model_copy(update={"drug": name})

    assert len(check_anticoagulation(_ctx(anticoagulants=[drug]))) == 1
