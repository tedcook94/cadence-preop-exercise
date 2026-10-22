from collections.abc import Sequence
from typing import Any

import pytest

from rules import RuleContext, check_missing_data
from schemas import AnticoagulantFacts, AnticoagulantStatus, CaseFacts, NoteVital

BP = {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-30T08:00:00Z"}
TEMP = {"type": "temperature", "value_f": 98.6, "date": "2026-03-30T08:00:00Z"}


def _ctx(
    *,
    procedure: dict[str, Any] | None = None,
    vitals: Sequence[dict[str, Any]] = (BP, TEMP),
    note_vitals: Sequence[NoteVital] = (),
    anticoagulants: Sequence[AnticoagulantFacts] = (),
) -> RuleContext:
    submission = {
        "procedure": procedure or {"procedure_risk": "LOW", "procedure_date": "2026-03-31"},
        "documents": [{"date": "2026-03-30"}],
        "vitals": list(vitals),
    }
    facts = CaseFacts(
        documents=[], labs=[], note_vitals=list(note_vitals), anticoagulants=list(anticoagulants)
    )
    return RuleContext(submission=submission, facts=facts)


def _drug(status: AnticoagulantStatus) -> AnticoagulantFacts:
    return AnticoagulantFacts(
        drug="warfarin",
        status=status,
        plan_doc_index=None,
        plan_has_hold=False,
        plan_has_resume=False,
        plan_matches_drug=False,
        evidence="active flag blank; patient unsure",
    )


def _sources(ctx: RuleContext) -> list[str]:
    issues = check_missing_data(ctx)
    assert all(i.category == "MISSING_REQUIRED_DATA" for i in issues)
    return [i.evidence.source for i in issues]


def test_complete_record_has_no_missing_data() -> None:
    assert _sources(_ctx()) == []


@pytest.mark.parametrize(
    ("procedure", "expected"),
    [
        ({"procedure_risk": "LOW", "procedure_date": None}, ["procedure.procedure_date"]),
        ({"procedure_risk": "LOW", "procedure_date": "TBD"}, ["procedure.procedure_date"]),
        ({"procedure_risk": None, "procedure_date": "2026-03-31"}, ["procedure.procedure_risk"]),
    ],
)
def test_missing_procedure_fields(procedure: dict[str, Any], expected: list[str]) -> None:
    assert _sources(_ctx(procedure=procedure)) == expected


def test_missing_vitals_reported_separately() -> None:
    issues = check_missing_data(_ctx(vitals=[BP]))

    assert [i.description for i in issues] == ["Missing temperature reading"]
    assert [i.description for i in check_missing_data(_ctx(vitals=[]))] == [
        "Missing blood pressure reading",
        "Missing temperature reading",
    ]


def test_note_vital_satisfies_presence() -> None:
    note_temp = NoteVital(
        doc_index=0, kind="TEMP", systolic=None, diastolic=None,
        value=37.0, unit="C", timestamp=None, evidence="temp 37.0 C",
    )

    assert _sources(_ctx(vitals=[BP], note_vitals=[note_temp])) == []


def test_uncertain_anticoagulant_is_missing_data() -> None:
    issues = check_missing_data(_ctx(anticoagulants=[_drug("UNCERTAIN"), _drug("ACTIVE")]))

    assert [i.description for i in issues] == ["Unconfirmed anticoagulant use: warfarin"]
    assert "patient unsure" in issues[0].evidence.details
