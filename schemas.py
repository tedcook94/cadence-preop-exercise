"""Structured facts extracted by the LLM from a single submission.

Every field is required (no defaults) and every model forbids extra keys so the
generated JSON schema is valid for OpenAI strict structured outputs. Optional
values are expressed as nullable types rather than omitted keys.
"""


from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocKind = Literal["HP_PREOP", "HP_OTHER", "CONSENT", "ANTICOAG_PLAN", "NURSING", "OTHER"]
ConsentStatus = Literal["SIGNED", "PENDING", "UNSIGNED", "N/A"]
ConsentMatch = Literal["YES", "NO", "N/A"]
LabTest = Literal["CBC", "CMP", "BMP", "OTHER"]
VitalKind = Literal["BP", "TEMP"]
TempUnit = Literal["F", "C"]
AnticoagulantStatus = Literal["ACTIVE", "INACTIVE", "UNCERTAIN"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentFacts(StrictModel):
    index: int = Field(description="Position in submission.documents.")
    doc_kind: DocKind
    consent_status: ConsentStatus = Field(description="N/A unless doc_kind is CONSENT.")
    consent_matches_procedure: ConsentMatch = Field(
        description="N/A unless doc_kind is CONSENT."
    )
    evidence: str = Field(description="Short quote or paraphrase supporting the values.")


class LabFacts(StrictModel):
    index: int = Field(description="Position in submission.labs.")
    test: LabTest
    evidence: str


class NoteVital(StrictModel):
    doc_index: int = Field(description="Position in submission.documents.")
    kind: VitalKind
    systolic: float | None = Field(description="Set only when kind is BP.")
    diastolic: float | None = Field(description="Set only when kind is BP.")
    value: float | None = Field(description="Set only when kind is TEMP.")
    unit: TempUnit | None = Field(description="Set only when kind is TEMP.")
    timestamp: str | None = Field(
        description="ISO 8601 time of the reading if stated, else null."
    )
    evidence: str


class AnticoagulantFacts(StrictModel):
    drug: str = Field(description="Generic name of the anticoagulant.")
    status: AnticoagulantStatus
    plan_doc_index: int | None = Field(
        description="Document holding the perioperative plan for this drug, else null."
    )
    plan_has_hold: bool
    plan_has_resume: bool
    plan_matches_drug: bool
    evidence: str


class CaseFacts(StrictModel):
    """Everything the rules need that cannot be read directly from structured fields."""

    documents: list[DocumentFacts]
    labs: list[LabFacts]
    note_vitals: list[NoteVital]
    anticoagulants: list[AnticoagulantFacts]


def case_facts_json_schema() -> dict[str, object]:
    """Return the JSON schema sent to the model for strict structured outputs."""

    return CaseFacts.model_json_schema()
