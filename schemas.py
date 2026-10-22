"""Pydantic schemas for the triage pipeline.

- Submission input: lenient models for the incoming patient package.
- LLM extraction: CaseFacts, the structured facts extracted by the LLM.
- Triage output: the decision and issues returned to callers.

Every CaseFacts field is required (no defaults) and every CaseFacts model
forbids extra keys so the generated JSON schema is valid for OpenAI strict
structured outputs. Optional values are expressed as nullable types rather
than omitted keys.
"""

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

DocKind = Literal["HP_PREOP", "HP_OTHER", "CONSENT", "ANTICOAG_PLAN", "NURSING", "OTHER"]
ConsentStatus = Literal["SIGNED", "PENDING", "UNSIGNED", "N/A"]
ConsentMatch = Literal["YES", "NO", "N/A"]
LabTest = Literal["CBC", "CMP", "BMP", "OTHER"]
VitalKind = Literal["BP", "TEMP"]
TempUnit = Literal["F", "C"]
AnticoagulantStatus = Literal["ACTIVE", "INACTIVE", "UNCERTAIN"]


# --- Submission input ---------------------------------------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None


# --- LLM extraction -----------------------------------------------------------


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


# --- Triage output ------------------------------------------------------------

class IssueEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Field path the issue is anchored to, e.g. documents[4].")
    details: str = Field(description="Why the source supports the issue; may cite other paths.")


class TriageIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: IssueCategory
    description: str
    evidence: IssueEvidence

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str
