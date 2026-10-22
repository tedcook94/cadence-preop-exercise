"""Policy rules that turn submission data plus extracted CaseFacts into issues.

Each rule is a pure function that returns every issue it finds; no rule stops
early or depends on another rule's result.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from normalize import latest, parse_date, parse_datetime, to_fahrenheit, within_days
from schemas import (
    CaseFacts,
    DocumentFacts,
    IssueCategory,
    IssueEvidence,
    LabTest,
    TriageIssue,
)

HP_WINDOW_DAYS = 30
LAB_WINDOW_DAYS = {"LOW": 30, "MODERATE": 30, "HIGH": 14}
REQUIRED_TESTS: dict[str, tuple[LabTest, ...]] = {
    "LOW": ("CBC",),
    "MODERATE": ("CBC",),
    "HIGH": ("CBC", "CMP"),
}
SYSTOLIC_LIMIT = 180  # mmHg, inclusive
DIASTOLIC_LIMIT = 110  # mmHg, inclusive
TEMP_LIMIT_F = 100.4  # exclusive
# FHIR statuses for results that must not be used clinically.
INVALID_LAB_STATUSES = frozenset({"entered-in-error", "cancelled"})


@dataclass(frozen=True)
class RuleContext:
    submission: dict[str, Any]
    facts: CaseFacts

    @property
    def procedure(self) -> dict[str, Any]:
        return self.submission.get("procedure") or {}

    @property
    def procedure_date(self) -> date | None:
        return parse_date(self.procedure.get("procedure_date"))

    @property
    def procedure_risk(self) -> str | None:
        return self.procedure.get("procedure_risk")

    def documents(self) -> list[dict[str, Any]]:
        return self.submission.get("documents") or []

    def labs(self) -> list[dict[str, Any]]:
        return self.submission.get("labs") or []


def make_issue(category: IssueCategory, description: str, source: str, details: str) -> TriageIssue:
    return TriageIssue(
        category=category,
        description=description,
        evidence=IssueEvidence(source=source, details=details),
    )


def _doc_date(ctx: RuleContext, index: int) -> date | None:
    docs = ctx.documents()
    return parse_date(docs[index].get("date")) if 0 <= index < len(docs) else None


def _usable_docs(ctx: RuleContext, kind: str) -> list[tuple[DocumentFacts, date | None]]:
    """Documents of `kind` with their dates, excluding any dated after the procedure."""

    cutoff = ctx.procedure_date
    found = []
    for doc in ctx.facts.documents:
        if doc.doc_kind != kind:
            continue
        doc_date = _doc_date(ctx, doc.index)
        if cutoff and doc_date and doc_date > cutoff:
            continue
        found.append((doc, doc_date))
    return found


def check_history_and_physical(ctx: RuleContext) -> list[TriageIssue]:
    """Rule 1: a pre-op H&P must exist and be within 30 days of the procedure."""

    candidates = _usable_docs(ctx, "HP_PREOP")
    if not candidates:
        return [
            make_issue(
                "REQUIRED_DOCUMENTATION",
                "Missing pre-operative History and Physical",
                "documents",
                "No pre-operative H&P found among documents",
            )
        ]

    procedure_date = ctx.procedure_date
    if procedure_date is None:
        return []  # Window check needs a date; the missing date is reported separately.

    dated = [(doc, d) for doc, d in candidates if d is not None]
    if not dated:
        doc = candidates[0][0]
        return [
            make_issue(
                "REQUIRED_DOCUMENTATION",
                "Pre-operative H&P has no date",
                f"documents[{doc.index}]",
                f"H&P date is missing, so the 30-day window cannot be confirmed. {doc.evidence}",
            )
        ]

    doc, doc_date = max(dated, key=lambda pair: pair[1])
    if within_days(doc_date, procedure_date, HP_WINDOW_DAYS):
        return []
    age = (procedure_date - doc_date).days
    return [
        make_issue(
            "REQUIRED_DOCUMENTATION",
            "Pre-operative H&P is outdated",
            f"documents[{doc.index}]",
            f"Most recent pre-op H&P ({doc_date}) is {age} days before the procedure "
            f"({procedure_date}); limit is {HP_WINDOW_DAYS}. {doc.evidence}",
        )
    ]


def check_surgical_consent(ctx: RuleContext) -> list[TriageIssue]:
    """Rule 1: a signed consent covering the scheduled procedure must exist."""

    consents = [doc for doc, _ in _usable_docs(ctx, "CONSENT")]
    if any(d.consent_status == "SIGNED" and d.consent_matches_procedure != "NO" for d in consents):
        return []

    if not consents:
        return [
            make_issue(
                "REQUIRED_DOCUMENTATION",
                "Missing surgical consent",
                "documents",
                "No surgical consent found among documents",
            )
        ]

    latest_consent = max(consents, key=lambda d: _doc_date(ctx, d.index) or date.min)

    if latest_consent.consent_status == "SIGNED":
        description = "Signed consent does not cover the scheduled procedure"
    else:
        description = "Surgical consent not signed by patient"
    return [
        make_issue(
            "REQUIRED_DOCUMENTATION",
            description,
            f"documents[{latest_consent.index}]",
            latest_consent.evidence,
        )
    ]


def _lab_test(ctx: RuleContext, index: int, lab: dict[str, Any]) -> str:
    for fact in ctx.facts.labs:
        if fact.index == index:
            return fact.test
    code = str(lab.get("code") or "").upper().removeprefix("LAB-")
    return code if code in ("CBC", "CMP", "BMP") else "OTHER"


def check_required_testing(ctx: RuleContext) -> list[TriageIssue]:
    """Rule 2: required tests by risk, using the most recent valid result for each."""

    risk = ctx.procedure_risk
    procedure_date = ctx.procedure_date
    if risk not in REQUIRED_TESTS or procedure_date is None:
        return []  # Missing risk or date is reported separately.

    window = LAB_WINDOW_DAYS[risk]
    issues = []
    for test in REQUIRED_TESTS[risk]:
        results = []
        for index, lab in enumerate(ctx.labs()):
            if _lab_test(ctx, index, lab) != test:
                continue
            if str(lab.get("status") or "").lower() in INVALID_LAB_STATUSES:
                continue
            lab_date = parse_date(lab.get("effective_at"))
            if lab_date is None or lab_date > procedure_date:
                continue
            results.append((index, lab_date))

        if not results:
            issues.append(
                make_issue(
                    "REQUIRED_TESTING",
                    f"Missing {test}",
                    "labs",
                    f"No valid {test} result on or before the procedure ({procedure_date}); "
                    f"{risk} risk requires one within {window} days",
                )
            )
            continue

        index, lab_date = max(results, key=lambda pair: pair[1])
        if not within_days(lab_date, procedure_date, window):
            age = (procedure_date - lab_date).days
            issues.append(
                make_issue(
                    "REQUIRED_TESTING",
                    f"{test} outside required window",
                    f"labs[{index}]",
                    f"Most recent {test} ({lab_date}) is {age} days before the procedure "
                    f"({procedure_date}); {risk} risk limit is {window} days",
                )
            )
    return issues


def check_anticoagulation(ctx: RuleContext) -> list[TriageIssue]:
    """Rule 3: each active anticoagulant needs a clear hold-and-resume plan."""

    issues = []
    for drug in ctx.facts.anticoagulants:
        if drug.status != "ACTIVE":
            continue  # UNCERTAIN is reported as missing data separately.

        plan_index = drug.plan_doc_index
        plan_date = _doc_date(ctx, plan_index) if plan_index is not None else None
        cutoff = ctx.procedure_date
        if plan_index is None or (cutoff and plan_date and plan_date > cutoff):
            problem, source = "no perioperative plan documented", "medications"
        elif not drug.plan_matches_drug:
            problem, source = (
                "plan does not address this drug",
                f"documents[{plan_index}]",
            )
        else:
            missing = [
                label
                for label, present in (
                    ("hold", drug.plan_has_hold),
                    ("resume", drug.plan_has_resume),
                )
                if not present
            ]
            if not missing:
                continue
            problem = f"plan missing {' and '.join(missing)} instructions"
            source = f"documents[{plan_index}]"

        issues.append(
            make_issue(
                "ANTICOAGULATION_MANAGEMENT",
                f"Anticoagulant {drug.drug}: {problem}",
                source,
                drug.evidence,
            )
        )
    return issues


@dataclass(frozen=True)
class Reading:
    at: datetime
    source: str
    details: str
    systolic: float | None = None
    diastolic: float | None = None
    temp_f: float | None = None


def collect_readings(ctx: RuleContext) -> list[Reading]:
    """Merge structured vitals and note vitals into dated readings before the procedure."""

    readings: list[Reading] = []
    for index, vital in enumerate(ctx.submission.get("vitals") or []):
        at = parse_datetime(vital.get("date"))
        if at is None:
            continue
        source = f"vitals[{index}]"
        if vital.get("systolic") is not None or vital.get("diastolic") is not None:
            readings.append(
                Reading(
                    at,
                    source,
                    f"BP {vital.get('systolic')}/{vital.get('diastolic')} at {at:%Y-%m-%dT%H:%MZ}",
                    systolic=vital.get("systolic"),
                    diastolic=vital.get("diastolic"),
                )
            )
        elif vital.get("value_f") is not None:
            readings.append(
                Reading(
                    at,
                    source,
                    f"Temperature {vital['value_f']} F at {at:%Y-%m-%dT%H:%MZ}",
                    temp_f=float(vital["value_f"]),
                )
            )

    for note in ctx.facts.note_vitals:
        # A date-only fallback lands at midnight, so a timed same-day reading wins.
        at = parse_datetime(note.timestamp)
        if at is None:
            doc_date = _doc_date(ctx, note.doc_index)
            at = parse_datetime(doc_date.isoformat()) if doc_date else None
        if at is None:
            continue
        source = f"documents[{note.doc_index}]"
        if note.kind == "BP" and (
            note.systolic is not None or note.diastolic is not None
        ):
            readings.append(
                Reading(
                    at,
                    source,
                    note.evidence,
                    systolic=note.systolic,
                    diastolic=note.diastolic,
                )
            )
        elif note.kind == "TEMP" and note.value is not None and note.unit is not None:
            readings.append(
                Reading(
                    at,
                    source,
                    note.evidence,
                    temp_f=to_fahrenheit(note.value, note.unit),
                )
            )

    cutoff = ctx.procedure_date
    return [r for r in readings if cutoff is None or r.at.date() <= cutoff]


def check_acute_safety(ctx: RuleContext) -> list[TriageIssue]:
    """Rule 4: the most recent BP and temperature must be below exclusion limits."""

    readings = collect_readings(ctx)
    issues = []

    bp = latest(
        (r for r in readings if r.systolic is not None or r.diastolic is not None),
        key=lambda r: r.at,
    )
    if bp is not None:
        breaches = []
        if bp.systolic is not None and bp.systolic >= SYSTOLIC_LIMIT:
            breaches.append(f"systolic {bp.systolic:g} >= {SYSTOLIC_LIMIT}")
        if bp.diastolic is not None and bp.diastolic >= DIASTOLIC_LIMIT:
            breaches.append(f"diastolic {bp.diastolic:g} >= {DIASTOLIC_LIMIT}")
        if breaches:
            issues.append(
                make_issue(
                    "ACUTE_SAFETY_EXCLUSION",
                    f"Most recent blood pressure exceeds limit ({', '.join(breaches)})",
                    bp.source,
                    bp.details,
                )
            )

    temp = latest((r for r in readings if r.temp_f is not None), key=lambda r: r.at)
    if temp is not None and temp.temp_f is not None and temp.temp_f > TEMP_LIMIT_F:
        issues.append(
            make_issue(
                "ACUTE_SAFETY_EXCLUSION",
                f"Most recent temperature {temp.temp_f:.1f} F exceeds {TEMP_LIMIT_F} F",
                temp.source,
                temp.details,
            )
        )
    return issues


def check_missing_data(ctx: RuleContext) -> list[TriageIssue]:
    """Final determination: fields needed to evaluate a rule must be present and known."""

    issues = []
    if ctx.procedure_date is None:
        raw = ctx.procedure.get("procedure_date")
        issues.append(
            make_issue(
                "MISSING_REQUIRED_DATA",
                "Missing procedure date",
                "procedure.procedure_date",
                f"procedure.procedure_date is {'null' if raw is None else f'unparseable ({raw!r})'}; "
                "H&P and testing windows cannot be evaluated",
            )
        )
    if ctx.procedure_risk not in REQUIRED_TESTS:
        issues.append(
            make_issue(
                "MISSING_REQUIRED_DATA",
                "Missing procedure risk",
                "procedure.procedure_risk",
                f"procedure.procedure_risk is {ctx.procedure_risk!r}; required testing cannot be determined",
            )
        )

    readings = collect_readings(ctx)
    for label, present in (
        ("blood pressure", any(r.systolic is not None or r.diastolic is not None for r in readings)),
        ("temperature", any(r.temp_f is not None for r in readings)),
    ):
        if not present:
            issues.append(
                make_issue(
                    "MISSING_REQUIRED_DATA",
                    f"Missing {label} reading",
                    "vitals",
                    f"No dated {label} found in structured vitals or document text; "
                    "acute safety exclusions cannot be evaluated",
                )
            )

    for drug in ctx.facts.anticoagulants:
        if drug.status == "UNCERTAIN":
            issues.append(
                make_issue(
                    "MISSING_REQUIRED_DATA",
                    f"Unconfirmed anticoagulant use: {drug.drug}",
                    "medications",
                    f"Cannot determine whether a perioperative plan is required. {drug.evidence}",
                )
            )
    return issues


RULES = (
    check_missing_data,
    check_history_and_physical,
    check_surgical_consent,
    check_required_testing,
    check_anticoagulation,
    check_acute_safety,
)


def evaluate_rules(submission: dict[str, Any], facts: CaseFacts) -> list[TriageIssue]:
    """Run every rule and return all issues in a stable order."""

    ctx = RuleContext(submission=submission, facts=facts)
    return [issue for rule in RULES for issue in rule(ctx)]
