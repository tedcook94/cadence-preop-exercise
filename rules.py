"""Policy rules that turn submission data plus extracted CaseFacts into issues.

Each rule is a pure function that returns every issue it finds; no rule stops
early or depends on another rule's result.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

from core import IssueCategory, IssueEvidence, TriageIssue
from normalize import parse_date, within_days
from schemas import CaseFacts, DocumentFacts, LabTest

HP_WINDOW_DAYS = 30
LAB_WINDOW_DAYS = {"LOW": 30, "MODERATE": 30, "HIGH": 14}
REQUIRED_TESTS: dict[str, tuple[LabTest, ...]] = {
    "LOW": ("CBC",),
    "MODERATE": ("CBC",),
    "HIGH": ("CBC", "CMP"),
}
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
