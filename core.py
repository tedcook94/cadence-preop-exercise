"""The triage_submission entry point: LLM fact extraction, then policy rules."""

import logging

from schemas import (
    Decision,
    IssueCategory,
    IssueEvidence,
    PatientSubmission,
    ProcedureRisk,
    TriageIssue,
    TriageOutput,
)

__all__ = [
    "Decision",
    "IssueCategory",
    "IssueEvidence",
    "PatientSubmission",
    "ProcedureRisk",
    "TriageIssue",
    "TriageOutput",
    "build_explanation",
    "decide",
    "triage_submission",
]

logger = logging.getLogger(__name__)

ACUTE_CATEGORIES: frozenset[IssueCategory] = frozenset({"ACUTE_SAFETY_EXCLUSION"})
READY_EXPLANATION = "All required documentation, testing, and clinical criteria are satisfied."


def decide(issues: list[TriageIssue]) -> Decision:
    """Acute safety issues take precedence over all others; no issues means READY."""

    if any(issue.category in ACUTE_CATEGORIES for issue in issues):
        return "NOT_CLEARED"
    return "NEEDS_FOLLOW_UP" if issues else "READY"


def build_explanation(issues: list[TriageIssue]) -> str:
    if not issues:
        return READY_EXPLANATION
    return " | ".join(f"{issue.category}: {issue.description}" for issue in issues)


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Triage one submission: LLM fact extraction, then deterministic policy rules."""

    # Imported here so the schema re-exports above load without the OpenAI SDK.
    from extract import extract_case_facts
    from fallback import apply_confidence_fallback
    from rules import evaluate_rules

    if not isinstance(submission, PatientSubmission):
        submission = PatientSubmission.model_validate(submission)
    payload = submission.model_dump()

    extraction = extract_case_facts(payload, model=model)
    facts, demoted = apply_confidence_fallback(extraction.facts, extraction.confidences)
    if demoted:
        case_id = (payload.get("procedure") or {}).get("case_id")
        logger.info("Case %s: low-confidence fields treated as failing: %s", case_id, demoted)

    issues = evaluate_rules(payload, facts)
    return TriageOutput(
        decision=decide(issues),
        issues=issues,
        explanation=build_explanation(issues),
    )
