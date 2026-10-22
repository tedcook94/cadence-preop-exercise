"""Prompt text for the fact-extraction LLM call.

The instructions are static and come first so repeated calls share a cacheable
prefix; only the case JSON varies, and it goes last.
"""

import json
from typing import Any

POLICY = """
Cadence Surgical Center Pre-Operative Scheduling Policy (effective Jan 1, 2026)

Rule 1: Required documentation
- History and Physical (H&P) completed within 30 days of the procedure date.
- Signed Surgical Consent.

Rule 2: Required testing by procedure risk
- LOW or MODERATE risk: CBC within 30 days of the procedure date.
- HIGH risk: CBC within 14 days and CMP within 14 days.
- Only the most recent result for each required test is considered.

Rule 3: Anticoagulation management
- Patients currently taking an anticoagulant need a documented perioperative
  plan that clearly describes how the medication is managed before AND after
  the procedure.

Rule 4: Acute safety exclusions (most recent reading)
- Systolic BP >= 180 mmHg, diastolic BP >= 110 mmHg, or temperature > 100.4 F.
""".strip()

EXTRACTION_GUIDE = """
Your job is EXTRACTION ONLY. Do not decide readiness, compute date windows,
compare vitals to thresholds, or convert units. Report what the record says.
Downstream code applies the policy to your output.

Read the whole submission before answering. Documents can correct, supersede,
or add to each other and to the structured fields, so judge each item in the
context of all the others. Every item in submission.documents and
submission.labs carries an "index" field; copy it exactly into index and
doc_index, and use it when citing paths. Do not count positions yourself.
Every "evidence" value is one short sentence quoting or paraphrasing
the source, citing paths such as documents[2] or medications[1].

documents: one entry per item in submission.documents, in order.
- doc_kind: classify by content, not by title alone; titles may be misspelled,
  abbreviated, or mislabeled.
  - HP_PREOP: a history and physical performed as a pre-operative evaluation
    for the planned procedure.
  - HP_OTHER: any other H&P-style note (wellness, establish-care, annual, or
    retained prior H&P not done for this procedure), or a note that says the
    H&P is not yet on file.
  - CONSENT: a surgical/procedure consent form or consent discussion.
  - ANTICOAG_PLAN: a note whose content gives perioperative instructions for an
    anticoagulant (consults, medication plans, anticoagulation clinic notes).
  - NURSING: nursing intake, assessment, or pre-admission notes.
  - OTHER: anything else.
- consent_status (N/A unless doc_kind is CONSENT):
  - SIGNED: the patient (or legal representative) has signed, including
    electronic or scanned signatures.
  - PENDING: signature explicitly pending, deferred, or awaiting return.
  - UNSIGNED: printed, sent, or discussed but not signed, or signature blank.
- consent_matches_procedure (N/A unless doc_kind is CONSENT): YES if the
  consent covers submission.procedure.procedure_type (or does not name a
  procedure); NO if it names a different procedure.

labs: one entry per item in submission.labs, in order.
- test: CBC (complete blood count), CMP (comprehensive metabolic panel),
  BMP (basic metabolic panel), or OTHER. A BMP is never a CMP. Use code and
  display together; they may use synonyms or abbreviations.

note_vitals: every blood pressure or temperature reading stated inside
document text. Do not repeat readings from submission.vitals.
- One entry per reading. A recheck is a separate reading.
- BP: set systolic and diastolic; value and unit null.
- TEMP: set value and unit exactly as written (F or C); systolic and
  diastolic null.
- timestamp: ISO 8601 date-time of the reading. If only a time is stated, use
  the document date with that time. If no time is stated, use the document
  date at 00:00:00Z. Use null only if no date is known.
- Skip references that contain no number (e.g. "see flowsheet", "vitals
  deferred").

anticoagulants: one entry per distinct anticoagulant drug mentioned anywhere
(medication list or any document text), using the generic name. Recognize
brand names (e.g. Eliquis = apixaban, Xarelto = rivaroxaban, Coumadin =
warfarin, Pradaxa = dabigatran, Savaysa = edoxaban, Lovenox = enoxaparin).
Antiplatelets (aspirin, clopidogrel, prasugrel, ticagrelor) are NOT
anticoagulants; omit them. Enoxaparin used only as a bridge in another
drug's plan is not a separate entry.
- status:
  - ACTIVE: the patient is currently taking it, per the medication list or any
    document (a document may report a drug missing from the list).
  - INACTIVE: clearly stopped or discontinued and not currently taken.
  - UNCERTAIN: the record conflicts or explicitly cannot confirm current use,
    or the active flag is missing with no clarifying text.
- plan_doc_index: the document holding the most recent perioperative plan for
  this drug, else null.
- plan_has_hold: the plan gives concrete pre-operative instructions (when to
  stop or last dose). Deferring to another team or "recommendations to
  follow" is false.
- plan_has_resume: the plan gives concrete post-operative restart
  instructions. "Will be addressed after surgery" is false.
- plan_matches_drug: the plan addresses this specific drug (not a previously
  used one). False when plan_doc_index is null.
""".strip()

INSTRUCTIONS = f"{POLICY}\n\n{EXTRACTION_GUIDE}"


INDEXED_SECTIONS = ("documents", "labs")


def build_case_input(submission: dict[str, Any]) -> str:
    """Serialize the submission identically for identical input.

    Items in INDEXED_SECTIONS get an explicit "index" so the model copies a label
    instead of counting array positions, which it miscounts when items are not in
    date order.
    """

    labeled = dict(submission)
    for section in INDEXED_SECTIONS:
        labeled[section] = [
            {**item, "index": i} for i, item in enumerate(submission.get(section) or [])
        ]
    payload = json.dumps(labeled, sort_keys=True, separators=(",", ":"))
    return f"Submission JSON:\n{payload}"
