# Implementation Plan: Pre-Op Triage Service

## Goal

Improve the triage service's results on `data/patients_sample_50.jsonl` (baseline score 74.0) without overfitting. The approach: one LLM call per case pulls structured facts out of the free-form text, and Python rules turn those facts into a decision based on `docs/appendix.pdf`.

## Constraints

- Leave evaluation code unchanged (`run_evals.py`).
- Leave `run_baseline.py` unchanged (no concurrency).
- Keep the `triage_submission(submission, *, model)` signature unchanged.
- Output must follow `data/example_output.json`.
- Evaluate every rule and return every issue found.
- Decision precedence: any acute safety issue → `NOT_CLEARED`; any other issue → `NEEDS_FOLLOW_UP`; no issues → `READY`.

## Decisions

- **Low-confidence enums:** asymmetric fallback. A value that would *clear* a check (e.g. consent `SIGNED`, anticoagulant `INACTIVE`) counts as failing that check when its confidence falls below the threshold. `NOT_CLEARED` is never downgraded. There is no second LLM call. Low-confidence fields are logged so the threshold can be tuned.
- **Overfitting guard:** rules come only from the policy text and generic data patterns, never from a specific sample case. Each rule gets synthetic unit tests. No holdout set for this POC.
- **Commits:** each commit is small and leaves `make test` passing. The user reviews and commits all changes.

## Architecture

```
triage_submission(submission, model)            # signature unchanged
  ├─ normalize (Python): parse dates, index every field path
  ├─ extract (LLM, 1 call, strict schema)       → CaseFacts
  ├─ rules (Python, pure functions, all run)    → list[Issue]
  └─ decide (Python): acute → NOT_CLEARED; any → NEEDS_FOLLOW_UP; else READY
```

### LLM extraction (`CaseFacts`)

Every field is required and has an `evidence` string that cites sources by index (e.g. `documents[2]`).

- `documents[]`: `index`, `doc_kind` (`HP_PREOP | HP_OTHER | CONSENT | ANTICOAG_PLAN | NURSING | OTHER`), `consent_status` (`SIGNED | PENDING | UNSIGNED | N/A`), `consent_matches_procedure` (`YES | NO | N/A`), `evidence`
- `labs[]`: `index`, `test` (`CBC | CMP | BMP | OTHER`)
- `note_vitals[]`: `doc_index`, `kind` (`BP | TEMP`), `systolic`, `diastolic`, `value`, `unit` (`F | C`), `timestamp`, `evidence`
- `anticoagulants[]`: `drug`, `status` (`ACTIVE | INACTIVE | UNCERTAIN`), `plan_doc_index | null`, `plan_has_hold`, `plan_has_resume`, `plan_matches_drug`, `evidence`

### Python responsibilities

Python handles all date math, unit conversion, latest-reading selection, thresholds, lab status (`entered-in-error`), and the final decision.

### Generic rule conventions (from the policy and data patterns)

1. If no temperature or BP exists anywhere in the record, raise `MISSING_REQUIRED_DATA`. A missing vital is not a safety pass.
2. If the procedure date or risk is missing, raise `MISSING_REQUIRED_DATA` and skip the window checks that depend on it.
3. If it is unclear whether the patient is on an anticoagulant, raise `MISSING_REQUIRED_DATA`, not `ANTICOAGULATION_MANAGEMENT`.
4. Vitals from structured fields and from note text are merged, and the latest timestamp wins (rechecks included). Celsius is converted to Fahrenheit. Thresholds: systolic ≥ 180, diastolic ≥ 110, temperature > 100.4 °F. A note that only refers elsewhere (e.g. "see flowsheet") is not a reading.
5. Day windows are inclusive.
6. A BMP is not a CMP.
7. A consent only counts if it is signed and covers the scheduled procedure. A later signed consent replaces an earlier pending one.
8. A wellness or establish-care visit does not count as a pre-op H&P.
9. An anticoagulant plan must include both a hold and a resume instruction, and it must cover the current drug. Antiplatelets do not need a plan.

### OpenAI API changes

| Change | Source |
|---|---|
| One shared module-level client with `max_retries`/`timeout` | openai-python README, "Retries" and "Timeouts" |
| `temperature=0`; pinned model snapshot (`gpt-4.1-mini-2025-04-14`) | OpenAI Models docs, snapshots |
| Strict structured outputs via `responses.parse(text_format=CaseFacts)` | Structured Outputs guide |
| Static policy and extraction guide first, case JSON last; `prompt_cache_key` | Prompt Caching guide (needs a prefix of at least 1024 tokens) |
| `store=False` (responses are stored by default; PHI concern) | Responses API reference, `store` |
| Logprobs via `include=["message.output_text.logprobs"]`, mapped to enum fields | Responses API reference, `include` |

Logprobs are only loosely calibrated, so they are used for gating, not reported as accuracy.

## Phases and commits

Status: `[ ]` todo, `[x]` done.

### Phase 0: Plan doc
- [x] `docs: add implementation plan`: this file

### Phase 1: Schemas (about 45 min)
- [x] `Add extraction schema for case facts`: `schemas.py` with `CaseFacts` and its nested models, enums, `extra="forbid"`, all fields required. Test: strict-mode schema check.
- [x] `Add structured issue evidence to output schema`: `Issue{category, description, evidence{source, details}}`. Test: `example_output.json` validates.

### Phase 2: Extraction (about 90 min)
- [x] `Add extraction prompt with static policy prefix`: `prompts.py` with the policy, the extraction guide, and the case-JSON builder. Test: the builder's output is deterministic.
- [x] `Add LLM extraction client`: `extract.py` with the shared client, `responses.parse`, `temperature=0`, pinned snapshot, `store=False`, and `prompt_cache_key`. Test: mocked client, request kwargs asserted.
- [x] `Map logprobs to enum field confidences`: align enum tokens to field paths. Test: synthetic token stream.

### Phase 3: Rules (about 2 h, each commit has synthetic unit tests)
- [x] `Add normalization helpers`: date parsing, °C→°F, inclusive window check, latest-reading selection.
- [x] `Add documentation and testing rules`: Rule 1 (H&P, consent) and Rule 2 (CBC/CMP by risk).
- [x] `Add anticoagulation and acute safety rules`: Rule 3 and Rule 4.
- [x] `Add missing-data rules and low-confidence fallback`.

### Phase 4: Wiring (about 30 min)
- [x] `Wire extraction and rules into triage_submission`: pipeline, decision precedence, and explanation builder. Rewrite `tests/test_triage_submission.py` for the mocked flow; this is moved here so `make test` stays green.

### Phase 5: End-to-end tests (about 45 min)
- [x] `Add end-to-end tests with recorded facts`: hand-written `CaseFacts` fixtures run through rules → decision, covering each decision type. No network.

### Phase 6: Evaluate and fix (1–2 h)
- [x] `Update baseline outputs and eval reports`: `make baseline evals determinism`.
- [x] `fix: <generic cause>` (0–N): one commit per generic failure cause, each with a test.

Target: score above 74.0 and 100% decision stability.

### Phase 7: Threshold (about 30 min)
- [ ] `Tune confidence threshold`: pick the threshold from the logged confidence distribution and record the rationale here.

## Threshold rationale

_To be filled in during Phase 7._
