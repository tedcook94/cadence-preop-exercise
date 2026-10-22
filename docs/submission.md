# Submission for Pre-Op Triage Take-Home

## Summary

`triage_submission` now takes a hybrid approach: one LLM call per case extracts structured facts from the records, and then deterministic Python rules apply the policy to those facts.

| Metric | Baseline | Hybrid Approach |
|---|---|---|
| Primary score (`make evals`) | 74% | 100% |
| Decision stability (`make determinism`) | 90% | 100% |
| Output match (`make determinism`) | 10% | 100% |

## Building and running

None of the existing `make` commands were changed, so the instructions in the README still apply.

## Approach and architecture

### Why a hybrid?

Manually reviewing the first 10 sample cases showed me two things:

1. Most of the relevant facts live in free-form text fields with inconsistent formatting: misspelled document titles, temperatures in a mixture of Celsius and Fahrenheit, important medication and vitals only in document notes, etc.
2. The policy depends on parsing, unit conversion, thresholds and ordering that LLMs are unreliable at.

Each part does what it's best at: the LLM reads and categorizes free text and Python does the comparisons to make the decision.

### Pipeline

```
triage_submission
  ├─ extract_case_facts        → One call to LLM generates `CaseFacts` containing document types,
  │                              consent status, vitals and anticoagulant plans
  ├─ apply_confidence_fallback → Low logprob values that would otherwise clear a check are treated as failing
  ├─ evaluate_rules            → Python checks against every rule function matching the policy and returns
  │                              all issues found with evidence
  └─ decide                    → Checks issues to determine whether any are acute enough for `NOT_CLEARED`;
                                 otherwise returns `NEEDS_FOLLOW_UP` unless there are no issues
```

## Design decisions

1. The LLM only extracts.
    - Date parsing, unit conversion, latest-reading selection, thresholds, lab status and the final decision are all in Python. This makes the decision reproducible from the extracted facts and lets each rule be unit tested directly.
1. Strict extraction schema.
    - Every field is required, uses enums where possible and the schema is sent with strict Structured Outputs. This gives the rules a stable set of values to work with.
1. Every rule runs and every issue is reported.
    - Rules never stop early on the first failure, reporting every issue that contributes to `NEEDS_FOLLOW_UP` or `NOT_CLEARED`.
1. Low-confidence fallbacks.
    - Logprobs are mapped to enum and boolean fields. If a field that would clear a check (consent is `SIGNED`, anticoagulant is `INACTIVE`, etc.) falls below 0.8 confidence, it falls back to the failing value. This prevents a weak signal from clearing a potentially risky procedure.
    - A second targeted LLM call was considered, but the added cost and latency against a signal that was already unclear seemed less preferable.
1. Improved OpenAI API usage.
    - Reuse one shared client rather than instantiating a new client for each call.
    - Set `temperature` to `0` and pinned the model snapshot to reduce variation between runs.
    - Static policy and extraction guide provided first and case JSON last to maximize prefix caching between calls.
    - Set `store` to `False` to prevent PHI from being stored by default.

## Assumptions

1. Date windows
    - Windows include the boundary day (exactly 30 days ago passes) and documents/measurements dated after the procedure aren't considered.
    - Any date without an explicit time is assumed to be at midnight.
1. Vitals
    - Readings in note text count along with structured vital readings.
    - Latest reading always takes precedence.
    - Blood pressure thresholds are inclusive but temperature threshold is not (based on observations from sample cases).
1. Testing
    - A wellness visit does not count for pre-op H&P.
    - A BMP is not equivalent to a CMP.
    - Labs marked `entered-in-error` are ignored.

## Agent session

- `docs/opencode_session_complete.md`: Entire session history, including agent thinking and tool calls.
- `docs/opencode_session_conversation.md`: Only the conversational portions of the session between myself and the agent.
