# Pre-Op Triage Take-Home

## Current Status

`triage_submission` (in `core.py`) is reimplemented as a hybrid pipeline. One LLM call per case extracts structured facts, and deterministic Python rules apply the policy from `docs/appendix.pdf`.

| Metric | Baseline | Current |
|---|---|---|
| Primary score (`make evals`) | 74.0% | 100% (two consecutive runs) |
| Decision stability (`make determinism`, `case_00000`, 10 runs) | 90% (9 NEEDS_FOLLOW_UP, 1 READY) | 100% (10 READY) |
| Exact output match (same run) | 10% | 100% |

The `case_00000` label is READY, so the baseline was both less stable and mostly wrong on that case. Baseline numbers come from commit `91fbf1d`, the last commit before this work.

Design decisions and the commit history are tracked in `docs/implementation_plan.md`.

### How it works

```
triage_submission(submission, model)
  ├─ extract (LLM, 1 call)   → CaseFacts: document kinds, consent status, lab types,
  │                            vitals in note text, anticoagulants and their plans
  ├─ confidence fallback     → low-logprob values that would clear a check are treated as failing
  ├─ rules (Python, all run) → every issue found, with evidence citing source paths
  └─ decide                  → any ACUTE_SAFETY_EXCLUSION → NOT_CLEARED;
                               any other issue → NEEDS_FOLLOW_UP; none → READY
```

- **The LLM extracts only.** Date math, unit conversion, latest-reading selection, thresholds, and the decision are all in Python.
- **Every rule always runs**, and the output lists every issue. Each issue has `category`, `description`, and `evidence{source, details}`; `details` includes the LLM's quote from the record.
- **Low-confidence fallback is asymmetric.** Uncertainty can add issues but never remove one, and never downgrades `NOT_CLEARED`. The threshold is 0.8, untuned (see the plan doc).

### Modules

| File | Purpose |
|---|---|
| `core.py` | `triage_submission`, decision precedence, explanation; re-exports schemas |
| `schemas.py` | Submission input, `CaseFacts` extraction, and triage output models |
| `prompts.py` | Static policy and extraction guide (cacheable prefix), case input builder |
| `extract.py` | Shared OpenAI client and the extraction request |
| `confidence.py` | Maps response logprobs to per-field confidences |
| `fallback.py` | Low-confidence demotion of values that would clear a check |
| `normalize.py` | Date parsing, unit conversion, windows, latest-reading selection |
| `rules.py` | One function per policy rule, plus missing-data checks |

### OpenAI API usage

- One shared client with retries and a timeout.
- Pinned model snapshot (`gpt-4.1-mini` → `gpt-4.1-mini-2025-04-14`) and `temperature=0`.
- Strict structured outputs via `responses.parse`. `documents` and `labs` are pinned to the submission's item counts per request, and each item carries an explicit `index`.
- Static instructions first and `prompt_cache_key` set, for prompt caching. The per-case schema limits cache hits on the schema portion.
- `store=False`, since submissions contain PHI.
- Logprobs requested for confidence gating.

### Known limitations

- The confidence threshold is not calibrated, and logprobs are poorly calibrated in general.
- The evaluation set is 50 cases. Rules follow the policy text and generic data patterns, not individual cases, but there is no holdout set.
- `exact_output_match_pct` measured 100%, but it can fall below that, because evidence text comes from the LLM.
- Cases run sequentially; `run_baseline.py` is unchanged.

## Original Brief

### Objective

Implement `triage_submission(...)` in `core.py` - it is a pre-op triage function for a single submission package. It is currently a naive LLM-based solution that makes a real model API call. The starter implementation intentionally does not follow some best practices in using the OpenAI API. You may use whatever file structure makes sense for your solution.

Your output must match this schema:

- `decision`: `READY | NEEDS_FOLLOW_UP | NOT_CLEARED`
- `issues[]`: `category` + evidence (how you design this is up to you)
- `explanation`

### What Is Provided

- `data/patients_sample_50.jsonl` includes:
  - `case_id`
  - `submission`
  - `label` (what a human labeled this case as)
- `run_baseline.py` runs your `triage_submission` implementation and writes outputs.
- `run_evals.py` scores outputs against provided `label` and can run determinism checks.

### Completion

Note: this exercise is evaluated on engineering judgment. You may not reach a 100% score, and that is OK! We are looking to understand how you approached the problem and designed a working solution.

## Setup

1. Confirm `uv` is installed.

```bash
uv --version
```

2. Set your OpenAI API key, either exported or in a git-ignored `.env` file. Never commit the key.

```bash
export OPENAI_API_KEY="<your_api_key>"
# or, with a .env file:
uv run --env-file .env make baseline evals
```

## Recommended Workflow

1. Implement `triage_submission` in `core.py`.
2. Run baseline outputs:

```bash
make baseline
```

3. Run eval scoring:

```bash
make evals
```

4. Run determinism check:

```bash
make determinism
```

5. Print score:

```bash
make score
```

6. View the interactive report (TUI):

```bash
make report
```

This opens a terminal UI (`view_report.py`) that shows per-case results side-by-side with oracle expectations. You can browse records, see metric pass/fail status, and inspect submission data. Press `f` on a metric row to filter the case list to failures. Press `q` to quit.

## Tests

Run the unit tests (no network; the OpenAI client is mocked) with:

```bash
make test
```

## Outputs

- Baseline outputs: `data/baseline_outputs.jsonl`
- Eval report: `data/eval_report.json`
- Determinism report: `data/determinism_report.json`

## Configurable Variables

- `MODEL` (default `gpt-4.1-mini`)
- `INPUT` (default `data/patients_sample_50.jsonl`)
- `OUTPUT` (default `data/baseline_outputs.jsonl`)
- `REPORT` (default `data/eval_report.json`)
- `DETERMINISM_REPORT` (default `data/determinism_report.json`)
