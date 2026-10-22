"""Map token logprobs from the extraction response onto enum and boolean fields.

A field's confidence is the joint probability of the tokens that spell its value.
Logprobs are poorly calibrated, so these numbers gate decisions; they are not
accuracy estimates.
"""

import math
import re
from collections import Counter
from typing import Any

# Tracked key -> the CaseFacts list it belongs to. Each key must appear in only
# one list, so the n-th occurrence of a key in the output is item n of that list.
SECTION_OF: dict[str, str] = {
    "doc_kind": "documents",
    "consent_status": "documents",
    "consent_matches_procedure": "documents",
    "test": "labs",
    "kind": "note_vitals",
    "unit": "note_vitals",
    "status": "anticoagulants",
    "plan_has_hold": "anticoagulants",
    "plan_has_resume": "anticoagulants",
    "plan_matches_drug": "anticoagulants",
}
CONFIDENCE_KEYS = frozenset(SECTION_OF)

# A tracked key followed by its value: an enum string (no quotes or escapes),
# or a bare true/false/null. Keys quoted inside free text appear as \"key\"
# and do not match because the opening quote must not be escaped.
_FIELD = re.compile(
    r'(?<!\\)"(?P<key>' + "|".join(SECTION_OF) + r')"\s*:\s*'
    r'(?:"(?P<str>[^"\\]*)"|(?P<lit>true|false|null))'
)

Span = tuple[int, int]


def field_value_spans(text: str) -> dict[str, Span]:
    """Return character spans of tracked values in a JSON document, keyed by path."""

    seen: Counter[str] = Counter()
    spans: dict[str, Span] = {}
    for match in _FIELD.finditer(text):
        key = match["key"]
        index = seen[key]
        seen[key] += 1
        group = "str" if match["str"] is not None else "lit"
        if match[group] != "null":
            spans[f"{SECTION_OF[key]}[{index}].{key}"] = match.span(group)
    return spans


def _output_tokens(response: Any) -> list[tuple[str, float]]:
    """Flatten (token, logprob) pairs from every output_text part of a response.

    Returns an empty list when the response carries no logprobs, e.g. when the
    request omitted the logprobs include or the model does not support them.
    """

    tokens: list[tuple[str, float]] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            for entry in getattr(content, "logprobs", None) or []:
                tokens.append((entry.token, entry.logprob))
    return tokens


def field_confidences(response: Any) -> dict[str, float]:
    """Return {field path: probability} for tracked fields, or {} if unavailable."""

    tokens = _output_tokens(response)
    text = "".join(token for token, _ in tokens)
    if not tokens or text != getattr(response, "output_text", text):
        return {}

    starts: list[int] = []
    offset = 0
    for token, _ in tokens:
        starts.append(offset)
        offset += len(token)

    confidences: dict[str, float] = {}
    for path, (start, end) in field_value_spans(text).items():
        total = sum(
            logprob
            for (token, logprob), token_start in zip(tokens, starts)
            if token_start < end and token_start + len(token) > start
        )
        confidences[path] = math.exp(total)
    return confidences
