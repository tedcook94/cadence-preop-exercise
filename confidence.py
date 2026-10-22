"""Map token logprobs from the extraction response onto enum and boolean fields.

A field's confidence is the joint probability of the tokens that spell its value.
Logprobs are poorly calibrated, so these numbers gate decisions; they are not
accuracy estimates.
"""

import json.decoder
import math
import re
from typing import Any

CONFIDENCE_KEYS = frozenset(
    {
        "doc_kind",
        "consent_status",
        "consent_matches_procedure",
        "test",
        "kind",
        "unit",
        "status",
        "plan_has_hold",
        "plan_has_resume",
        "plan_matches_drug",
    }
)

# Zero or more JSON whitespace characters (space, tab, newline, carriage return).
_WS = re.compile(r"[ \t\n\r]*")
# A bare JSON literal: true, false, null, or a number with optional sign,
# fraction, and exponent (e.g. -3, 98.6, 1e-5).
_LITERAL = re.compile(r"true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

Span = tuple[int, int]


def field_value_spans(text: str) -> dict[str, Span]:
    """Return character spans of tracked values in a JSON document, keyed by path."""

    spans: dict[str, Span] = {}

    def skip_ws(i: int) -> int:
        return _WS.match(text, i).end()

    def parse(i: int, path: str, key: str | None) -> int:
        i = skip_ws(i)
        char = text[i]
        if char == "{":
            i = skip_ws(i + 1)
            if text[i] == "}":
                return i + 1
            while True:
                name, i = json.decoder.scanstring(text, skip_ws(i) + 1)
                i = skip_ws(i) + 1  # colon
                i = skip_ws(parse(i, f"{path}.{name}" if path else name, name))
                if text[i] == "}":
                    return i + 1
                i += 1  # comma
        if char == "[":
            i = skip_ws(i + 1)
            if text[i] == "]":
                return i + 1
            index = 0
            while True:
                i = skip_ws(parse(i, f"{path}[{index}]", None))
                if text[i] == "]":
                    return i + 1
                i += 1
                index += 1
        if char == '"':
            _, end = json.decoder.scanstring(text, i + 1)
            if key in CONFIDENCE_KEYS:
                spans[path] = (i + 1, end - 1)
            return end
        match = _LITERAL.match(text, i)
        if match is None:
            raise ValueError(f"Unexpected JSON at offset {i}")
        if key in CONFIDENCE_KEYS and match.group() in ("true", "false"):
            spans[path] = match.span()
        return match.end()

    parse(0, "", None)
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
