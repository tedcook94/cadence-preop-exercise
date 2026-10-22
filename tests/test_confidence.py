import math
from typing import get_args
from types import SimpleNamespace

import pytest

from confidence import SECTION_OF, field_confidences, field_value_spans
from schemas import CaseFacts

TEXT = (
    '{"documents":[{"index":0,"doc_kind":"CONSENT","consent_status":"SIGNED",'
    '"evidence":"status: pending"}],'
    '"anticoagulants":[{"drug":"apixaban","plan_has_hold":true}]}'
)


def _response(tokens: list[tuple[str, float]], output_text: str | None = None) -> SimpleNamespace:
    logprobs = [SimpleNamespace(token=t, logprob=lp) for t, lp in tokens]
    content = SimpleNamespace(type="output_text", logprobs=logprobs)
    text = "".join(t for t, _ in tokens) if output_text is None else output_text
    return SimpleNamespace(output=[SimpleNamespace(content=[content])], output_text=text)


def _tokenize(text: str, low: dict[str, float]) -> list[tuple[str, float]]:
    """Split text so each value in `low` is its own token with the given logprob."""

    tokens: list[tuple[str, float]] = []
    rest = text
    for value, logprob in low.items():
        head, rest = rest.split(value, 1)
        tokens += [(head, 0.0), (value, logprob)]
    return tokens + [(rest, 0.0)]


def test_spans_cover_only_tracked_fields() -> None:
    spans = {path: TEXT[s:e] for path, (s, e) in field_value_spans(TEXT).items()}

    assert spans == {
        "documents[0].doc_kind": "CONSENT",
        "documents[0].consent_status": "SIGNED",
        "anticoagulants[0].plan_has_hold": "true",
    }


def test_escaped_quotes_in_strings_do_not_break_spans() -> None:
    text = (
        '{"documents":[{"evidence":"says \\"doc_kind\\":\\"CONSENT\\" \\\\",'
        '"doc_kind":"OTHER"}]}'
    )

    spans = {path: text[s:e] for path, (s, e) in field_value_spans(text).items()}

    assert spans == {"documents[0].doc_kind": "OTHER"}


def test_null_values_are_counted_but_not_spanned() -> None:
    text = '{"note_vitals":[{"kind":"BP","unit":null},{"kind":"TEMP","unit":"C"}]}'

    spans = {path: text[s:e] for path, (s, e) in field_value_spans(text).items()}

    assert spans == {
        "note_vitals[0].kind": "BP",
        "note_vitals[1].kind": "TEMP",
        "note_vitals[1].unit": "C",
    }


def test_section_map_matches_schema() -> None:
    sections: dict[str, list[str]] = {}
    for section, field in CaseFacts.model_fields.items():
        item_model = get_args(field.annotation)[0]
        for key in item_model.model_fields:
            sections.setdefault(key, []).append(section)

    for key, section in SECTION_OF.items():
        assert sections[key] == [section], f"{key} must belong only to {section}"


def test_confidence_is_joint_probability_of_value_tokens() -> None:
    tokens = _tokenize(TEXT, {"SIG": math.log(0.5), "NED": math.log(0.8), "true": math.log(0.9)})

    confidences = field_confidences(_response(tokens))

    assert confidences["documents[0].doc_kind"] == pytest.approx(1.0)
    assert confidences["documents[0].consent_status"] == pytest.approx(0.4)
    assert confidences["anticoagulants[0].plan_has_hold"] == pytest.approx(0.9)


def test_token_merged_with_quotes_still_counts() -> None:
    tokens = _tokenize(TEXT, {'":"SIGNED","': math.log(0.3)})

    confidences = field_confidences(_response(tokens))

    assert confidences["documents[0].consent_status"] == pytest.approx(0.3)


def test_missing_or_mismatched_logprobs_yield_empty() -> None:
    assert field_confidences(_response([])) == {}
    assert field_confidences(_response([(TEXT, 0.0)], output_text=TEXT + " ")) == {}
