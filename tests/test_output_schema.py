import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas import TriageOutput

EXAMPLE_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "example_output.json"


def test_example_output_validates() -> None:
    payload = json.loads(EXAMPLE_OUTPUT.read_text(encoding="utf-8"))

    output = TriageOutput.model_validate(payload)

    assert output.model_dump() == payload


def test_issue_requires_structured_evidence() -> None:
    issue = {
        "category": "REQUIRED_TESTING",
        "description": "Missing CBC",
        "evidence": "no CBC in labs",
    }

    with pytest.raises(ValidationError):
        TriageOutput.model_validate(
            {"decision": "NEEDS_FOLLOW_UP", "issues": [issue], "explanation": "x"}
        )
