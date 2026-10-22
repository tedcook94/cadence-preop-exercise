
from typing import Any

import pytest
from pydantic import ValidationError

from schemas import CaseFacts, case_facts_json_schema


def _object_schemas(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_object_schemas(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_object_schemas(item))
    return found


def _contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        return key in node or any(_contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(item, key) for item in node)
    return False


def test_schema_meets_strict_mode_requirements() -> None:
    schema = case_facts_json_schema()
    objects = _object_schemas(schema)

    assert objects
    for obj in objects:
        assert obj.get("additionalProperties") is False
        assert sorted(obj.get("required", [])) == sorted(obj["properties"])
    assert not _contains_key(schema, "default")


def test_rejects_unknown_enum_and_extra_keys() -> None:
    doc = {
        "index": 0,
        "doc_kind": "HP_PREOP",
        "consent_status": "N/A",
        "consent_matches_procedure": "N/A",
        "evidence": "documents[0]: pre-op H&P",
    }
    base = {"documents": [doc], "labs": [], "note_vitals": [], "anticoagulants": []}
    CaseFacts.model_validate(base)

    with pytest.raises(ValidationError):
        CaseFacts.model_validate({**base, "documents": [{**doc, "doc_kind": "H&P"}]})
    with pytest.raises(ValidationError):
        CaseFacts.model_validate({**base, "extra": True})
    with pytest.raises(ValidationError):
        CaseFacts.model_validate({k: v for k, v in base.items() if k != "labs"})
