from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import extract
from prompts import INSTRUCTIONS, build_case_input
from schemas import CaseFacts

FACTS = CaseFacts(documents=[], labs=[], note_vitals=[], anticoagulants=[])
SUBMISSION = {"procedure": {"procedure_risk": "LOW"}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock()
    mock.responses.parse.return_value = SimpleNamespace(output_parsed=FACTS)
    monkeypatch.setattr(extract, "_client", mock)
    return mock


def test_request_uses_strict_cached_deterministic_settings(client: Mock) -> None:
    result = extract.extract_case_facts(SUBMISSION, model="gpt-4.1-mini")

    assert result.facts == FACTS
    assert result.confidences == {}
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-mini-2025-04-14"
    assert kwargs["instructions"] == INSTRUCTIONS
    assert kwargs["input"] == [{"role": "user", "content": build_case_input(SUBMISSION)}]
    assert kwargs["text_format"] is CaseFacts
    assert kwargs["temperature"] == 0
    assert kwargs["store"] is False
    assert kwargs["prompt_cache_key"] == extract.PROMPT_CACHE_KEY
    assert kwargs["include"] == ["message.output_text.logprobs"]


def test_unknown_model_passes_through(client: Mock) -> None:
    extract.extract_case_facts(SUBMISSION, model="test-model")

    assert client.responses.parse.call_args.kwargs["model"] == "test-model"


def test_client_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = Mock()
    monkeypatch.setattr(extract, "_client", None)
    monkeypatch.setattr(extract, "OpenAI", constructor)

    assert extract.get_client() is extract.get_client()
    constructor.assert_called_once_with(
        max_retries=extract.MAX_RETRIES, timeout=extract.REQUEST_TIMEOUT_S
    )


def test_missing_parsed_output_raises(client: Mock) -> None:
    client.responses.parse.return_value = SimpleNamespace(output_parsed=None)

    with pytest.raises(ValueError):
        extract.extract_case_facts(SUBMISSION, model="test-model")
