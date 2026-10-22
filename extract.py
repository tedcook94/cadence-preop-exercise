"""Single LLM call that turns a submission into structured CaseFacts."""

from typing import Any

from openai import OpenAI

from prompts import INSTRUCTIONS, build_case_input
from schemas import CaseFacts

# Aliases float to newer snapshots over time; pinning keeps behavior stable.
MODEL_SNAPSHOTS = {"gpt-4.1-mini": "gpt-4.1-mini-2025-04-14"}

PROMPT_CACHE_KEY = "preop-triage-extraction-v1"
REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 3

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return the process-wide client, creating it on first use."""

    global _client
    if _client is None:
        _client = OpenAI(max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT_S)
    return _client


def resolve_model(model: str) -> str:
    return MODEL_SNAPSHOTS.get(model, model)


def build_request(submission: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": resolve_model(model),
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": build_case_input(submission)}],
        "text_format": CaseFacts,
        "temperature": 0,
        # Responses are retained by default; submissions contain PHI.
        "store": False,
        "prompt_cache_key": PROMPT_CACHE_KEY,
    }


def extract_case_facts(submission: dict[str, Any], *, model: str) -> CaseFacts:
    """Extract CaseFacts from one submission with strict structured outputs."""

    response = get_client().responses.parse(**build_request(submission, model))
    facts = response.output_parsed
    if facts is None:
        raise ValueError("Model returned no parsed CaseFacts (refusal or empty output)")
    return facts
