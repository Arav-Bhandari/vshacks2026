"""Shared Anthropic client + robust JSON parsing for LLM-backed services."""
import json
import re

from anthropic import AsyncAnthropic

from app.config import ANTHROPIC_API_KEY


def get_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])
