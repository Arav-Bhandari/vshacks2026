"""Shared language-model helpers."""
import json
import re

from openai import AsyncOpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

NON_THINKING = {"thinking": {"type": "disabled"}}


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])
