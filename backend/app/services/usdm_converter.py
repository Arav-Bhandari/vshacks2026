"""Convert protocol text to USDM JSON."""
import json
import re

from app.config import DEEPSEEK_MODEL
from app.services.llm_utils import NON_THINKING, _parse_json_response, get_client

MAX_INPUT_CHARS = 150_000

_ROMAN_TO_ARABIC = {"I": "1", "II": "2", "III": "3", "IV": "4"}

USDM_SHAPE = """{
  "study": {
    "name": "",
    "description": "",
    "studyType": "INTERVENTIONAL|OBSERVATIONAL|EXPANDED_ACCESS|",
    "phase": "",
    "therapeuticArea": "",
    "sponsor": {"name": "", "class": "INDUSTRY|NIH|FED|OTHER|"},
    "plannedStartDate": "",
    "conditions": [],
    "interventions": [{"name": "", "type": "", "description": ""}],
    "objectives": [
      {
        "level": "primary|secondary",
        "description": "",
        "endpoints": [{"name": "", "description": "", "timeframe": ""}]
      }
    ],
    "population": {
      "criteria": {"inclusion": [], "exclusion": []},
      "plannedEnrollment": "",
      "minimumAge": "",
      "maximumAge": "",
      "ageRange": "",
      "sex": "",
      "healthyVolunteers": ""
    },
    "arms": [{"name": "", "type": "", "description": ""}],
    "design": {
      "allocation": "",
      "masking": "",
      "interventionModel": "",
      "primaryPurpose": "",
      "randomization": ""
    },
    "sites": [{"name": "", "country": ""}],
    "countries": [],
    "collaborators": [{"name": "", "class": ""}],
    "scheduleOfActivities": {
      "visits": [{"name": "", "timing": "", "procedures": []}]
    },
    "estimatedDuration": ""
  }
}"""


def normalize_phase(s: str) -> str:
    if not s:
        return s

    def repl(m: re.Match) -> str:
        val = m.group(1).upper()
        return f"Phase {_ROMAN_TO_ARABIC.get(val, val)}"

    return re.sub(r"(?i)phase\s*([ivx]+|\d+)", repl, s)


def _build_prompt(markdown: str) -> str:
    return (
        "Convert the following clinical trial protocol document into "
        "CDISC USDM v3-flavored JSON. Return ONLY the raw JSON object, "
        "no markdown code fences, no commentary.\n\n"
        f"Required top-level shape:\n{USDM_SHAPE}\n\n"
        "Fill every field you can infer from the document; use empty "
        "string/array when information is not present.\n\n"
        f"Document:\n{markdown}"
    )


async def _ask(client, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        max_tokens=8000,
        extra_body=NON_THINKING,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _normalize(data: dict) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("study"), dict):
        raise ValueError("USDM conversion must return an object containing a study object")

    def prune(value):
        if isinstance(value, dict):
            cleaned = {key: item for key, raw in value.items() if (item := prune(raw)) is not None}
            return cleaned or None
        if isinstance(value, list):
            cleaned = [item for raw in value if (item := prune(raw)) is not None]
            return cleaned or None
        if isinstance(value, str):
            text = value.strip()
            return None if not text or ("|" in text and text.endswith("|")) else text
        return value if value is not None else None

    normalized = prune(data) or {"study": {}}
    study = normalized.setdefault("study", {})
    if study.get("phase"):
        study["phase"] = normalize_phase(study["phase"])
    return normalized


async def convert_to_usdm(markdown: str) -> dict:
    truncated = markdown
    if len(markdown) > MAX_INPUT_CHARS:
        truncated = markdown[:MAX_INPUT_CHARS] + "\n\n[NOTE: input truncated to 150K characters]"

    client = get_client()
    prompt = _build_prompt(truncated)
    raw = await _ask(client, prompt)
    try:
        data = _parse_json_response(raw)
    except (ValueError, json.JSONDecodeError):
        retry_prompt = prompt + (
            "\n\nYour previous response was not valid JSON. "
            "Return ONLY the raw JSON object, no commentary, no code fences."
        )
        raw = await _ask(client, retry_prompt)
        data = _parse_json_response(raw)

    return _normalize(data)
