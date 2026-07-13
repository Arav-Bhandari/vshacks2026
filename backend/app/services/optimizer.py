"""Generate protocol improvement suggestions."""
import json

from app.config import DEEPSEEK_MODEL
from app.services.llm_utils import NON_THINKING, _parse_json_response, get_client


def _format_trials(trials: list[dict]) -> str:
    lines = []
    for t in trials[:10]:
        lines.append(
            f"- {t.get('nct_id')}: {t.get('title')} "
            f"(Phase {t.get('phase')}, {t.get('duration_months')} mo, "
            f"n={t.get('enrollment')}) outcomes: {t.get('primary_outcomes')}"
        )
    return "\n".join(lines) or "(none)"


def _build_prompt(usdm: dict, similar_trials: list[dict], fda_analysis: dict, burden: dict) -> str:
    return (
        "You are optimizing a clinical trial protocol. Use the context below "
        "to propose concrete improvements. Every change must cite an NCT ID "
        "from the similar trials list or an FDA document name from the FDA "
        "analysis.\n\n"
        f"Current protocol (USDM JSON):\n{json.dumps(usdm)}\n\n"
        f"Top similar trials:\n{_format_trials(similar_trials)}\n\n"
        f"FDA analysis / gaps:\n{json.dumps(fda_analysis)}\n\n"
        f"Patient burden scores:\n{json.dumps(burden)}\n\n"
        "Return ONLY raw JSON, no code fences, no commentary, in this shape:\n"
        '{"summary": "", "changes": [{"section": "", "change": "", '
        '"rationale": "", "citation": ""}], "markdown": ""}\n'
        "The \"markdown\" field is the full improved protocol draft."
    )


async def optimize_protocol(usdm: dict, similar_trials: list[dict], fda_analysis: dict, burden: dict) -> dict:
    client = get_client()
    prompt = _build_prompt(usdm, similar_trials, fda_analysis, burden)

    async def ask(p: str) -> str:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=8000,
            extra_body=NON_THINKING,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": p}],
        )
        return response.choices[0].message.content or ""

    raw = await ask(prompt)
    try:
        return _parse_json_response(raw)
    except (ValueError, json.JSONDecodeError):
        retry_prompt = prompt + "\n\nYour previous response was not valid JSON. Return ONLY raw JSON."
        raw = await ask(retry_prompt)
        return _parse_json_response(raw)
