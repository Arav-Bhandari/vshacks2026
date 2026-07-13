"""Check a protocol against FDA guidance."""
import json
import re

import fitz

from app.config import DEEPSEEK_MODEL, FDA_DIR
from app.services.llm_utils import NON_THINKING, _parse_json_response, get_client

MANIFEST_PATH = FDA_DIR / "manifest.json"
DOC_CHAR_BUDGET = 40000
TOTAL_CHAR_BUDGET = 150000


def list_guidance_docs() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    docs = json.loads(MANIFEST_PATH.read_text())
    return [d for d in docs if (FDA_DIR / d["filename"]).exists()]


def extract_pdf_text(path, max_chars: int = 60000) -> str:
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    return text[:max_chars]


def _study_summary(usdm: dict) -> str:
    study = usdm.get("study", usdm)
    versions = study.get("versions", [study]) if isinstance(study, dict) else [study]
    v = versions[0] if versions else {}
    phase = v.get("studyPhase") or usdm.get("phase") or "unknown"
    conditions = v.get("studyIndications") or usdm.get("conditions") or []
    interventions = v.get("studyInterventions") or usdm.get("interventions") or []
    area = v.get("therapeuticArea") or usdm.get("therapeuticArea") or "unknown"
    return (
        f"Phase: {phase}\nTherapeutic area: {area}\n"
        f"Conditions: {conditions}\nIntervention types: {interventions}"
    )


def _parse_indices(text: str, n: int) -> list[int]:
    match = re.search(r"\{([^{}]*)\}", text)
    if match:
        raw = match.group(1)
    else:
        raw = text
    found = re.findall(r"\d+", raw)
    indices = [int(i) for i in found if 0 <= int(i) < n]
    return indices[:3]


async def _select_docs(client, docs: list[dict], usdm: dict) -> list[int]:
    listing = "\n".join(f"{i}: [{d['category']}] {d['title']}" for i, d in enumerate(docs))
    prompt = (
        "Available FDA guidance documents:\n"
        f"{listing}\n\nStudy summary:\n{_study_summary(usdm)}\n\n"
        "Pick the 2-3 most relevant documents for an FDA compliance review. "
        "Answer with only the indices in curly brackets, e.g. {0,3,5}."
    )
    try:
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=100,
            extra_body=NON_THINKING,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        indices = _parse_indices(text, len(docs))
        if indices:
            return indices
    except Exception:
        pass
    fallback = [i for i, d in enumerate(docs) if d["category"] == "general"]
    return fallback[:3] if fallback else list(range(min(3, len(docs))))


def _build_docs_context(docs: list[dict], indices: list[int]) -> str:
    parts = []
    total = 0
    for i in indices:
        d = docs[i]
        budget = min(DOC_CHAR_BUDGET, TOTAL_CHAR_BUDGET - total)
        if budget <= 0:
            break
        text = extract_pdf_text(FDA_DIR / d["filename"], max_chars=budget)
        parts.append(f"=== {d['title']} ({d['category']}) ===\n{text}")
        total += len(text)
    return "\n\n".join(parts)


def _default_result() -> dict:
    return {
        "compliance_score": 0,
        "summary": "Analysis failed to produce a valid result.",
        "gaps": [],
        "strengths": [],
    }


async def analyze_fda_compliance(usdm: dict) -> dict:
    docs = list_guidance_docs()
    if not docs:
        result = _default_result()
        result["documents_used"] = []
        return result

    client = get_client()
    indices = await _select_docs(client, docs, usdm)
    docs_context = _build_docs_context(docs, indices)

    prompt = (
        "You are an FDA regulatory compliance reviewer. Using the guidance excerpts "
        "below, assess the following clinical trial protocol (in USDM JSON format).\n\n"
        f"GUIDANCE EXCERPTS:\n{docs_context}\n\nUSDM PROTOCOL:\n{json.dumps(usdm)}\n\n"
        "Respond with only JSON: {\"compliance_score\": 0-100, \"summary\": str, "
        "\"gaps\": [{\"element\", \"severity\": \"high|medium|low\", \"recommendation\", "
        "\"source\"}], \"strengths\": [str]}. The source field must name the guidance "
        "document it refers to."
    )

    result = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=8000,
            extra_body=NON_THINKING,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        try:
            result = _parse_json_response(text)
            break
        except (ValueError, json.JSONDecodeError):
            continue

    if result is None:
        result = _default_result()

    result["documents_used"] = [
        {"filename": docs[i]["filename"], "title": docs[i]["title"], "category": docs[i]["category"]}
        for i in indices
    ]
    return result
