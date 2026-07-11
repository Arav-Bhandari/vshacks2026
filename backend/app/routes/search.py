"""Keyword and natural-language trial search."""
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import ANTHROPIC_API_KEY, SONNET_MODEL
from app.database import db
from app.services.llm_utils import get_client

router = APIRouter(tags=["search"])

_TOOL = {
    "name": "search_trials",
    "description": "Full-text search clinical trials by keywords.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["query"],
    },
}


class NLQuery(BaseModel):
    query: str


@router.get("/api/trials/search")
async def search(q: str, limit: int = 50):
    return {"trials": db.search_trials(q, limit)}


@router.post("/api/search/nl")
async def search_nl(body: NLQuery):
    if not ANTHROPIC_API_KEY:
        trials = db.search_trials(body.query, 20)
        return {"answer": "AI search unavailable, showing keyword results", "trials": trials}

    client = get_client()
    messages = [{"role": "user", "content": body.query}]
    seen: dict[str, dict] = {}

    for _ in range(3):
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=1024,
            tools=[_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            text = "".join(b.text for b in response.content if b.type == "text")
            trials = list(seen.values())[:20]
            return {"answer": text, "trials": trials}

        results = []
        for call in tool_calls:
            trials = db.search_trials(
                call.input.get("query", ""), call.input.get("limit", 20)
            )
            for t in trials:
                if t.get("nct_id") and t["nct_id"] not in seen:
                    seen[t["nct_id"]] = t
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(trials),
                }
            )
        messages.append({"role": "user", "content": results})

    trials = list(seen.values())[:20]
    return {"answer": "", "trials": trials}
