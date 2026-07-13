"""Search clinical trials."""
import json

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from app.database import db
from app.services.llm_utils import NON_THINKING, get_client

router = APIRouter(tags=["search"])

_TOOL = {
    "type": "function",
    "function": {
        "name": "search_trials",
        "description": "Full-text search clinical trials by keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
}


class NLQuery(BaseModel):
    query: str


@router.get("/api/trials/search")
async def search(q: str, limit: int = 50):
    return {"trials": db.search_trials(q, limit)}


@router.post("/api/search/nl")
async def search_nl(body: NLQuery):
    if not DEEPSEEK_API_KEY:
        trials = db.search_trials(body.query, 20)
        return {"answer": "AI search unavailable, showing keyword results", "trials": trials}

    client = get_client()
    messages = [{"role": "user", "content": body.query}]
    seen: dict[str, dict] = {}

    for _ in range(3):
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=1024,
            extra_body=NON_THINKING,
            tools=[_TOOL],
            messages=messages,
        )
        message = response.choices[0].message
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }

        tool_calls = message.tool_calls or []
        if not tool_calls:
            trials = list(seen.values())[:20]
            return {"answer": message.content or "", "trials": trials}

        assistant_message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
        messages.append(assistant_message)

        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            trials = db.search_trials(
                arguments.get("query", ""), arguments.get("limit", 20)
            )
            for t in trials:
                if t.get("nct_id") and t["nct_id"] not in seen:
                    seen[t["nct_id"]] = t
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(trials),
                }
            )

    trials = list(seen.values())[:20]
    return {"answer": "", "trials": trials}
