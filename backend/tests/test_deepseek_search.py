import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.routes import search as mod


def _response(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_nl_search_runs_deepseek_tool_cycle(monkeypatch):
    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="search_trials",
            arguments=json.dumps({"query": "oncology", "limit": 2}),
        ),
    )
    create = AsyncMock(
        side_effect=[_response(tool_calls=[call]), _response("Two trials found")]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    trial = {"nct_id": "NCT1", "title": "Trial"}
    monkeypatch.setattr(mod, "DEEPSEEK_API_KEY", "test-key")

    with (
        patch.object(mod, "get_client", return_value=client),
        patch.object(mod.db, "search_trials", return_value=[trial]) as search,
    ):
        result = await mod.search_nl(mod.NLQuery(query="find oncology trials"))

    assert result == {"answer": "Two trials found", "trials": [trial]}
    search.assert_called_once_with("oncology", 2)
    assert create.await_args_list[0].kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    second_messages = create.await_args_list[1].kwargs["messages"]
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call_1"
