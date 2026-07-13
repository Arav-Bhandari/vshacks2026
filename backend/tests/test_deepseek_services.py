import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.optimizer import optimize_protocol
from app.services.usdm_converter import convert_to_usdm
from app.services import llm_utils


def _response(content: str):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(*responses):
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, create


def test_client_uses_deepseek_endpoint():
    with patch.object(llm_utils, "AsyncOpenAI") as client_class:
        llm_utils.get_client()

    client_class.assert_called_once_with(
        api_key=llm_utils.DEEPSEEK_API_KEY,
        base_url=llm_utils.DEEPSEEK_BASE_URL,
    )


@pytest.mark.asyncio
async def test_usdm_conversion_uses_chat_completions():
    client, create = _client(_response('{"study":{"name":"Trial"}}'))

    with patch("app.services.usdm_converter.get_client", return_value=client):
        result = await convert_to_usdm("# Trial")

    assert result == {"study": {"name": "Trial"}}
    kwargs = create.await_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_optimizer_uses_chat_completions():
    payload = {"summary": "Done", "changes": [], "markdown": "# Trial"}
    client, create = _client(_response(json.dumps(payload)))

    with patch("app.services.optimizer.get_client", return_value=client):
        result = await optimize_protocol({}, [], {}, {})

    assert result == payload
    kwargs = create.await_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "thinking" not in kwargs
