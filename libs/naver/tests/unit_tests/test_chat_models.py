import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)

from langchain_naver import ChatClovaX

os.environ["CLOVASTUDIO_API_KEY"] = "test_api_key"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatClovaX()


def test_chat_model_param() -> None:
    llm = ChatClovaX(model="foo")
    assert llm.model_name == "foo"
    llm = ChatClovaX(model_name="foo")  # type: ignore
    assert llm.model_name == "foo"
    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "naver"


def test_function_dict_to_message_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test_convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "65caeb6999d34615ae7c217583eb366b",
        "created": 1744703673905,
        "model": "HCX-003",
        "usage": {"completion_tokens": 85, "prompt_tokens": 161, "total_tokens": 246},
        "seed": 1,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bab",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_chat_model_invoke(mock_completion: dict) -> None:
    llm = ChatClovaX()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bab")
        assert res.content == "Bab"
    assert completed


async def test_chat_model_ainvoke(mock_completion: dict) -> None:
    llm = ChatClovaX()
    mock_client = AsyncMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.ainvoke("bab")
        assert res.content == "Bab"
    assert completed


def test_chat_model_extra_kwargs() -> None:
    """Test extra kwargs to chat model."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatClovaX(foo=3, max_tokens=10)  # type: ignore
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatClovaX(foo=3, model_kwargs={"bar": 2})  # type: ignore
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatClovaX(foo=3, model_kwargs={"foo": 2})  # type: ignore
