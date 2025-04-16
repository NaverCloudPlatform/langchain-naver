"""Test ChatClovaX chat model."""

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)

from langchain_naver.chat_models import ChatClovaX


def test_stream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX()

    for token in llm.stream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert token.response_metadata["model_name"]
            assert token.response_metadata["finish_reason"]
            if "token_usage" in token.response_metadata:
                token_usage = token.response_metadata["token_usage"]
                assert token_usage["completion_tokens"]
                assert token_usage["prompt_tokens"]
                assert token_usage["total_tokens"]


async def test_astream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX()

    async for token in llm.astream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert token.response_metadata["model_name"]
            assert token.response_metadata["finish_reason"]
            if "token_usage" in token.response_metadata:
                token_usage = token.response_metadata["token_usage"]
                assert token_usage["completion_tokens"]
                assert token_usage["prompt_tokens"]
                assert token_usage["total_tokens"]


async def test_abatch() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = await llm.abatch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert token.response_metadata["model_name"]
            assert token.response_metadata["finish_reason"]
            if "token_usage" in token.response_metadata:
                token_usage = token.response_metadata["token_usage"]
                assert token_usage["completion_tokens"]
                assert token_usage["prompt_tokens"]
                assert token_usage["total_tokens"]


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = await llm.abatch(["I'm Clova", "I'm not Clova"], config={"tags": ["foo"]})
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert token.response_metadata["model_name"]
            assert token.response_metadata["finish_reason"]
            if "token_usage" in token.response_metadata:
                token_usage = token.response_metadata["token_usage"]
                assert token_usage["completion_tokens"]
                assert token_usage["prompt_tokens"]
                assert token_usage["total_tokens"]


def test_batch() -> None:
    """Test batch tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = llm.batch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert token.response_metadata["model_name"]
            assert token.response_metadata["finish_reason"]
            if "token_usage" in token.response_metadata:
                token_usage = token.response_metadata["token_usage"]
                assert token_usage["completion_tokens"]
                assert token_usage["prompt_tokens"]
                assert token_usage["total_tokens"]


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = await llm.ainvoke("I'm Clova", config={"tags": ["foo"]})
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    if result.response_metadata:
        assert result.response_metadata["model_name"]
        assert result.response_metadata["finish_reason"]
        if "token_usage" in result.response_metadata:
            token_usage = result.response_metadata["token_usage"]
            assert token_usage["completion_tokens"]
            assert token_usage["prompt_tokens"]
            assert token_usage["total_tokens"]


def test_invoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = llm.invoke("I'm Clova", config=dict(tags=["foo"]))
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    if result.response_metadata:
        assert result.response_metadata["model_name"]
        assert result.response_metadata["finish_reason"]
        if "token_usage" in result.response_metadata:
            token_usage = result.response_metadata["token_usage"]
            assert token_usage["completion_tokens"]
            assert token_usage["prompt_tokens"]
            assert token_usage["total_tokens"]


def test_stream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(ValueError):
        for _ in llm.stream(prompt * 1000):
            pass


async def test_astream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(ValueError):
        async for _ in llm.astream(prompt * 1000):
            pass
