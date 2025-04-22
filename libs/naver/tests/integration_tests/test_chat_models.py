"""Test ChatClovaX chat model."""

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)
from pydantic import BaseModel, Field

from langchain_naver.chat_models import ChatClovaX


class GetWeather(BaseModel):
    """주어진 위치의 현재 날씨를 조회합니다."""

    location: str = Field(
        ...,
        description="날씨를 조회하고자 하는 위치의 시도명. 예) 경기도 성남시 분당구",
    )


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


def test_invoke_with_extra_body() -> None:
    messages = [
        (
            "system",
            "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 수 "
            "있는 개발 도구입니다.",
        ),
        (
            "human",
            "CLOVA Studio는 무엇인가요?",
        ),
    ]
    llm = ChatClovaX(top_k=30, repetition_penalty=0.5)
    result = llm.invoke(messages)
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


def test_invoke_bind_tools() -> None:
    """Test function call from ChatClovaX."""
    llm = ChatClovaX(max_tokens=2048, top_k=5, repetition_penalty=1.0)
    chat_with_tool = llm.bind_tools([GetWeather])
    result = chat_with_tool.invoke("분당과 판교 중 어디가 더 덥지?")
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert isinstance(result.tool_calls, list)


async def test_ainvoke_bind_tools() -> None:
    """Test function call from ChatClovaX."""
    llm = ChatClovaX(max_tokens=2048, top_k=5, repetition_penalty=1.0)
    chat_with_tool = llm.bind_tools([GetWeather])
    result = await chat_with_tool.ainvoke("분당과 판교 중 어디가 더 덥지?")
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert isinstance(result.tool_calls, list)


def test_langgraph_create_react_agent() -> None:
    """Test LangGraph from ChatClovaX."""
    from langgraph.prebuilt import create_react_agent

    # Define tool for test
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    # Define the chat model
    chat = ChatClovaX(
        model="HCX-005",
        max_tokens=1024,
        disable_streaming=True,
    )

    # Create the prebuilt react agent
    agent = create_react_agent(
        model=chat, tools=[get_weather], prompt="You are a helpful assistant"
    )

    agent.invoke({"messages": "what is the weather in seoul?"})


@pytest.mark.skip(reason="changed target model")
def test_stream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(ValueError):
        for _ in llm.stream(prompt * 1000):
            pass


@pytest.mark.skip(reason="changed target model")
async def test_astream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(ValueError):
        async for _ in llm.astream(prompt * 1000):
            pass
