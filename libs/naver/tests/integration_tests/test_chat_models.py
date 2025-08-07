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
    llm = ChatClovaX(max_tokens=2048, top_k=5, repetition_penalty=1.0)  # type: ignore[call-arg]
    chat_with_tool = llm.bind_tools([GetWeather])
    result = chat_with_tool.invoke("분당과 판교 중 어디가 더 덥지?")
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert isinstance(result.tool_calls, list)


async def test_ainvoke_bind_tools() -> None:
    """Test function call from ChatClovaX."""
    llm = ChatClovaX(max_tokens=2048, top_k=5, repetition_penalty=1.0)  # type: ignore[call-arg]
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
        max_tokens=1024,  # type: ignore[call-arg]
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


def test_invoke_reasoning() -> None:
    """Test reasoning(thinking) from ChatClovaX."""
    llm = ChatClovaX(
        model="HCX-007",
        max_completion_tokens=5120,  # or max_tokens=5120
        reasoning_effort="low",  # or thinking={"effort": "low"},
    )

    response = llm.invoke("What is the cube root of 50.653?")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert "thinking_content" in response.additional_kwargs
    assert response.type == "ai"
    if response.response_metadata:
        assert response.response_metadata["model_name"]
        assert response.response_metadata["finish_reason"]
        if "token_usage" in response.response_metadata:
            token_usage = response.response_metadata["token_usage"]
            assert token_usage["completion_tokens"]
            assert token_usage["prompt_tokens"]
            assert token_usage["total_tokens"]
            if "completion_tokens_details" in token_usage:
                completion_tokens_details = token_usage["completion_tokens_details"]
                assert completion_tokens_details["reasoning_tokens"] >= 0


def test_stream_reasoning() -> None:
    """Test reasoning(thinking) streaming tokens from ChatClovaX."""
    llm = ChatClovaX(
        model="HCX-007",
        max_completion_tokens=5120,  # or max_tokens=5120
        reasoning_effort="low",  # or thinking={"effort": "low"},
    )
    messages = [
        (
            "system",
            "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 수 "
            "있는 개발 도구입니다.",
        ),
        (
            "human",
            "What is the cube root of 50.653?",
        ),
    ]
    for token in llm.stream(messages):
        if isinstance(token, AIMessageChunk):
            assert isinstance(token.content, str)
            assert "thinking_content" in token.additional_kwargs
        if token.response_metadata:
            assert token.response_metadata["model_name"]
            assert token.response_metadata["finish_reason"]
            if "token_usage" in token.response_metadata:
                token_usage = token.response_metadata["token_usage"]
                assert token_usage["completion_tokens"]
                assert token_usage["prompt_tokens"]
                assert token_usage["total_tokens"]


def test_invoke_structured_output() -> None:
    """Test structured output from ChatClovaX."""

    class ResponseFormatter(BaseModel):
        """Always use this tool to structure your response to the user."""

        answer: str = Field(description="The answer to the user's question")
        followup_question: str = Field(
            description="A followup question the user could ask"
        )

    llm = ChatClovaX(
        model="HCX-007",
        max_completion_tokens=5120,  # or max_tokens=5120
        reasoning_effort="none",  # or  thinking = {"effort": "none"},
        disabled_params={"parallel_tool_calls": None},
    )
    # Bind the schema to the model
    model_with_structure = llm.with_structured_output(ResponseFormatter)
    # Invoke the model
    structured_output = model_with_structure.invoke(
        "What is the powerhouse of the cell?"
    )
    # Get back the pydantic object
    assert isinstance(structured_output, ResponseFormatter)
    assert len(structured_output.answer) > 0
    assert len(structured_output.followup_question) > 0


class SimpleTest(BaseModel):
    """Simple test model for structured output"""

    message: str = Field(description="Test message")
    success: bool = Field(description="Success status")


async def test_ainvoke_structured_output() -> None:
    """Test ainvoke structured output from ChatClovaX."""
    llm = ChatClovaX(
        model="HCX-007", thinking={"effort": "none"}, max_completion_tokens=256
    )

    model_with_structure = llm.with_structured_output(SimpleTest, method="json_schema")
    result = await model_with_structure.ainvoke(
        [("human", "Generate a test message. Set success to true.")]
    )
    assert isinstance(result, SimpleTest)
    assert len(result.message) > 0
    assert isinstance(result.success, bool)


@pytest.mark.skip(reason="structured_output model known issue")
async def test_astream_structured_output() -> None:
    """Test astream structured output from ChatClovaX."""
    llm = ChatClovaX(
        model="HCX-007", thinking={"effort": "none"}, max_completion_tokens=256
    )

    model_with_structure = llm.with_structured_output(SimpleTest, method="json_schema")

    async for token in model_with_structure.astream(
        [("human", "Generate a test message. Set success to true.")]
    ):
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
