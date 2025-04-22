from __future__ import annotations

import json
import uuid
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import (
    BaseChatOpenAI,
    _convert_message_to_dict,
)
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_naver.const import USER_AGENT


def _convert_tool_calls_argument_base_message(message: BaseMessage) -> None:
    if message.role == "assistant":  # type: ignore[attr-defined]
        if (tool_calls := message.tool_calls) is not None:  # type: ignore[attr-defined]
            for raw_tool_call in tool_calls:
                if (function_dict := raw_tool_call.function) is not None:
                    if (arguments := function_dict.arguments) is not None:
                        if isinstance(arguments, dict):
                            function_dict.arguments = json.dumps(arguments)


def _convert_tool_calls_argument_dict(message: dict) -> None:
    if message.get("role") == "assistant":
        if (tool_calls := message.get("tool_calls")) is not None:
            for raw_tool_call in tool_calls:
                if (function_dict := raw_tool_call.get("function")) is not None:
                    if (arguments := function_dict.get("arguments")) is not None:
                        if isinstance(arguments, dict):
                            function_dict["arguments"] = json.dumps(arguments)


def _convert_response_tool_calls(response: Union[dict, openai.BaseModel]) -> None:
    if isinstance(response, openai.BaseModel):
        for choice in response.choices:  # type: ignore[attr-defined]
            if (message := choice.message) is not None:
                _convert_tool_calls_argument_base_message(message)
    elif isinstance(response, dict):
        for choice in response["choices"]:
            if (message := choice["message"]) is not None:
                _convert_tool_calls_argument_dict(message)


def _convert_payload_messages(payload: dict) -> None:
    if (messages := payload.get("messages")) is None:
        return

    for message in messages:
        if message.get("content") is None:
            message["content"] = ""


class ChatClovaX(BaseChatOpenAI):
    """ChatClovaX chat model.

    To use, you should have the environment variable `CLOVASTUDIO_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_naver import ChatClovaX

            model = ChatClovaX()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "CLOVASTUDIO_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "naver"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.naver_api_base:
            attributes["naver_api_base"] = self.naver_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-naver"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "naver"
        return params

    model_name: str = Field(default="HCX-005", alias="model")
    """Model name to use."""
    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "CLOVASTUDIO_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `CLOVASTUDIO_API_KEY`."
            ),
        ),
    )
    """Automatically inferred from env are `CLOVASTUDIO_API_KEY` if not provided."""
    naver_api_base: Optional[str] = Field(
        default_factory=from_env(
            "CLOVASTUDIO_API_BASE_URL",
            default="https://clovastudio.stream.ntruss.com/v1/openai",
        ),
        alias="base_url",
    )
    """Base URL path for API requests, leave blank if not using a proxy or service 
    emulator."""
    openai_api_key: Optional[SecretStr] = Field(default=None)
    default_headers: Union[Mapping[str, str], None] = {
        "User-Agent": USER_AGENT,
    }
    """openai api key is not supported for naver. 
    use `api_key` instead."""
    openai_api_base: Optional[str] = Field(default=None)
    """openai api base is not supported for naver. use `naver_api_base` instead."""
    openai_organization: Optional[str] = Field(default=None)
    """openai organization is not supported for naver."""
    tiktoken_model_name: Optional[str] = None
    """tiktoken is not supported for naver."""
    top_k: Optional[int] = Field(ge=0, le=128, default=None)
    """생성 토큰 후보군에서 확률이 높은 K개를 후보로 지정하여 샘플링."""
    repetition_penalty: Optional[float] = Field(gt=0.0, le=2.0, default=None)
    """같은 토큰을 생성하는 것에 대한 패널티 정도(설정값이 높을수록 같은 결괏값을 
    반복 생성할 확률 감소). Chat Completion v3 API에서만 사용 가능."""
    repeat_penalty: Optional[float] = Field(gt=0.0, le=10, default=None)
    """같은 토큰을 생성하는 것에 대한 패널티 정도(설정값이 높을수록 같은 결괏값을 
    반복 생성할 확률 감소). Chat Completion API에서만 사용 가능."""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: dict = {
            "api_key": (self.api_key.get_secret_value() if self.api_key else None),
            "base_url": self.naver_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        self.extra_body = {}
        if self.top_k is not None:
            self.extra_body["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            self.extra_body["repetition_penalty"] = self.repetition_penalty
        if self.repeat_penalty is not None:
            self.extra_body["repeat_penalty"] = self.repeat_penalty

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return self

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        _convert_payload_messages(payload)
        response = self.client.create(
            **payload,
            extra_headers={"X-NCP-CLOVASTUDIO-REQUEST-ID": f"lcnv-{str(uuid.uuid4())}"},
        )
        _convert_response_tool_calls(response)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        _convert_payload_messages(payload)
        response = await self.async_client.create(
            **payload,
            extra_headers={"X-NCP-CLOVASTUDIO-REQUEST-ID": f"lcnv-{str(uuid.uuid4())}"},
        )
        _convert_response_tool_calls(response)
        return self._create_chat_result(response)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop
        return {
            "messages": [_convert_message_to_dict(m) for m in messages],
            **self._default_params,
            **kwargs,
        }
