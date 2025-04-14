from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
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
        return {"ncp_clovastudio_api_key": "CLOVASTUDIO_API_KEY"}

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

    model_name: str = Field(default="HCX-003", alias="model")
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
    """openai api key is not supported for naver. 
    use `ncp_clovastudio_api_key` instead."""
    openai_api_base: Optional[str] = Field(default=None)
    """openai api base is not supported for naver. use `naver_api_base` instead."""
    openai_organization: Optional[str] = Field(default=None)
    """openai organization is not supported for naver."""
    tiktoken_model_name: Optional[str] = None
    """tiktoken is not supported for naver."""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: dict = {
            "api_key": (
                self.api_key.get_secret_value()
                if self.api_key
                else None
            ),
            "base_url": self.naver_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

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
        response = self.client.create(**payload)
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
        response = await self.async_client.create(**payload)
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
