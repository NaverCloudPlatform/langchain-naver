from __future__ import annotations

from operator import itemgetter
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    cast,
    overload,
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
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import (
    BaseChatOpenAI,
    _AllReturnType,
    _convert_message_to_dict,
    _DictOrPydantic,
    _DictOrPydanticClass,
    _is_pydantic_class,
)
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self


class ChatClovaX(BaseChatOpenAI):
    """ChatClovaX chat model.

    To use, you should have the environment variable `NCP_CLOVASTUDIO_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_naver import ChatClovaX

            model = ChatClovaX()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"ncp_clovastudio_api_key": "NCP_CLOVASTUDIO_API_KEY"}

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
    ncp_clovastudio_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "NCP_CLOVASTUDIO_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `NCP_CLOVASTUDIO_API_KEY`."
            ),
        ),
        alias="api_key",
    )
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""
    naver_api_base: Optional[str] = Field(
        default_factory=from_env(
            "NCP_CLOVASTUDIO_API_BASE_URL", default="https://clovastudio.stream.ntruss.com/v1/openai"
        ),
        alias="base_url",
    )
    """Base URL path for API requests, leave blank if not using a proxy or service 
    emulator."""
    openai_api_key: Optional[SecretStr] = Field(default=None)
    """openai api key is not supported for naver. use `ncp_clovastudio_api_key` instead."""
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
                self.ncp_clovastudio_api_key.get_secret_value()
                if self.ncp_clovastudio_api_key
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

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text."""
        tokenizer = self._get_tokenizer()
        encode = tokenizer.encode(text, add_special_tokens=False)
        return encode.ids

    def get_num_tokens_from_messages(
        self, messages: List[BaseMessage], tools: Sequence[Any] | None = None
    ) -> int:
        """Calculate num tokens for solar model."""
        tokenizer = self._get_tokenizer()
        tokens_per_message = 5  # <|im_start|>{role}\n{message}<|im_end|>
        tokens_prefix = 1  # <|startoftext|>
        tokens_suffix = 3  # <|im_start|>assistant\n

        num_tokens = 0

        num_tokens += tokens_prefix

        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(
                    tokenizer.encode(str(value), add_special_tokens=False)
                )
        # every reply is primed with <|im_start|>assistant
        num_tokens += tokens_suffix
        return num_tokens

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
        using_doc_parsing_model = self._using_doc_parsing_model(kwargs)

        if using_doc_parsing_model:
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(HumanMessage(document_contents))

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

    # TODO: Fix typing.
    @overload  # type: ignore[override]
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]:
        ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec or be a valid JSON schema
                with top level 'title' and 'description' keys specified.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_naver import ChatClovaX
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatClovaX(model="HCX-003", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_naver import ChatClovaX
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatClovaX(model="HCX-003", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_naver import ChatClovaX
                from langchain_core.utils.function_calling import convert_to_openai_tool
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatClovaX(model="HCX-003", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if schema is None:
            raise ValueError("schema must be specified. Received None.")
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        llm = self.bind_tools([schema], tool_choice=tool_name)
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast(type, schema)], first_tool_only=True
            )
        else:
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name, first_tool_only=True
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser
