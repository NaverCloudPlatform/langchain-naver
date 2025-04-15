from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

DEFAULT_EMBED_BATCH_SIZE = 10
MAX_EMBED_BATCH_SIZE = 100


class ClovaXEmbeddings(BaseModel, Embeddings):
    """ClovaXEmbeddings embedding model.

    To use, set the environment variable `CLOVASTUDIO_API_KEY` with your API key or
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_naver import ClovaXEmbeddings

            model = ClovaXEmbeddings(model='clir-emb-dolphin')
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = Field(default="clir-emb-dolphin")
    """NCP ClovaStudio embedding model name"""
    dimensions: Optional[int] = None
    """The number of dimensions the resulting output embeddings should have.
    
    Not yet supported. 
    """
    encoding_format: Literal["float", "base64"] = "float"
    """Not yet supported. """
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
    """Endpoint URL to use."""
    embedding_ctx_length: int = 4096
    """The maximum number of tokens to embed at once.
    
    Not yet supported.
    """
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    allowed_special: Union[Literal["all"], Set[str]] = set()
    """Not yet supported."""
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    """Not yet supported."""
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch.
    
    Not yet supported.
    """
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Upstage embedding API. Can be float, httpx.Timeout or
        None."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding.
    
    Not yet supported.
    """
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error.
    Defaults to not skipping.
    
    Not yet supported."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client. Only used for sync invocations. Must specify 
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify 
        http_client as well if you'd like a custom client for sync invocations."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""

        client_params: dict = {
            "api_key": (self.api_key.get_secret_value() if self.api_key else None),
            "base_url": self.naver_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).embeddings
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).embeddings
        return self

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        params: Dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        if self.encoding_format is not None:
            params["encoding_format"] = self.encoding_format
        return params

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using passage model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        response = self.client.create(input=text, **self._invocation_params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using passage model asynchronously.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            embedding = await self.aembed_query(text)
            embeddings.append(embedding)
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        response = await self.async_client.create(input=text, **self._invocation_params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]
