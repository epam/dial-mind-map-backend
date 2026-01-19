"""Partially adapts to StatGPT RAG evaluation. Contains StatGPT classes and functions"""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from io import BytesIO
from typing import (
    Any,
    AsyncIterator,
    Generator,
    Generic,
    Iterable,
    Literal,
    NamedTuple,
    TypeVar,
)

import httpx
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_token_usage_context_var: ContextVar = ContextVar("token_usage_context")
logger = logging.getLogger(__name__)
HTTP_METHOD_TYPE = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


class Pricing(BaseModel):
    unit: str
    prompt: float
    completion: float


class TokenUsageBase(BaseModel):
    deployment: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenUsageItem(TokenUsageBase):

    @property
    def id(self) -> str:
        return f"{self.deployment}_{self.model}"

    def __add__(self, other) -> "TokenUsageItem":
        if not isinstance(other, TokenUsageItem):
            return NotImplemented

        if self.id != other.id:
            raise ValueError("Cannot add TokenUsageItem with different id")

        return TokenUsageItem(
            deployment=self.deployment,
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class TokenUsagePricedItem(TokenUsageBase):
    costs: float | None = Field(
        ge=0, description="The total cost of the token usage"
    )


class TokenUsageCostCalculator:
    def __init__(self, model_to_pricing_map: dict) -> None:
        self._model_to_pricing_map = model_to_pricing_map

    def get_token_usage_with_costs(
        self, token_usage: list[TokenUsageItem]
    ) -> list[TokenUsagePricedItem]:
        return [
            TokenUsagePricedItem(
                deployment=item.deployment,
                model=item.model,
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                costs=self._calculate_usage_cost(item),
            )
            for item in token_usage
        ]

    def _calculate_usage_cost(self, item: TokenUsageItem) -> float | None:
        """Calculate the cost of the token usage."""
        if model_pricing := self._model_to_pricing_map.get(item.model):
            return (
                item.prompt_tokens * model_pricing.prompt
                + item.completion_tokens * model_pricing.completion
            )
        return None


class DialSettings(BaseSettings):
    """
    DIAL Core API connection settings
    """

    model_config = SettingsConfigDict(env_prefix="DIAL_")

    url: str = Field(
        default="http://localhost:8080",
        description="URL of the DIAL Core API where this app is deployed",
    )

    api_key: SecretStr = Field(
        default=SecretStr(""), description="API key for the DIAL Core API"
    )


class TokenUsageManager:
    def __init__(self):
        self._usage = {}

    def add_usage(self, item: TokenUsageItem):
        if item.id not in self._usage:
            self._usage[item.id] = item
        else:
            self._usage[item.id] += item

    def get_usage(self) -> list[TokenUsageItem]:
        return list(self._usage.values())


def encode_url_characters(url: str) -> str:
    mapping = {
        "[": "%5B",
        "]": "%5D",
    }
    for char, replacement in mapping.items():
        url = url.replace(char, replacement)
    return url


def get_token_usage_manager() -> TokenUsageManager:
    return _token_usage_context_var.get()


@contextmanager
def token_usage_context() -> Generator[TokenUsageManager, None, None]:
    token_usage = _token_usage_context_var.set(TokenUsageManager())
    try:
        yield token_usage.var.get()
    finally:
        _token_usage_context_var.reset(token_usage)


class TokenUsageByModelsCallback(AsyncCallbackHandler):
    """Callback to track token usage across different models."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # type: ignore[override]
        deployment_id = None

        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                else:
                    usage_metadata = None

                if generation.generation_info:
                    deployment_id = generation.generation_info.get("model_name")
            except AttributeError:
                usage_metadata = None
        else:
            usage_metadata = None

        if usage_metadata:
            completion_tokens = usage_metadata["output_tokens"]
            prompt_tokens = usage_metadata["input_tokens"]
        else:
            if response.llm_output is None:
                return None
            if "token_usage" not in response.llm_output:
                return None
            # compute tokens and cost for this request
            # noinspection PyTypeHints
            token_usage = response.llm_output["token_usage"]
            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)

        if not deployment_id and response.llm_output:
            deployment_id = response.llm_output.get("model_name")

        if not deployment_id:
            deployment_id = "unknown"

        logger.info(
            f"Token usage for model {deployment_id!r}:"
            f" prompt_tokens={prompt_tokens!r}, completion_tokens={completion_tokens!r}"
        )

        token_usage_manager = get_token_usage_manager()
        token_usage_manager.add_usage(
            TokenUsageItem(
                deployment=deployment_id,
                model=deployment_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

        return None


class DialCore:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

        # Each instance of `DialCore` has its own api_key inside the client,
        # so we can cache the bucket
        self._bucket_id: str | None = None

    async def call_custom_endpoint(
        self, endpoint: str, method: HTTP_METHOD_TYPE = "GET", **kwargs
    ) -> dict:
        response = await self._client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get_user_info(self) -> dict[str, Any]:
        """Retrieve information about a user."""
        response = await self._client.get("/v1/user/info")
        response.raise_for_status()
        return response.json()

    async def get_models(self) -> dict[str, Any]:
        response = await self._client.get("/openai/models")
        response.raise_for_status()
        return response.json()

    async def get_model_by(self, name: str) -> dict[str, Any]:
        response = await self._client.get(f"/openai/models/{name}")
        response.raise_for_status()
        return response.json()

    async def get_bucket_json(self) -> dict[str, str]:
        response = await self._client.get("/v1/bucket")
        response.raise_for_status()
        return response.json()

    async def load_bucket(self) -> str:
        bucket_json = await self.get_bucket_json()
        if "appdata" in bucket_json:
            return bucket_json["appdata"]
        elif "bucket" in bucket_json:
            return bucket_json["bucket"]
        else:
            raise ValueError("No appdata or bucket found")

    async def get_bucket(self, refresh: bool = False) -> str:
        """Get the bucket ID from cache or load it from the API.
        Use `refresh=True` to force loading from the API.
        """

        if self._bucket_id is None or refresh:
            self._bucket_id = await self.load_bucket()

        return self._bucket_id

    async def get_file(self, url: str) -> bytes:
        """Get the file content from the specified URL.

        Args:
            url: The value of the `url` filed returned by the DIAL API.

        Returns:
            The file content as bytes.
        """

        response = await self._client.get(f"/v1/{url}")
        response.raise_for_status()
        return response.content

    async def get_file_by_path(
        self, path: str, *, bucket: str | None = None
    ) -> tuple[bytes, str]:
        """
        Get the file content from the specified path.
        :param path: path to the file
        :param bucket: bucket to use
        :return: tuple of file content and content type
        """
        if not bucket:
            bucket = await self.get_bucket()

        response = await self._client.get(f"/v1/files/{bucket}/{path}")
        response.raise_for_status()
        return response.content, response.headers["Content-Type"]

    async def delete_file(self, url: str) -> None:
        """Delete the file at the specified URL.

        Args:
            url: The value of the `url` filed returned by the DIAL API.
        """
        response = await self._client.delete(f"/v1/{url}")
        response.raise_for_status()

    async def put_file(
        self,
        name: str,
        mime_type: str,
        content: BytesIO | bytes,
        *,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        if not bucket:
            bucket = await self.get_bucket()

        response = await self._client.put(
            f"/v1/files/{bucket}/{name}",
            files={name: (name, content, mime_type)},
        )
        response.raise_for_status()
        return response.json()

    async def put_local_file(
        self, name: str, path: str, *, bucket: str | None = None
    ) -> dict[str, Any]:
        """Put file from local drive."""

        if not bucket:
            bucket = await self.get_bucket()

        response = await self._client.put(
            f"/v1/files/{bucket}/{name}",
            files={name: open(path, "rb")},
        )
        response.raise_for_status()
        return response.json()

    async def get_file_metadata(
        self,
        path: str,
        *,
        token: str | None = None,
        limit: int = 100,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        """Call this endpoint to retrieve metadata for a file or folder at the specified path.
        If the path is a folder, it must end with a "/".

        If it is called for a folder, there can be optional `nextToken` field in the response to
        be used to request next items if present.

        Args:
            path: The path of the file or folder.
            token: The token from the previous request to request next items.
            limit: Limit on the number of items in the response.
            bucket: The bucket to use. If not provided, it will be fetched from the API.

        """

        if not bucket:
            bucket = await self.get_bucket()

        params: dict[str, Any] = {"limit": limit}
        if token:
            params["token"] = token

        url = f"/v1/metadata/files/{bucket}/{path}"
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def put_conversation(
        self, path: str, json_data: dict, *, bucket: str | None = None
    ) -> dict[str, Any]:
        """Method to add a conversation to the specified bucket and path."""

        if not bucket:
            bucket = await self.get_bucket()

        path = encode_url_characters(path)
        response = await self._client.put(
            url=f"/v1/conversations/{bucket}/{path}",
            json=json_data,
        )
        response.raise_for_status()
        return response.json()

    async def create_publication_request(
        self, json_data: dict
    ) -> dict[str, Any]:
        """Method to create a publish or unpublish request."""

        response = await self._client.post(
            url="/v1/ops/publication/create",
            json=json_data,
        )
        response.raise_for_status()
        return response.json()


@asynccontextmanager
async def dial_core_factory(
    base_url: str, api_key: str | SecretStr
) -> AsyncIterator[DialCore]:
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Api-Key": api_key},
        timeout=600,
    ) as client:
        yield DialCore(client)


T = TypeVar("T")


class CacheItem(NamedTuple, Generic[T]):
    value: T
    expiry: float


class Cache(Generic[T]):

    def __init__(self, ttl: int = 3600):
        import time as _time

        self._cache: dict[str, CacheItem[T]] = {}
        self._ttl = ttl
        self.time = _time

    def set(self, key: str, value: T) -> None:
        expiry = self.time.time() + self._ttl
        self._cache[key] = CacheItem(value=value, expiry=expiry)

    def get(self, key: str, default: T | None = None) -> T | None:
        if key in self._cache:
            item = self._cache[key]
            if self.time.time() < item.expiry:
                return item.value
            else:
                self._remove_expired_item(key)
        return default

    def clear(self) -> None:
        """Clear all items from the cache"""
        self._cache.clear()

    def cleanup(self) -> None:
        """Remove all expired items from the cache"""
        current_time = self.time.time()
        expired_keys = [
            key
            for key, item in self._cache.items()
            if current_time >= item.expiry
        ]
        for key in expired_keys:
            self._remove_expired_item(key)

    def _remove_expired_item(self, key: str) -> None:
        self._cache.pop(key, None)


_CACHE: Cache[Pricing] = Cache(ttl=24 * 3600)  # 24 hours


class ModelPricingGetter:

    def __init__(self, dial_core: DialCore):
        self._dial_core = dial_core

    async def get_model_pricing(self, model: str) -> Pricing | None:
        if pricing := _CACHE.get(model):
            return pricing

        if pricing := await self._load_pricing(model):
            _CACHE.set(model, pricing)
            return pricing

        return None

    async def _load_pricing(self, model: str) -> Pricing | None:
        try:
            model_data = await self._dial_core.get_model_by(name=model)
        except Exception as e:
            logger.error(f"Failed to fetch model data for model {model}: {e}")
            return None

        if "pricing" not in model_data:
            return None

        try:
            return Pricing.model_validate(model_data["pricing"])
        except ValidationError as e:
            logger.info(f"{model_data=}")
            logger.error(f"Failed to validate pricing for model {model}: {e}")
            return None


dial_settings = DialSettings()


class AuthContext(ABC):
    """Authentication context for data access."""

    @property
    @abstractmethod
    def is_system(self) -> bool:
        """Indicates if the context is for a system user."""

    @property
    @abstractmethod
    def dial_access_token(self) -> str | None:
        pass

    @property
    @abstractmethod
    def api_key(self) -> str:
        """DIAL API key for the request."""


class ModelPricingAuthContext(AuthContext):

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        return None

    @property
    def api_key(self) -> str:
        return dial_settings.api_key.get_secret_value()


async def _load_pricing(models: Iterable[str]) -> dict[str, Pricing]:
    async with dial_core_factory(
        base_url=dial_settings.url, api_key=ModelPricingAuthContext().api_key
    ) as dial_core:
        getter = ModelPricingGetter(dial_core)

        res = {}
        for model in models:
            if pricing := await getter.get_model_pricing(model):
                res[model] = pricing
    return res


async def calc_token_usage_costs(
    token_usage_manager: TokenUsageManager,
) -> list[TokenUsagePricedItem]:
    usage = token_usage_manager.get_usage()

    models = {item.model for item in usage}
    models_pricing = await _load_pricing(models)

    priced_usage = TokenUsageCostCalculator(
        models_pricing
    ).get_token_usage_with_costs(usage)
    return priced_usage
