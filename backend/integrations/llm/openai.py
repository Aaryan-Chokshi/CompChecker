from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

from openai import APIError, APITimeoutError, OpenAI
from pydantic import BaseModel, ValidationError

from backend.core.config import Settings
from backend.core.errors import ProviderError

ResultT = TypeVar("ResultT", bound=BaseModel)


class OpenAILLM:
    def __init__(self, settings: Settings, client_factory: Callable[..., Any] = OpenAI):
        self.settings = settings
        self._client_factory = client_factory
        self._client = None
        self._lock = RLock()

    @property
    def initialized(self) -> bool:
        return self._client is not None

    def _get_client(self):
        with self._lock:
            if self._client is None:
                key = self.settings.openai_api_key
                if key is None:
                    raise ProviderError(
                        "llm_not_configured",
                        "Set OPENAI_API_KEY in the backend environment.",
                        503,
                    )
                self._client = self._client_factory(
                    api_key=key.get_secret_value(),
                    timeout=self.settings.openai_timeout_seconds,
                    max_retries=0,
                )
            return self._client

    def generate_structured(
        self, instructions: str, user_input: str, response_model: type[ResultT]
    ) -> ResultT:
        """Provider primitive; retrieval and compliance orchestration belong to features."""
        if not instructions.strip() or not user_input.strip():
            raise ValueError("Instructions and input must be nonempty.")
        try:
            response = self._get_client().responses.parse(
                model=self.settings.openai_model,
                instructions=instructions,
                input=user_input,
                text_format=response_model,
                store=False,
            )
        except APITimeoutError as exc:
            raise ProviderError(
                "llm_timeout",
                "OpenAI timed out. Try again or adjust the backend timeout.",
                504,
            ) from exc
        except APIError as exc:
            raise ProviderError(
                "llm_api_error",
                "OpenAI request failed. Check credentials, quota, and connectivity.",
            ) from exc
        except ValidationError as exc:
            raise ProviderError(
                "llm_invalid_output", "OpenAI returned invalid structured output."
            ) from exc

        for output in response.output:
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) == "refusal":
                    raise ProviderError("llm_refusal", "OpenAI declined this request.", 422)
        if response.status != "completed":
            raise ProviderError("llm_incomplete", "OpenAI did not complete the response.")
        if not isinstance(response.output_parsed, response_model):
            raise ProviderError("llm_invalid_output", "OpenAI returned no valid structured result.")
        return response.output_parsed

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
