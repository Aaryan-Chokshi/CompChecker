"""Groq structured output adapter shared by chunking and comparison."""

import json
from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

from groq import APIError, APITimeoutError, AuthenticationError, Groq, RateLimitError
from pydantic import BaseModel, ValidationError

from backend.core.config import Settings
from backend.core.errors import ProviderError

ResultT = TypeVar("ResultT", bound=BaseModel)


def _strict_schema(model: type[BaseModel]) -> dict:
    """Make Pydantic's schema satisfy Groq strict output requirements."""
    schema = model.model_json_schema()

    def normalize(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                normalize(value)
        elif isinstance(node, list):
            for value in node:
                normalize(value)

    normalize(schema)
    return schema


class GroqLLM:
    def __init__(self, settings: Settings, client_factory: Callable[..., Any] = Groq):
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
                key = self.settings.groq_api_key
                if key is None:
                    raise ProviderError(
                        "llm_not_configured",
                        "Set GROQ_API_KEY in the backend environment.",
                        503,
                    )
                self._client = self._client_factory(
                    api_key=key.get_secret_value(),
                    timeout=self.settings.groq_timeout_seconds,
                    max_retries=0,
                )
            return self._client

    def generate_structured(
        self, instructions: str, user_input: str, response_model: type[ResultT]
    ) -> ResultT:
        if not instructions.strip() or not user_input.strip():
            raise ValueError("Instructions and input must be nonempty.")
        try:
            completion = self._get_client().chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": _strict_schema(response_model),
                    },
                },
            )
        except APITimeoutError as exc:
            raise ProviderError(
                "llm_timeout", "Groq timed out. Try again or adjust GROQ_TIMEOUT_SECONDS.", 504
            ) from exc
        except AuthenticationError as exc:
            raise ProviderError(
                "llm_auth_failed",
                "Groq rejected GROQ_API_KEY. Set a valid key in the backend environment.",
                503,
            ) from exc
        except RateLimitError as exc:
            raise ProviderError(
                "llm_rate_limited", "Groq rate limit reached. Retry after the limit resets.", 429
            ) from exc
        except APIError as exc:
            raise ProviderError(
                "llm_api_error",
                "Groq request failed. Check credentials, model access, rate limits, and connectivity.",
            ) from exc

        if not completion.choices:
            raise ProviderError("llm_incomplete", "Groq returned no completion.")
        choice = completion.choices[0]
        if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
            raise ProviderError("llm_refusal", "Groq declined this request.", 422)
        if choice.finish_reason != "stop":
            raise ProviderError("llm_incomplete", "Groq did not complete the response.")
        if not choice.message.content:
            raise ProviderError("llm_invalid_output", "Groq returned no structured result.")
        try:
            return response_model.model_validate(json.loads(choice.message.content))
        except (ValueError, ValidationError) as exc:
            raise ProviderError(
                "llm_invalid_output", "Groq returned invalid structured output."
            ) from exc

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
