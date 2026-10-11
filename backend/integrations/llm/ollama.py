"""Ollama structured output adapter shared by chunking and comparison."""

import json
from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

import httpx
from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

from backend.core.config import Settings
from backend.core.errors import ProviderError

ResultT = TypeVar("ResultT", bound=BaseModel)


class OllamaLLM:
    def __init__(self, settings: Settings, client_factory: Callable[..., Any] = Client):
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
                key = self.settings.ollama_api_key
                if key is None:
                    raise ProviderError(
                        "llm_not_configured",
                        "Set OLLAMA_API_KEY in the backend environment or .env file.",
                        503,
                    )
                self._client = self._client_factory(
                    host="https://ollama.com",
                    headers={"Authorization": f"Bearer {key.get_secret_value()}"},
                    timeout=self.settings.ollama_timeout_seconds,
                )
            return self._client

    def generate_structured(
        self, instructions: str, user_input: str, response_model: type[ResultT]
    ) -> ResultT:
        if not instructions.strip() or not user_input.strip():
            raise ValueError("Instructions and input must be nonempty.")

        schema = response_model.model_json_schema()
        schema_instruction = (
            f" Return only JSON matching this schema: {json.dumps(schema, separators=(',', ':'))}"
        )
        try:
            response = self._get_client().chat(
                model=self.settings.ollama_model,
                messages=[
                    {"role": "system", "content": instructions + schema_instruction},
                    {"role": "user", "content": user_input},
                ],
                format=schema,
                stream=False,
                think=False,
                options={"temperature": 0},
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "llm_timeout",
                "Ollama timed out. Try again or adjust OLLAMA_TIMEOUT_SECONDS.",
                504,
            ) from exc
        except ResponseError as exc:
            if exc.status_code == 401:
                raise ProviderError(
                    "llm_auth_failed",
                    "Ollama rejected OLLAMA_API_KEY. Set a valid key in the backend environment or .env file.",
                    503,
                ) from exc
            if exc.status_code == 402:
                raise ProviderError(
                    "llm_payment_required",
                    "The configured Ollama plan does not include this model.",
                    402,
                ) from exc
            if exc.status_code == 404:
                raise ProviderError(
                    "llm_model_not_found",
                    "Ollama could not find OLLAMA_MODEL. Check the configured model name.",
                    502,
                ) from exc
            if exc.status_code == 429:
                raise ProviderError(
                    "llm_rate_limited",
                    "Ollama usage limit reached. Retry after the limit resets.",
                    429,
                ) from exc
            raise ProviderError(
                "llm_api_error",
                "Ollama request failed. Check credentials, model access, usage limits, and connectivity.",
            ) from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise ProviderError(
                "llm_api_error", "Could not connect to the Ollama cloud API.", 502
            ) from exc

        content = response.message.content
        if not content:
            raise ProviderError("llm_invalid_output", "Ollama returned no structured result.")
        try:
            return response_model.model_validate_json(content)
        except (ValueError, ValidationError) as exc:
            raise ProviderError(
                "llm_invalid_output", "Ollama returned invalid structured output."
            ) from exc

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
