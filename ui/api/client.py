from typing import Any

import httpx

from ui.config import UISettings


class APIUnavailable(Exception):
    pass


class APIClient:
    def __init__(self, settings: UISettings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    def _get(self, path: str, params: dict | None = None) -> Any:
        try:
            # Request-local clients have a bounded lifetime and release connections.
            with httpx.Client(
                timeout=self.settings.api_timeout_seconds, transport=self.transport
            ) as client:
                response = client.get(f"{self.settings.api_base_url}{path}", params=params)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise APIUnavailable("The backend took too long to respond. Try refreshing.") from exc
        except httpx.HTTPStatusError as exc:
            raise APIUnavailable(
                f"The backend returned an error (HTTP {exc.response.status_code})."
            ) from exc
        except httpx.RequestError as exc:
            raise APIUnavailable(
                "The backend is unavailable. Start the API, then refresh."
            ) from exc
        except ValueError as exc:
            raise APIUnavailable("The backend returned an unexpected response.") from exc

    def liveness(self) -> dict:
        result = self._get("/health/live")
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise APIUnavailable("The backend returned an unexpected health response.")
        return result

    def dependencies(self) -> dict:
        result = self._get("/health/dependencies")
        keys = ("database", "embeddings", "vector_store", "llm")
        if not isinstance(result, dict) or any(not isinstance(result.get(k), dict) for k in keys):
            raise APIUnavailable("The backend returned an unexpected dependency response.")
        return result

    def documents(self, role: str | None = None) -> list[dict]:
        params = {"limit": 100}
        if role:
            params["role"] = role
        return self._records(self._get("/documents", params))

    def checks(self) -> list[dict]:
        return self._records(self._get("/checks", {"limit": 100}))

    def _post(self, path: str, **kwargs) -> dict:
        try:
            with httpx.Client(
                timeout=self.settings.workflow_timeout_seconds, transport=self.transport
            ) as client:
                response = client.post(f"{self.settings.api_base_url}{path}", **kwargs)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Invalid response")
                return result
        except httpx.HTTPStatusError as exc:
            try:
                message = exc.response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = f"The backend returned HTTP {exc.response.status_code}."
            raise APIUnavailable(str(message)) from exc
        except httpx.TimeoutException as exc:
            raise APIUnavailable(
                "Processing took longer than the UI timeout. Refresh the document library or history to check its status before retrying."
            ) from exc
        except httpx.RequestError as exc:
            raise APIUnavailable("The backend is unavailable. Start the API, then retry.") from exc
        except ValueError as exc:
            raise APIUnavailable("The backend returned an unexpected response.") from exc

    def upload(self, filename: str, content: bytes, role: str):
        return self._post("/documents", files={"file": (filename, content)}, data={"role": role})

    def create_check(self, document_id: str, policy_ids: list[str], specifications: list[str]):
        return self._post(
            "/checks",
            json={
                "document_id": document_id,
                "policy_ids": policy_ids,
                "specifications": specifications,
            },
        )

    def clarification_url(self, check_id: str, format: str):
        return f"{self.settings.api_base_url}/checks/{check_id}/clarifications?format={format}"

    @staticmethod
    def _records(result: Any) -> list[dict]:
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise APIUnavailable("The backend returned an unexpected list response.")
        return result
