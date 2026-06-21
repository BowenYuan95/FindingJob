"""LM Studio HTTP adapter with one retry and response policy."""

import logging
import time
from typing import Any

import requests

from config import LMSTUDIO_BASE

logger = logging.getLogger(__name__)


class LMStudioClient:
    def __init__(self, base_url: str = LMSTUDIO_BASE, attempts: int = 3):
        self.base_url = base_url.rstrip("/")
        self.attempts = attempts

    def _post_json(
        self, endpoint: str, payload: dict[str, Any], *, timeout: int, operation: str,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/{endpoint.lstrip('/')}",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("response is not a JSON object")
                return data
            except Exception as e:
                last_error = e
                logger.warning(f"[{operation}] 第 {attempt}/{self.attempts} 次调用失败: {e}")
                if attempt < self.attempts:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"{operation} 连续失败: {last_error}") from last_error

    def loaded_models(self, timeout: int = 5) -> set[str]:
        response = requests.get(f"{self.base_url}/models", timeout=timeout)
        response.raise_for_status()
        return {
            str(item.get("id")) for item in response.json().get("data", [])
            if item.get("id")
        }

    def embeddings(self, texts: list[str], model: str) -> list[list[float]]:
        data = self._post_json(
            "embeddings",
            {"model": model, "input": [text[:6000] for text in texts]},
            timeout=120,
            operation="embed",
        ).get("data", [])
        if len(data) != len(texts):
            raise ValueError(f"embedding count mismatch: {len(data)} != {len(texts)}")
        return [item["embedding"] for item in data]

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout: int = 240,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return self._post_json(
            "chat/completions", payload, timeout=timeout, operation="llm"
        )


LM_CLIENT = LMStudioClient()
