"""Singleton OpenAI wrapper with retry, JSON mode, and cost tracking."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pricing per 1 M tokens for gpt-4o-mini (as of 2025-06)
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class LLMClient:
    """Singleton OpenAI chat-completion wrapper.

    * ``chat()``       → plain text response
    * ``chat_json()``  → JSON-mode response parsed to ``dict``
    * Automatic retry with exponential back-off (3 attempts)
    * Per-session cost tracking (tokens + USD)
    """

    _instance: Optional["LLMClient"] = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        from config import settings

        self._settings = settings
        self._client: Any = None
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._initialised = True

    # ── lazy OpenAI client ────────────────────────────────
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._settings.OPENAI_API_KEY)
            except Exception as exc:
                logger.error("Failed to create OpenAI client: %s", exc)
                raise
        return self._client

    # ── public API ────────────────────────────────────────
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant message text.

        Retries up to 3 times with exponential back-off on transient errors.
        """
        temperature = temperature if temperature is not None else self._settings.OPENAI_TEMPERATURE
        max_tokens = max_tokens or self._settings.OPENAI_MAX_TOKENS

        kwargs: Dict[str, Any] = {
            "model": self._settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(**kwargs)
                # track usage
                usage = response.usage
                if usage:
                    self._total_input_tokens += usage.prompt_tokens
                    self._total_output_tokens += usage.completion_tokens
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("LLM call attempt %d failed (%s). Retrying in %ds…", attempt, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after 3 attempts: {last_exc}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Like ``chat()`` but returns the parsed JSON dict."""
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        return json.loads(raw)

    # ── cost tracking ─────────────────────────────────────
    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        pricing = _PRICING.get(self._settings.OPENAI_MODEL, _PRICING["gpt-4o-mini"])
        return (
            self._total_input_tokens * pricing["input"] / 1_000_000
            + self._total_output_tokens * pricing["output"] / 1_000_000
        )

    def reset_cost(self) -> None:
        self._total_input_tokens = 0
        self._total_output_tokens = 0


# Convenience module-level singleton
llm_client = LLMClient()
