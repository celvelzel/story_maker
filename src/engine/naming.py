"""Semantic archive naming helpers for story saves.

生成语义化存档文件名的工具模块。

The main entry point is :func:`generate_archive_name`, which asks the shared
LLM client for a short English summary of the story opening and then formats
the archive name as ``{summary}_{model}_{timestamp}.json``.

Examples:
    >>> name = generate_archive_name(
    ...     "You enter a magical forest. Creatures surround you.",
    ...     "gpt-4o-mini",
    ...     "fantasy",
    ... )
    >>> name.endswith(".json")
    True
    >>> "_gpt-4o-mini_" in name
    True
"""

from __future__ import annotations

import datetime
import logging
import re
import time

from config import settings
from src.utils.api_client import llm_client

logger = logging.getLogger(__name__)


def _sanitize_filename_component(value: str, fallback: str) -> str:
    """Normalize a string so it is safe for filename components."""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9-]", "", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or fallback


def _build_fallback_name(genre: str, timestamp: str) -> str:
    """Build the fallback archive filename."""
    genre_slug = _sanitize_filename_component(genre, "archive")
    return f"{genre_slug}_auto_{timestamp}.json"


def generate_archive_name(story_text: str, model_name: str, genre: str) -> str:
    """Generate a semantic archive filename for a saved story.

    The function tries to create a short English summary (3-5 words) from the
    first 500 characters of ``story_text`` using the shared LLM client. The
    final filename follows ``{summary}_{model}_{timestamp}.json``.

    If the LLM call or summary processing fails for any reason, the function
    falls back to ``{genre}_auto_{timestamp}.json``.

    Args:
        story_text: Story opening text used as the summary source.
        model_name: Model identifier used in the filename.
        genre: Genre used for fallback naming.

    Returns:
        A JSON archive filename.

    Examples:
        >>> generate_archive_name("Short story text", "gpt-4o-mini", "fantasy")
        '...'
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_slug = _sanitize_filename_component(settings.OPENAI_MODEL, settings.OPENAI_MODEL)

    try:
        excerpt = (story_text or "")[:500]
        messages = [
            {
                "role": "system",
                "content": "Generate a 3-5 word English summary of this story opening. Return ONLY the summary, no explanation.",
            },
            {"role": "user", "content": excerpt},
        ]

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                summary_raw = llm_client.chat(
                    messages,
                    temperature=settings.OPENAI_TEMPERATURE,
                    max_tokens=50,
                )
                summary = _sanitize_filename_component(summary_raw, "summary")
                if not summary:
                    raise ValueError("Empty archive summary")
                summary = summary[:50].strip("-") or "summary"
                return f"{summary}_{model_slug}_{timestamp}.json"
            except Exception as exc:
                last_exc = exc
                wait = attempt
                logger.warning("Archive name generation attempt %d failed (%s). Retrying in %d seconds…", attempt, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"Archive name generation failed after 3 retries: {last_exc}")
    except Exception as exc:
        logger.warning("Falling back to genre-based archive name: %s", exc)
        return _build_fallback_name(genre, timestamp)
