"""Global configuration for StoryWeaver — hybrid NLU + API-based NLG."""
from pathlib import Path
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:  # graceful fallback
    from pydantic import BaseSettings  # type: ignore[no-redef]

from pydantic import Field


class Settings(BaseSettings):
    """Centralised settings read from .env file automatically."""

    # ── Paths ──────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = Path(__file__).parent / "data"

    # ── OpenAI / LLM API ──────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_BASE_URL: str = Field(default="", description="OpenAI-compatible API base URL (e.g. https://your-server.com/v1)")
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 1024
    OPENAI_TEMPERATURE: float = 0.85
    OPENAI_TOP_P: float = 0.95

    # ── NLU Config ────────────────────────────────────────
    INTENT_MODEL_NAME: str = "roberta-base"
    INTENT_LABELS: List[str] = [
        "action", "dialogue", "explore", "use_item",
        "ask_info", "rest", "trade", "other",
    ]
    SPACY_MODEL: str = "en_core_web_sm"

    # ── NLG Config ────────────────────────────────────────
    NUM_OPTIONS: int = 3

    # ── Knowledge Graph Config ────────────────────────────
    KG_MAX_NODES: int = 200
    KG_ENTITY_TYPES: List[str] = ["person", "location", "item", "creature", "event"]
    KG_RELATION_TYPES: List[str] = [
        "located_at", "possesses", "ally_of", "enemy_of",
        "knows", "part_of", "caused_by", "has_attribute",
    ]

    # ── Game Config ───────────────────────────────────────
    NARRATIVE_HISTORY_WINDOW: int = 6
    MAX_CONTEXT_TOKENS: int = 512

    # ── Gradio ────────────────────────────────────────────
    GRADIO_PORT: int = 7860

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton settings instance used by every module
settings = Settings()
