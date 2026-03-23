"""Global configuration for StoryWeaver — hybrid NLU + API-based NLG.

StoryWeaver 项目全局配置文件。
支持从 .env 文件自动读取环境变量，提供统一的配置管理。
"""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

from pydantic import Field


class Settings(BaseSettings):
    """Centralised settings read from .env file automatically.
    
    集中式配置类，通过 Pydantic 自动从 .env 文件读取配置。
    所有模块共享同一个 settings 单例实例。
    """

    # ── Paths ──────────────────────────────────────────────
    # 路径配置
    PROJECT_ROOT: Path = Path(__file__).parent  # 项目根目录
    DATA_DIR: Path = Path(__file__).parent / "data"  # 数据目录

    # ── OpenAI / LLM API ──────────────────────────────────
    # OpenAI / LLM API 配置
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")  # API 密钥
    OPENAI_BASE_URL: str = Field(default="", description="OpenAI-compatible API base URL (e.g. https://your-server.com/v1)")  # API 基础 URL，支持兼容 OpenAI 的第三方服务
    OPENAI_MODEL: str = "gpt-4o-mini"  # 使用的模型名称
    OPENAI_MAX_TOKENS: int = 1024  # 最大生成 token 数
    OPENAI_TEMPERATURE: float = 0.85  # 温度参数，控制生成随机性（0-1）
    OPENAI_TOP_P: float = 0.95  # Top-p 采样参数

    # ── NLU Config ────────────────────────────────────────
    # 自然语言理解（NLU）配置
    INTENT_MODEL_NAME: str = "distilbert-base-uncased"  # 意图分类模型名称
    INTENT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "intent_classifier"  # 意图分类模型路径
    INTENT_MAX_LENGTH: int = 128  # 意图分类最大 token 长度
    INTENT_CPU_BATCH_SIZE: int = 8  # CPU 推理批次大小
    INTENT_LABELS: List[str] = [  # 意图标签列表（8 类）
        "action", "dialogue", "explore", "use_item",
        "ask_info", "rest", "trade", "other",
    ]
    SPACY_MODEL: str = "en_core_web_sm"  # spaCy NER 模型

    # ── NLG Config ────────────────────────────────────────
    # 自然语言生成（NLG）配置
    NUM_OPTIONS: int = 3  # 每次生成的玩家选项数量

    # ── Knowledge Graph Config ────────────────────────────
    # 知识图谱配置
    KG_MAX_NODES: int = 200  # 知识图谱最大节点数
    KG_ENTITY_TYPES: List[str] = ["person", "location", "item", "creature", "event"]  # 实体类型
    KG_RELATION_TYPES: List[str] = [  # 关系类型
        "located_at", "possesses", "ally_of", "enemy_of",
        "knows", "part_of", "caused_by", "has_attribute",
        "causes", "prevents", "enables", "follows",
    ]

    # ── KG Strategy Config ────────────────────────────────
    # 知识图谱策略配置
    KG_CONFLICT_RESOLUTION: str = "llm_arbitrate"  # 冲突解决策略
    # 可选值: "keep_latest"（保留最新）| "llm_arbitrate"（LLM 仲裁）

    KG_EXTRACTION_MODE: str = "dual_extract"  # 关系提取模式
    # 可选值: "story_only"（仅故事文本）| "dual_extract"（双重提取：玩家输入+故事文本）

    KG_IMPORTANCE_MODE: str = "composite"  # 重要性计算模式
    # 可选值: "degree_only"（仅度数）| "composite"（全量综合）| "incremental"（增量综合）

    KG_SUMMARY_MODE: str = "layered"  # 知识图谱摘要模式
    # 可选值: "flat"（扁平）| "layered"（分层：按重要性分层展示）

    # ── KG Tuning Params ──────────────────────────────────
    # 知识图谱调优参数
    KG_IMPORTANCE_DECAY_FACTOR: float = 0.95  # 重要性衰减因子（每回合乘以此值）
    KG_RELATION_DECAY_FACTOR: float = 0.90  # 关系强度衰减因子
    KG_RELATION_MIN_CONFIDENCE: float = 0.2  # 关系最小置信度阈值
    KG_IMPORTANCE_MENTION_BOOST: float = 0.15  # 提及次数对重要性的提升
    KG_IMPORTANCE_PLAYER_BOOST: float = 0.3  # 玩家提及对重要性的提升
    KG_MAX_TIMELINE_ENTRIES: int = 5  # 时间线最大条目数
    KG_DECAY_CADENCE: int = 1  # 每 N 回合执行一次关系衰减
    KG_INCREMENTAL_FULL_RECALC_INTERVAL: int = 10  # 增量模式每 N 回合执行一次全量重算

    # ── Safe rollback toggles ─────────────────────────────
    KG_ENABLE_INCREMENTAL_IMPORTANCE: bool = True  # 允许使用增量重要性计算
    KG_ENABLE_SUMMARY_CACHE: bool = True  # 允许在单回合缓存 KG 摘要

    # ── Game Config ───────────────────────────────────────
    # 游戏配置
    NARRATIVE_HISTORY_WINDOW: int = 6  # 叙事历史窗口大小（保留最近 N 条记录）
    MAX_CONTEXT_TOKENS: int = 512  # 最大上下文 token 数

    # ── KG Persistence ───────────────────────────────────
    # 知识图谱持久化配置
    KG_SAVE_DIR: Path = PROJECT_ROOT / "saves"  # 保存目录
    KG_AUTO_SAVE: bool = True  # 是否自动保存
    KG_SNAPSHOT_INTERVAL: int = 5  # 每 N 回合保存一次快照

    # ── Streamlit ─────────────────────────────────────────
    # Streamlit 配置
    STREAMLIT_PORT: int = 7860  # Streamlit 服务端口

    # Pydantic 模型配置
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# 单例 settings 实例，供所有模块使用
settings = Settings()
