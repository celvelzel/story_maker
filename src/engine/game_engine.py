"""Main game engine orchestrator for StoryWeaver.

StoryWeaver 主游戏引擎，协调整个 NLU → NLG → KG 流水线。

每回合处理流程：
1. 共指消解（Coreference resolution）
2. 意图分类（Intent classification）
3. 实体提取（Entity extraction）
4. 故事生成（Story generation via LLM）
5. 知识图谱更新（KG update: 双重提取 + 状态更新 + 衰减 + 重要性刷新）
6. 冲突检测 + 解决（Conflict detection + resolution）
7. 选项生成（Option generation）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from src.engine.state import GameState
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.nlu.coreference import CoreferenceResolver
from src.nlu.sentiment_analyzer import SentimentAnalyzer
from src.nlg.story_generator import StoryGenerator
from src.nlg.option_generator import OptionGenerator, StoryOption
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.relation_extractor import (
    extract as kg_extract,
    extract_dual as kg_extract_dual,
)
from src.knowledge_graph.conflict_detector import ConflictDetector, get_resolver
from src.knowledge_graph.visualizer import render_kg_html
from src.utils.api_client import llm_client

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Container returned after every game turn.
    
    每个游戏回合的结果容器。
    包含：故事文本、玩家选项、NLU调试信息、知识图谱HTML、冲突列表。
    """
    story_text: str  # 故事文本
    options: List[StoryOption]  # 玩家选项列表
    nlu_debug: Dict = field(default_factory=dict)  # NLU 调试信息
    kg_html: str = ""  # 知识图谱可视化 HTML
    conflicts: List[str] = field(default_factory=list)  # 冲突描述列表


class GameEngine:
    """Coordinates the full NLU → NLG → KG pipeline.
    
    游戏引擎主类，协调完整的 NLU → NLG → KG 流水线。
    负责管理游戏状态、调用各子模块、处理回合逻辑。
    """

    def __init__(
        self,
        genre: str = "fantasy",
        intent_model_path: Optional[str] = None,
        auto_load_nlu: bool = True,
        conflict_resolution: Optional[str] = None,
        extraction_mode: Optional[str] = None,
        importance_mode: Optional[str] = None,
        summary_mode: Optional[str] = None,
    ):
        """
        初始化游戏引擎。
        
        参数:
            genre: 故事类型（如 "fantasy", "sci-fi"）
            intent_model_path: 意图分类模型路径（可选）
            auto_load_nlu: 是否自动加载 NLU 组件
            conflict_resolution: 冲突解决策略（可选，使用配置默认值）
            extraction_mode: 关系提取模式（可选）
            importance_mode: 重要性计算模式（可选）
            summary_mode: 摘要生成模式（可选）
        """
        self.genre = genre
        self.state = GameState()
        self.kg = KnowledgeGraph()

        # Strategy configuration (fallback to settings defaults)
        self.conflict_resolution = conflict_resolution or settings.KG_CONFLICT_RESOLUTION
        self.extraction_mode = extraction_mode or settings.KG_EXTRACTION_MODE
        self.importance_mode = importance_mode or settings.KG_IMPORTANCE_MODE
        self.summary_mode = summary_mode or settings.KG_SUMMARY_MODE

        logger.info(
            "[Engine][init] genre=%s | strategies: conflict=%s extraction=%s importance=%s summary=%s",
            genre, self.conflict_resolution, self.extraction_mode,
            self.importance_mode, self.summary_mode,
        )

        # Evaluation tracking（评估追踪）
        self.turn_conflict_counts: List[int] = []  # 记录每回合检测到的冲突数量

        # NLU 组件初始化
        self.coref = CoreferenceResolver()  # 共指消解器
        self.intent_clf = IntentClassifier(model_path=intent_model_path)  # 意图分类器
        self.entity_ext = EntityExtractor()  # 实体提取器
        self.sentiment = SentimentAnalyzer()  # 情感分析器
        # NLU 组件加载状态追踪
        self.nlu_status: Dict[str, object] = {
            "coref_loaded": False,
            "intent_model_loaded": False,
            "intent_backend": "rule_fallback",
            "entity_model_loaded": False,
            "sentiment_loaded": False,
        }

        if auto_load_nlu:
            self._load_nlu_components()
        else:
            logger.info("NLU auto-load disabled. Engine will run with lazy/fallback behavior.")

        # NLG 组件初始化
        self.story_gen = StoryGenerator()  # 故事生成器
        self.option_gen = OptionGenerator()  # 选项生成器

        # KG 辅助组件
        self.conflict_det = ConflictDetector(self.kg)  # 冲突检测器
        self.conflict_resolver = get_resolver(self.conflict_resolution)  # 冲突解决策略
        self._turn_cached_summary: Optional[str] = None

    def _current_kg_summary(self) -> str:
        """Per-turn KG summary cache to avoid repeated graph traversals."""
        if not settings.KG_ENABLE_SUMMARY_CACHE:
            return self.kg.to_summary()
        if self._turn_cached_summary is None:
            self._turn_cached_summary = self.kg.to_summary()
        return self._turn_cached_summary

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> TurnResult:
        """Begin a new adventure and return the opening turn.
        
        开始新游戏：重置状态，生成开场故事，初始化知识图谱，生成初始选项。
        返回第一个回合的结果。
        """
        logger.info("[Engine][start_game] Starting new game | genre=%s", self.genre)
        self.state = GameState()
        self.kg = KnowledgeGraph()
        self.conflict_det = ConflictDetector(self.kg)
        self.conflict_resolver = get_resolver(self.conflict_resolution)
        self.turn_conflict_counts = []

        story_text = self.story_gen.generate_opening(self.genre)
        self.state.add_narration(story_text)

        # Seed the KG from the opening
        self.kg.set_turn(self.state.turn_id)
        self._apply_kg_update(story_text, turn_id=self.state.turn_id)

        kg_summary = self.kg.to_summary()
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg.graph)

        logger.info("[Engine][start_game] Opening generated | KG: %d nodes, %d edges", self.kg.num_nodes, self.kg.num_edges)

        return TurnResult(
            story_text=story_text,
            options=options,
            kg_html=kg_html,
        )

    def process_turn(self, player_input: str) -> TurnResult:
        """Run the full pipeline for one player turn.
        
        处理玩家回合：执行完整的 NLU → NLG → KG 流水线。
        
        流程：
        1. 共指消解 - 解析代词引用
        2. 意图分类 - 识别玩家意图
        3. 情感分析 - 分析玩家情感
        4. 实体提取 - 提取游戏实体
        5. 故事生成 - 生成故事续写
        6. 知识图谱更新 - 更新世界状态
        7. 冲突检测 - 检测并解决矛盾
        8. 选项生成 - 生成玩家选项
        
        参数:
            player_input: 玩家输入文本
            
        返回:
            TurnResult: 包含故事文本、选项、调试信息等
        """
        logger.info(
            "[Engine][process_turn] === Turn %d START === | input='%s'",
            self.state.turn_id + 1, player_input[:60],
        )
        self._turn_cached_summary = None

        stage_metrics: Dict[str, Dict[str, float]] = {}

        def _stage_begin() -> tuple[float, Dict[str, float | int]]:
            return time.perf_counter(), llm_client.usage_snapshot()

        def _stage_end(stage_name: str, started_at: float, usage_before: Dict[str, float | int]) -> None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            usage_after = llm_client.usage_snapshot()
            usage_delta = llm_client.usage_delta(usage_before, usage_after)
            stage_metrics[stage_name] = {
                "elapsed_ms": elapsed_ms,
                "input_tokens": usage_delta["input_tokens"],
                "output_tokens": usage_delta["output_tokens"],
                "cost_usd": usage_delta["cost_usd"],
            }

        # ========== 1. 共指消解 ==========
        # 获取最近 4 条故事记录作为上下文
        t0, u0 = _stage_begin()
        recent_entries = self.state.story_history[-4:]
        recent_texts = [t["text"] for t in recent_entries]
        # 从知识图谱构建已知实体列表（用于实体类型感知的消解）
        known_entities = [
            {"text": data.get("name", key), "type": data.get("entity_type", "unknown")}
            for key, data in self.kg.graph.nodes(data=True)
        ]
        # 解析玩家输入中的代词（如 "it" → "dragon"）
        resolved = self.coref.resolve(player_input, recent_texts, known_entities=known_entities)
        logger.debug("[Engine][coref] 消解: '%s' → '%s'", player_input[:40], resolved[:40])
        _stage_end("coref", t0, u0)

        # ========== 2. 意图分类 ==========
        t0, u0 = _stage_begin()
        intent_result = self.intent_clf.predict(resolved)
        intent_raw = intent_result.get("intent", "other")
        intent = intent_raw if isinstance(intent_raw, str) else "other"
        logger.debug("[Engine][intent] 意图=%s 置信度=%.2f", intent, intent_result["confidence"])
        _stage_end("intent", t0, u0)

        # ========== 2b. 情感分析 ==========
        # 分析玩家输入的情感（用于调整叙事风格）
        t0, u0 = _stage_begin()
        emotion_result = self.sentiment.analyze(resolved)
        emotion_raw = emotion_result.get("emotion", "neutral")
        emotion = emotion_raw if isinstance(emotion_raw, str) else "neutral"
        logger.debug("[Engine][sentiment] 情感=%s 置信度=%.2f", emotion, emotion_result["confidence"])
        _stage_end("sentiment", t0, u0)

        # ========== 3. 实体提取 (NLU 层) ==========
        # 从解析后的文本中提取实体（使用知识图谱上下文辅助）
        t0, u0 = _stage_begin()
        kg_entity_names = list(self.kg.graph.nodes())
        entities = self.entity_ext.extract(resolved, known_entities=kg_entity_names)
        entity_names = [e["text"] for e in entities]
        logger.debug("[Engine][nlu_entities] 提取了 %d 个实体: %s", len(entities), entity_names)
        _stage_end("entity_extraction", t0, u0)

        # 记录玩家输入到游戏状态
        self.state.add_player_input(player_input)

        # ========== 4. 故事生成 ==========
        # 获取知识图谱摘要作为世界状态上下文
        t0, u0 = _stage_begin()
        kg_summary = self._current_kg_summary()
        # 获取最近 6 条历史记录
        history = self.state.recent_history(6)
        # 调用故事生成器续写故事
        story_text = self.story_gen.continue_story(
            player_input=resolved,     # 已消解的玩家输入
            intent=intent,            # 玩家意图
            kg_summary=kg_summary,     # 世界状态摘要
            history=history,           # 最近对话历史
            emotion=emotion,  # 玩家情感
        )
        # 将生成的故事添加到状态并推进回合
        self.state.add_narration(story_text)
        current_turn = self.state.turn_id
        _stage_end("story_generation", t0, u0)

        # ========== 5. 知识图谱更新 ==========
        t0, u0 = _stage_begin()
        self.kg.set_turn(current_turn)
        self._apply_kg_update(
            story_text=story_text,
            player_input=resolved,
            nlu_entities=entities,
            turn_id=current_turn,
            emotion=emotion,
        )
        self._turn_cached_summary = None
        _stage_end("kg_update", t0, u0)

        # ========== 6. 冲突检测 + 解决 ==========
        # 检测故事中的逻辑矛盾
        t0, u0 = _stage_begin()
        raw_conflicts = self.conflict_det.check_all(story_text)
        # 根据配置的策略解决冲突
        unresolved = self.conflict_resolver.resolve(raw_conflicts, self.kg)
        # 提取冲突描述用于显示
        conflict_descriptions = [c.get("description", str(c)) for c in unresolved]
        # 记录本回合的冲突数量（用于评估）
        self.turn_conflict_counts.append(len(unresolved))

        logger.info(
            "[Engine][conflicts] 检测=%d 已解决=%d 剩余=%d (策略: %s)",
            len(raw_conflicts),
            len(raw_conflicts) - len(unresolved),
            len(unresolved),
            self.conflict_resolution,
        )
        _stage_end("conflict_detection_resolution", t0, u0)

        # ========== 7. 选项生成 ==========
        # 更新知识图谱摘要（KG 更新后刷新）
        t0, u0 = _stage_begin()
        kg_summary = self._current_kg_summary()
        # 为玩家生成下一步可选的行动选项
        options = self.option_gen.generate(story_text, kg_summary)
        # 渲染知识图谱可视化
        kg_html = render_kg_html(self.kg.graph)
        _stage_end("options_and_render", t0, u0)

        nlu_debug = {
            "resolved_input": resolved,
            "intent": intent,
            "confidence": intent_result["confidence"],
            "entities": entities,
            "emotion": emotion,
            "emotion_confidence": emotion_result["confidence"],
            "emotion_scores": emotion_result.get("scores", {}),
            "intent_backend": self.nlu_status["intent_backend"],
            "intent_model_loaded": self.nlu_status["intent_model_loaded"],
            "coref_loaded": self.nlu_status["coref_loaded"],
            "entity_model_loaded": self.nlu_status["entity_model_loaded"],
            "sentiment_loaded": self.nlu_status["sentiment_loaded"],
            "stage_metrics": stage_metrics,
        }

        logger.info(
            "[Engine][process_turn] === Turn %d END === | KG: %d nodes, %d edges, %d conflicts",
            current_turn, self.kg.num_nodes, self.kg.num_edges, len(unresolved),
        )

        # Auto-save
        self._auto_save()
        self._turn_cached_summary = None

        return TurnResult(
            story_text=story_text,
            options=options,
            nlu_debug=nlu_debug,
            kg_html=kg_html,
            conflicts=conflict_descriptions,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @property
    def all_story_texts(self) -> List[str]:
        """Return all narrator texts from the session (for evaluation)."""
        return [
            e["text"] for e in self.state.story_history if e["role"] == "narrator"
        ]

    @property
    def kg_entity_names(self) -> List[str]:
        """Return the display names of all KG entities."""
        return [
            data.get("name", key)
            for key, data in self.kg.graph.nodes(data=True)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_nlu_components(self) -> None:
        """加载所有 NLU 子模块，失败时优雅降级。

        加载顺序：
        1. 共指消解器 (CoreferenceResolver)
        2. 意图分类器 (IntentClassifier)
        3. 实体提取器 (EntityExtractor)
        4. 情感分析器 (SentimentAnalyzer)

        如果某个模块加载失败，系统会使用规则回退策略继续运行，
        确保游戏不会因为某个 NLU 组件不可用而中断。
        """
        # 1. 加载共指消解器
        try:
            self.coref.load()
            self.nlu_status["coref_loaded"] = self.coref.model is not None
        except Exception as exc:
            logger.warning("共指消解器初始化失败: %s", exc)

        # 2. 加载意图分类器
        try:
            self.intent_clf.load()
            self.nlu_status["intent_model_loaded"] = (
                self.intent_clf.model is not None and self.intent_clf.tokenizer is not None
            )
            self.nlu_status["intent_backend"] = self.intent_clf.backend  # "distilbert" 或 "rule_fallback"
        except Exception as exc:
            logger.warning("意图分类器初始化失败: %s", exc)
            self.nlu_status["intent_backend"] = "rule_fallback"

        # 3. 加载实体提取器
        try:
            self.entity_ext.load()
            self.nlu_status["entity_model_loaded"] = self.entity_ext.nlp is not None
        except Exception as exc:
            logger.warning("实体提取器初始化失败: %s", exc)

        # 4. 加载情感分析器
        try:
            self.sentiment.load()
            self.nlu_status["sentiment_loaded"] = self.sentiment.model is not None
        except Exception as exc:
            logger.warning("情感分析器初始化失败: %s", exc)

        # 记录 NLU 组件加载状态
        logger.info(
            "NLU 组件加载状态: 共指消解=%s, 意图模型=%s, 意图后端=%s, 实体模型=%s",
            self.nlu_status["coref_loaded"],
            self.nlu_status["intent_model_loaded"],
            self.nlu_status["intent_backend"],
            self.nlu_status["entity_model_loaded"],
        )

    def _apply_kg_update(
        self,
        story_text: str,
        player_input: str = "",
        nlu_entities: Optional[List[Dict]] = None,
        turn_id: int = 0,
        emotion: Optional[str] = None,
    ) -> None:
        """Apply KG updates based on configured extraction mode.

        根据配置的提取模式应用知识图谱更新。

        步骤：
        1. 从故事文本（+可选玩家输入）提取实体和关系
        2. 添加/更新实体（带丰富属性）
        3. 添加/更新关系（带上下文和置信度）
        4. 从状态变更更新实体状态
        5. 刷新提及跟踪
        6. 应用时间衰减
        7. 重新计算重要性分数

        参数:
            story_text: 故事文本
            player_input: 玩家输入（可选）
            nlu_entities: NLU 层提取的实体列表（可选）
            turn_id: 当前回合 ID
            emotion: 情感标签（可选）
        """
        all_entities: List[Dict] = []
        all_relations: List[Dict] = []

        # ========== 步骤 1: LLM 关系提取 ==========
        try:
            if self.extraction_mode == "dual_extract" and player_input:
                # 双重提取模式：从玩家输入和故事文本同时提取
                existing_names = list(self.kg.graph.nodes())
                data = kg_extract_dual(player_input, story_text, existing_names)
                logger.debug(
                    "[Engine][kg_update] 双重提取: %d 个实体, %d 条关系",
                    len(data.get("entities", [])), len(data.get("relations", [])),
                )
            else:
                # 仅故事文本提取模式
                data = kg_extract(story_text)
                logger.debug(
                    "[Engine][kg_update] 故事提取: %d 个实体, %d 条关系",
                    len(data.get("entities", [])), len(data.get("relations", [])),
                )
            all_entities.extend(data.get("entities", []))
            all_relations.extend(data.get("relations", []))
        except Exception as exc:
            logger.warning("[Engine][kg_update] 提取失败: %s", exc)

        # ========== 步骤 2: 添加 NLU 层提取的实体 ==========
        nlu_names: List[str] = []
        if nlu_entities:
            for ent in nlu_entities:
                name = ent.get("text", "")
                etype = ent.get("type", "thing")
                if name:
                    # NLU 提取的实体标记为玩家提及
                    self.kg.add_entity(name, etype, turn_id=turn_id, is_player_mentioned=True, emotion=emotion)
                    nlu_names.append(name)

        # ========== 步骤 3: 添加 LLM 提取的实体（带丰富属性） ==========
        extracted_names: List[str] = []
        for ent in all_entities:
            name = ent.get("name") or ent.get("text", "")
            etype = ent.get("type", "unknown")
            desc = ent.get("description", "")
            status = ent.get("status", {})
            state_changes = ent.get("state_changes", {})
            if not name:
                continue

            # 添加或更新实体（带描述和状态）
            self.kg.add_entity(
                name, etype,
                description=desc,
                status=status,
                turn_id=turn_id,
            )
            extracted_names.append(name)

            # 对现有实体应用状态变更
            if state_changes:
                updated = self.kg.update_entity_state(name, state_changes, turn_id=turn_id)
                if updated:
                    logger.debug("[Engine][kg_update] '%s' 状态变更: %s", name, state_changes)

        # ========== 步骤 4: 添加提取的关系 ==========
        for rel in all_relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            label = rel.get("relation", "related_to")
            context = rel.get("context", "")
            if src and tgt:
                self.kg.add_relation(src, tgt, label, context=context, turn_id=turn_id)

        # ========== 步骤 5: 刷新提及跟踪 ==========
        # 合并所有被提及的实体名称
        all_mentioned = list(set(extracted_names + nlu_names))
        if all_mentioned:
            # 更新提及次数和重要性分数
            self.kg.refresh_mentions(all_mentioned, turn_id=turn_id, player_mentioned_names=nlu_names)

        # ========== 步骤 6: 应用时间衰减 ==========
        # 降低长时间未确认的关系的置信度，删除弱关系
        decay_cadence = max(1, settings.KG_DECAY_CADENCE)
        if turn_id % decay_cadence == 0:
            self.kg.apply_decay(turn_id=turn_id)

        # ========== 步骤 7: 重新计算重要性 ==========
        # 基于度数、新近度、提及次数等因素重新评估实体重要性
        self.kg.recalculate_importance()

        logger.info(
            "[Engine][kg_update] Turn %d complete | added: %d entities, %d relations | KG: %d nodes, %d edges",
            turn_id, len(extracted_names) + len(nlu_names), len(all_relations),
            self.kg.num_nodes, self.kg.num_edges,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_game(self, filepath: Optional[str] = None) -> str:
        """Save the current game state (KG + story history) to a JSON file.

        保存当前游戏状态（知识图谱 + 故事历史）到 JSON 文件。

        参数:
            filepath: 保存路径（可选，默认使用配置的保存目录）

        返回:
            str: 实际使用的文件路径
        """
        import json
        from pathlib import Path

        if filepath is None:
            save_dir = Path(settings.KG_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(save_dir / f"{self.genre}_latest.json")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        game_data = {
            "version": 1,
            "genre": self.genre,
            "state": {
                "turn_id": self.state.turn_id,
                "genre": self.state.genre,
                "story_history": self.state.story_history,
            },
            "kg": self.kg.to_dict(),
            "conflict_resolution": self.conflict_resolution,
            "extraction_mode": self.extraction_mode,
            "importance_mode": self.importance_mode,
            "summary_mode": self.summary_mode,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)

        logger.info("[Engine][save_game] Saved to %s | turn=%d", path, self.state.turn_id)
        return str(path)

    def load_game(self, filepath: str) -> None:
        """Load a saved game state from a JSON file.
        
        从 JSON 文件加载保存的游戏状态。
        恢复：游戏状态、知识图谱、冲突检测器、策略配置。
        
        参数:
            filepath: 要加载的文件路径
        """
        import json
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            logger.warning("[Engine][load_game] File not found: %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            game_data = json.load(f)

        # Restore state
        state_data = game_data.get("state", {})
        self.state.turn_id = state_data.get("turn_id", 0)
        self.state.genre = state_data.get("genre", self.genre)
        self.state.story_history = state_data.get("story_history", [])

        # Restore KG
        kg_data = game_data.get("kg", {})
        self.kg = KnowledgeGraph.from_dict(kg_data)
        self.conflict_det = ConflictDetector(self.kg)
        self.conflict_resolver = get_resolver(self.conflict_resolution)

        # Restore strategies
        self.conflict_resolution = game_data.get("conflict_resolution", self.conflict_resolution)
        self.extraction_mode = game_data.get("extraction_mode", self.extraction_mode)

        logger.info(
            "[Engine][load_game] Loaded from %s | turn=%d | KG: %d nodes, %d edges",
            path, self.state.turn_id, self.kg.num_nodes, self.kg.num_edges,
        )

    def _auto_save(self) -> None:
        """自动保存游戏状态（如果启用）。

        保存策略：
        - 每次回合后保存最新状态到 "{genre}_latest.json"
        - 每隔 N 回合（KG_SNAPSHOT_INTERVAL）保存一次快照

        快照命名格式："{genre}_turn_{turn_id}.json"
        """
        if not settings.KG_AUTO_SAVE:
            return
        try:
            save_dir = Path(settings.KG_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 保存最新状态
            self.save_game(str(save_dir / f"{self.genre}_latest.json"))

            # 定期快照（每 N 回合保存一次）
            if self.state.turn_id % settings.KG_SNAPSHOT_INTERVAL == 0:
                self.save_game(str(save_dir / f"{self.genre}_turn_{self.state.turn_id}.json"))
        except Exception as exc:
            logger.warning("[Engine][auto_save] 自动保存失败: %s", exc)
