"""LLM-based entity + relation extraction from story text.

基于 LLM 的故事文本实体与关系提取模块。

通过 OpenAI API 发送文本（JSON 模式），返回结构化的实体和关系数据，
用于填充知识图谱。

增强功能：
- 丰富的实体属性（描述、状态、状态变更）
- 丰富的关系属性（上下文）
- 双重提取模式（玩家输入 + 故事文本）
- 现有实体的状态变更检测

提取的实体类型：
- person（人物）、location（地点）、item（物品）
- creature（生物）、event（事件）

提取的关系类型：
- located_at（位于）、possesses（拥有）、ally_of（同盟）
- enemy_of（敌对）、knows（认识）、part_of（属于）
- caused_by（由...导致）、has_attribute（有属性）
- causes（导致）、prevents（阻止）、enables（使能）、follows（跟随）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# 同义词映射表：非标准类型名 → 标准 KG_ENTITY_TYPES
# 用于将 LLM 返回的各种类型名称统一为标准类型
_TYPE_SYNONYMS: Dict[str, str] = {
    # 人物类
    "character": "person",
    "npc": "person",
    "villain": "person",
    "hero": "person",
    "king": "person",
    "queen": "person",
    "guard": "person",
    "merchant": "person",
    "wizard": "person",
    "warrior": "person",
    # 地点类
    "place": "location",
    "room": "location",
    "area": "location",
    "region": "location",
    "building": "location",
    "kingdom": "location",
    # 物品类
    "weapon": "item",
    "armor": "item",
    "tool": "item",
    "object": "item",
    "artifact": "item",
    "relic": "item",
    "treasure": "item",
    "thing": "item",
    # 生物类
    "animal": "creature",
    "monster": "creature",
    "beast": "creature",
    "enemy": "creature",
    "boss": "creature",
    # 事件类
    "quest": "event",
    "mission": "event",
    "battle": "event",
    "encounter": "event",
    "situation": "event",
}


def _normalize_type(raw_type: str) -> str:
    """将实体类型字符串标准化为规范的 KG_ENTITY_TYPES 值。

    首先尝试直接匹配，然后查找同义词表。
    如果都无法匹配，返回 "unknown"。

    参数：
        raw_type: 原始类型字符串（如 "npc"、"weapon"）

    返回：
        str: 标准化的类型名，或 "unknown"（无法识别时）
    """
    normalized = raw_type.strip().lower()
    if not normalized:
        return "unknown"
    # 1. 直接匹配标准类型
    if normalized in settings.KG_ENTITY_TYPES:
        return normalized
    # 2. 查找同义词表
    mapped = _TYPE_SYNONYMS.get(normalized)
    if mapped and mapped in settings.KG_ENTITY_TYPES:
        return mapped
    # 3. 无法识别
    return "unknown"

_EXTRACTION_SYSTEM = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "Given a passage of story text, extract entities and relations with rich attributes.\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event", '
    '"description": str, "status": {"key": "value"}, "state_changes": {"key": "value"}}], '
    '"relations": [{"source": str, "target": str, "relation": str, "context": str}]}\n'
    "- description: a brief narrative description of the entity\n"
    "- status: current dynamic state (e.g. health, mood, location)\n"
    "- state_changes: state fields that changed in this passage (e.g. {\"health\": \"injured\"})\n"
    "- context: brief description of how this relation was established or confirmed\n"
    "- Keep relation types simple and lowercase: located_at, possesses, ally_of, enemy_of, knows, part_of, caused_by, has_attribute\n"
    "- Also extract causal relations when present:\n"
    '  "causes": A directly causes B\n'
    '  "prevents": A prevents B from happening\n'
    '  "enables": A enables B to happen\n'
    '  "follows": B temporally follows A (for event chains)\n'
    "- If an entity already exists, focus on state_changes rather than duplicating description\n"
    "- Be thorough but avoid trivial or redundant extractions"
)

_EXTRACTION_SYSTEM_LEGACY = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "Given a passage of story text, extract entities and relations.\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event"}], '
    '"relations": [{"source": str, "target": str, "relation": str}]}\n'
    "Keep relation types simple and lowercase, e.g. located_at, possesses, ally_of, enemy_of, knows, etc."
)

_EXTRACTION_SYSTEM_DUAL = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "You will receive two text passages:\n"
    "1. PLAYER INPUT: the player's action or dialogue\n"
    "2. STORY TEXT: the narrator's response describing events\n\n"
    "Extract entities and relations from BOTH passages.\n"
    "Player input typically contains the player's intended actions and newly mentioned entities.\n"
    "Story text contains narrative events, NPC reactions, and world state changes.\n\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event", '
    '"description": str, "status": {"key": "value"}, "state_changes": {"key": "value"}}], '
    '"relations": [{"source": str, "target": str, "relation": str, "context": str}]}\n'
    "- description: a brief narrative description of the entity\n"
    "- status: current dynamic state (e.g. health, mood, location)\n"
    "- state_changes: state fields that changed in this passage (e.g. {\"health\": \"injured\"})\n"
    "- context: brief description of how this relation was established or confirmed\n"
    "- Keep relation types simple and lowercase: located_at, possesses, ally_of, enemy_of, knows, part_of, caused_by, has_attribute\n"
    "- Also extract causal relations when present:\n"
    '  "causes": A directly causes B\n'
    '  "prevents": A prevents B from happening\n'
    '  "enables": A enables B to happen\n'
    '  "follows": B temporally follows A (for event chains)\n'
    "- If an entity already exists, focus on state_changes rather than duplicating description\n"
    "- Merge entities mentioned in both passages into a single entry\n"
    "- Be thorough but avoid trivial or redundant extractions"
)


class RelationExtractor:
    """使用 LLM 从故事文本中提取实体和关系。

    通过发送结构化提示到 OpenAI API，解析返回的 JSON 数据，
    提取游戏中的人物、地点、物品、生物、事件等实体及其关系。

    支持两种提取模式：
    - 增强模式（默认）：提取丰富的实体属性（描述、状态、状态变更）
    - 传统模式：仅提取基本实体和关系
    """

    def __init__(self, enhanced: bool = True) -> None:
        """初始化关系提取器。

        参数：
            enhanced: 是否使用增强模式（提取丰富属性）
        """
        self.enhanced = enhanced  # 是否启用增强提取

    def extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """从故事文本中提取实体和关系。

        调用 LLM API，发送结构化提示，返回提取的实体和关系列表。
        任何异常都会返回空结果，确保游戏不会崩溃。

        参数：
            text: 要提取的故事文本

        返回：
            Dict[str, List[Dict]]: {
                "entities": 实体列表，每个包含 name, type, description, status, state_changes,
                "relations": 关系列表，每个包含 source, target, relation, context
            }
        """
        try:
            from src.utils.api_client import llm_client

            # 选择提取提示（增强模式 vs 传统模式）
            system_prompt = _EXTRACTION_SYSTEM if self.enhanced else _EXTRACTION_SYSTEM_LEGACY
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            # 调用 LLM API（低温度确保一致性）
            data = llm_client.chat_json(messages, temperature=0.2, max_tokens=512)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # 标准化实体字段
            for ent in entities:
                ent.setdefault("name", "")
                ent["type"] = _normalize_type(ent.get("type", "unknown"))
                ent.setdefault("description", "")
                ent.setdefault("status", {})
                ent.setdefault("state_changes", {})

            # 标准化关系字段
            for rel in relations:
                rel.setdefault("source", "")
                rel.setdefault("target", "")
                rel.setdefault("relation", "related_to")
                rel.setdefault("context", "")

            logger.info(
                "[Extractor][extract] 从文本中提取了 %d 个实体, %d 条关系 (长度=%d)",
                len(entities), len(relations), len(text),
            )
            return {"entities": entities, "relations": relations}
        except Exception as exc:
            logger.warning("[Extractor][extract] 提取失败: %s", exc)
            return {"entities": [], "relations": []}

    def extract_dual(
        self,
        player_input: str,
        story_text: str,
        existing_entities: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """从玩家输入和故事文本中同时提取（双重提取）。

        在单次 LLM 调用中同时处理玩家输入和故事文本，
        返回合并后的实体和关系列表（自动去重）。

        适用场景：
        - 玩家输入通常包含玩家意图和新提到的实体
        - 故事文本包含叙述事件、NPC 反应和世界状态变更

        参数：
            player_input: 玩家输入文本
            story_text: 叙述者回复的故事文本
            existing_entities: 知识图谱中已存在的实体名称列表（可选）

        返回：
            Dict[str, List[Dict]]: 合并后的实体和关系列表
        """
        logger.debug(
            "[Extractor][dual_extract] 玩家输入 (长度=%d), 故事文本 (长度=%d), 已有实体=%d",
            len(player_input), len(story_text), len(existing_entities or []),
        )

        # 如果没有故事文本，仅从玩家输入提取
        if not story_text.strip():
            return self._extract_player_input(player_input)

        # 构建关于现有实体的上下文提示
        existing_hint = ""
        if existing_entities:
            existing_hint = (
                f"\n\nExisting entities in the world: {', '.join(existing_entities[:20])}\n"
                "For existing entities, focus on state_changes rather than duplicating descriptions."
            )

        # 单次 LLM 调用处理两个文本
        try:
            from src.utils.api_client import llm_client

            # 组合文本（带有格式标记）
            combined_text = (
                f"### PLAYER INPUT:\n{player_input}\n\n"
                f"### STORY TEXT:\n{story_text}"
                f"{existing_hint}"
            )

            messages = [
                {"role": "system", "content": _EXTRACTION_SYSTEM_DUAL},
                {"role": "user", "content": combined_text},
            ]
            data = llm_client.chat_json(messages, temperature=0.2, max_tokens=512)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # 标准化实体字段
            for ent in entities:
                ent.setdefault("name", "")
                ent["type"] = _normalize_type(ent.get("type", "unknown"))
                ent.setdefault("description", "")
                ent.setdefault("status", {})
                ent.setdefault("state_changes", {})

            # 标准化关系字段
            for rel in relations:
                rel.setdefault("source", "")
                rel.setdefault("target", "")
                rel.setdefault("relation", "related_to")
                rel.setdefault("context", "")

            logger.info(
                "[Extractor][dual_extract] 单次调用提取: %d 个实体, %d 条关系",
                len(entities), len(relations),
            )
            return {"entities": entities, "relations": relations}

        except Exception as exc:
            logger.warning("[Extractor][dual_extract] 单次调用失败 (%s)，回退到分离提取", exc)
            # 回退方案：分别提取后合并
            story_data = self.extract(story_text + existing_hint)
            player_data = self._extract_player_input(player_input + existing_hint)
            return self._merge_extractions(story_data, player_data)

    def _extract_player_input(self, player_input: str) -> Dict[str, List[Dict[str, Any]]]:
        """从玩家输入中提取实体和关系（简化提示版本）。

        专门针对玩家输入设计，只提取明确提到的新实体。
        使用较短的提示和更低的温度参数。

        参数：
            player_input: 玩家输入文本

        返回：
            Dict[str, List[Dict]]: 提取的实体和关系
        """
        try:
            from src.utils.api_client import llm_client

            system = (
                "You are a knowledge-graph extraction engine. "
                "Given a player's action input, extract any NEW entities or relations mentioned.\n"
                "Return ONLY valid JSON:\n"
                '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event", '
                '"description": str}], '
                '"relations": [{"source": str, "target": str, "relation": str, "context": str}]}\n'
                "Only extract what is explicitly mentioned. If nothing new, return empty arrays."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": player_input},
            ]
            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # 标准化字段
            for ent in entities:
                ent.setdefault("name", "")
                ent["type"] = _normalize_type(ent.get("type", "unknown"))
                ent.setdefault("description", "")
                ent.setdefault("status", {})
                ent.setdefault("state_changes", {})
            for rel in relations:
                rel.setdefault("source", "")
                rel.setdefault("target", "")
                rel.setdefault("relation", "related_to")
                rel.setdefault("context", "")

            return {"entities": entities, "relations": relations}
        except Exception as exc:
            logger.warning("[Extractor][_extract_player_input] 提取失败: %s", exc)
            return {"entities": [], "relations": []}

    def _merge_extractions(
        self,
        primary: Dict[str, List[Dict[str, Any]]],
        secondary: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """合并两个提取结果，按名称去重（大小写不敏感）。

        合并策略：
        - 实体：保留信息更丰富的版本；优先保留非 "unknown" 类型
        - 关系：按 (source, target, relation) 三元组去重

        参数：
            primary: 主要提取结果
            secondary: 次要提取结果

        返回：
            Dict[str, List[Dict]]: 合并后的实体和关系列表
        """
        # 合并实体 — 冲突时保留更丰富的版本
        entity_map: Dict[str, Dict[str, Any]] = {}
        for ent in primary.get("entities", []) + secondary.get("entities", []):
            name = (ent.get("name") or "").strip().lower()
            if not name:
                continue
            if name in entity_map:
                # 合并：优先保留非空字段
                existing = entity_map[name]
                for key in ("description", "status", "state_changes"):
                    if not existing.get(key) and ent.get(key):
                        existing[key] = ent[key]
                    elif isinstance(existing.get(key), dict) and isinstance(ent.get(key), dict):
                        existing[key] = {**existing[key], **ent[key]}
                # 优先保留非 "unknown" 类型
                if ent.get("type", "unknown") != "unknown":
                    existing["type"] = ent["type"]
            else:
                entity_map[name] = dict(ent)

        # 合并关系 — 按 (source, target, relation) 去重
        rel_set: set = set()
        merged_rels: List[Dict[str, Any]] = []
        for rel in primary.get("relations", []) + secondary.get("relations", []):
            src = (rel.get("source") or "").strip().lower()
            tgt = (rel.get("target") or "").strip().lower()
            label = rel.get("relation", "related_to")
            key = (src, tgt, label)
            if key not in rel_set and src and tgt:
                rel_set.add(key)
                merged_rels.append(rel)

        return {"entities": list(entity_map.values()), "relations": merged_rels}


# 模块级便捷单例
_extractor_enhanced = RelationExtractor(enhanced=True)  # 增强模式提取器
_extractor_legacy = RelationExtractor(enhanced=False)  # 传统模式提取器


def extract(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """模块级便捷函数，使用增强模式提取。

    参数：
        text: 要提取的故事文本

    返回：
        Dict[str, List[Dict]]: 提取的实体和关系
    """
    return _extractor_enhanced.extract(text)


def extract_legacy(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """模块级便捷函数，使用传统模式提取（向后兼容）。

    参数：
        text: 要提取的故事文本

    返回：
        Dict[str, List[Dict]]: 提取的实体和关系（仅基本字段）
    """
    return _extractor_legacy.extract(text)


def extract_dual(
    player_input: str,
    story_text: str,
    existing_entities: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """模块级便捷函数，使用双重提取模式。

    参数：
        player_input: 玩家输入文本
        story_text: 故事文本
        existing_entities: 已有实体列表（可选）

    返回：
        Dict[str, List[Dict]]: 合并后的实体和关系
    """
    return _extractor_enhanced.extract_dual(player_input, story_text, existing_entities)
