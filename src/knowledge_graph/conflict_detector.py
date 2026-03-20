"""Rule-based + LLM conflict detection and multi-strategy resolution for the knowledge graph.

知识图谱冲突检测与多策略解决模块。

检测层次：
- 第 1 层 — 确定性规则检测（互斥关系对、属性矛盾）
- 第 2 层 — LLM 推理检测（基于知识图谱摘要的逻辑矛盾）

解决策略：
- ``keep_latest`` — 保留最新信息，删除较早的矛盾数据
- ``llm_arbitrate`` — 让 LLM 决定保留哪些信息

检测的冲突类型：
- exclusive_relation（互斥关系）：如同时存在 ally_of 和 enemy_of
- dead_active（死亡活跃）：已死亡实体仍有活跃关系
- temporal（时间矛盾）：死亡后的行为、因果倒置
- llm（LLM检测）：LLM 推理发现的逻辑矛盾
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)

# 互斥关系对：同一 source→target 之间不能同时存在的关系类型
EXCLUSIVE_PAIRS: List[tuple[str, str]] = [
    ("ally_of", "enemy_of"),  # 同盟与敌对互斥
    ("alive", "dead"),        # 存活与死亡互斥
]


# ══════════════════════════════════════════════════════════
#  Detection（检测层）
# ══════════════════════════════════════════════════════════

class ConflictDetector:
    """知识图谱矛盾检测器。

    通过多层检测机制发现知识图谱中的逻辑矛盾：
    - 规则检测：互斥关系、死亡活跃实体
    - 时间检测：时间矛盾、因果倒置
    - LLM 检测：基于 LLM 推理的深度矛盾发现
    """

    def __init__(self, kg: Any) -> None:  # 避免循环导入
        """初始化冲突检测器。

        参数：
            kg: 关联的知识图谱对象
        """
        self.kg = kg  # 关联的知识图谱

    # ── public entry point ────────────────────────────────
    def check_all(self, new_text: str = "") -> List[Dict[str, str]]:
        """执行所有检测层，返回冲突字典列表。

        综合使用规则检测、时间检测和 LLM 检测。
        返回的每个冲突字典包含 type、description 等字段。

        参数：
            new_text: 新生成的故事文本（用于 LLM 检测，可选）

        返回：
            List[Dict[str, str]]: 冲突列表，可能为空
        """
        conflicts: List[Dict[str, str]] = []
        conflicts.extend(self._rule_based_check())  # 规则层检测
        conflicts.extend(self._temporal_check())   # 时间层检测
        if new_text:
            conflicts.extend(self._llm_check(new_text))  # LLM 层检测
        logger.info(
            "[ConflictDetector][check_all] 发现 %d 个冲突 (规则=%d, 时间=%d, LLM=%d)",
            len(conflicts),
            len([c for c in conflicts if c.get("type") not in ("llm", "temporal")]),
            len([c for c in conflicts if c.get("type") == "temporal"]),
            len([c for c in conflicts if c.get("type") == "llm"]),
        )
        return conflicts

    # ── layer 1: rule-based ───────────────────────────────
    def _rule_based_check(self) -> List[Dict[str, str]]:
        """基于规则的冲突检测（确定性层）。

        检测以下冲突类型：
        1. 互斥关系对：同一实体对之间存在矛盾的关系
        2. 死亡活跃：已标记为死亡的实体仍有活跃状态关系

        返回：
            List[Dict[str, str]]: 规则层检测到的冲突列表
        """
        conflicts: List[Dict[str, str]] = []
        g = self.kg.graph

        # 1. 检测互斥关系对
        for src, tgt, data in g.edges(data=True):
            rel = data.get("relation", "")
            for a, b in EXCLUSIVE_PAIRS:
                # 检查是否存在互斥关系
                opposite = b if rel == a else (a if rel == b else None)
                if opposite is None:
                    continue
                if g.has_edge(src, tgt):
                    for _k, edata in g[src][tgt].items():
                        if edata.get("relation") == opposite:
                            conflicts.append({
                                "type": "exclusive_relation",
                                "source": src,
                                "target": tgt,
                                "relation_a": rel,
                                "relation_b": opposite,
                                "description": (
                                    f"{src} 同时存在 '{rel}' 和 '{opposite}' 关系指向 {tgt}."
                                ),
                            })

        # 2. 检测死亡实体的活跃关系
        for node, ndata in g.nodes(data=True):
            status = ndata.get("status", {})
            if isinstance(status, dict) and status.get("status") == "dead":
                for _, tgt, edata in g.out_edges(node, data=True):
                    rel = edata.get("relation", "")
                    if rel in ("possesses", "located_at", "ally_of"):
                        conflicts.append({
                            "type": "dead_active",
                            "source": node,
                            "target": tgt,
                            "relation": rel,
                            "description": f"{node} 已死亡但仍有关系 '{rel}' → {tgt}.",
                        })
        return conflicts

    # ── layer 1b: temporal ────────────────────────────────
    def _temporal_check(self) -> List[Dict[str, str]]:
        """检查时间/因果矛盾。

        检测以下冲突类型：
        1. 死后行为：实体死亡后仍创建关系
        2. 因果倒置：原因实体在结果实体之后才被创建

        返回：
            List[Dict[str, str]]: 时间层检测到的冲突列表
        """
        conflicts: List[Dict[str, str]] = []
        g = self.kg.graph

        # 1. 检测死亡后的实体行为
        for node, ndata in g.nodes(data=True):
            status = ndata.get("status", {})
            if not isinstance(status, dict) or status.get("status") != "dead":
                continue

            # 查找实体的死亡回合（从状态历史中）
            death_turn = None
            for entry in reversed(ndata.get("status_history", [])):
                changes = entry.get("changes", {})
                if "status" in changes and "dead" in changes["status"]:
                    death_turn = entry["turn"]
                    break

            if death_turn is None:
                # 死亡状态直接设置，使用最后提及回合作为代理
                death_turn = ndata.get("last_mentioned_turn", 999)

            # 检查死亡后创建的关系
            for _, tgt, edata in g.out_edges(node, data=True):
                rel_turn = edata.get("created_turn", 0)
                if rel_turn > death_turn:
                    rel = edata.get("relation", "?")
                    conflicts.append({
                        "type": "temporal",
                        "subtype": "dead_entity_action",
                        "source": node,
                        "target": tgt,
                        "relation": rel,
                        "death_turn": death_turn,
                        "relation_turn": rel_turn,
                        "description": (
                            f"{node} (死亡于回合 {death_turn}) 仍有关系 "
                            f"'{rel}' → {tgt}，创建于回合 {rel_turn}."
                        ),
                    })

        # 2. 检测因果倒置
        for src, tgt, data in g.edges(data=True):
            relation = data.get("relation", "")
            if relation not in ("causes", "enables"):
                continue
            if not g.has_node(src) or not g.has_node(tgt):
                continue

            src_created = g.nodes[src].get("created_turn", 0)
            tgt_created = g.nodes[tgt].get("created_turn", 0)

            # 如果结果实体比原因实体更早创建，则存在因果倒置
            if tgt_created < src_created:
                conflicts.append({
                    "type": "temporal",
                    "subtype": "causal_inversion",
                    "source": src,
                    "target": tgt,
                    "relation": relation,
                    "description": (
                        f"因果倒置: {src} (创建于回合 {src_created}) "
                        f"'{relation}' {tgt} (创建于回合 {tgt_created}), "
                        f"但 {tgt} 实际上更早被创建."
                    ),
                })

        return conflicts

    # ── layer 2: LLM ─────────────────────────────────────
    def _llm_check(self, new_text: str) -> List[Dict[str, str]]:
        """基于 LLM 推理的冲突检测（深度分析层）。

        将当前世界状态和新生成的故事文本发送给 LLM，
        让 LLM 识别可能存在的逻辑矛盾。

        参数：
            new_text: 新生成的故事文本

        返回：
            List[Dict[str, str]]: LLM 检测到的冲突列表
        """
        try:
            from src.utils.api_client import llm_client

            kg_summary = self.kg.to_summary()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a consistency checker for a text adventure knowledge graph.\n"
                        "Given the current world state and a new story passage, identify any "
                        "logical contradictions.  Return JSON: "
                        '{"conflicts": [{"description": str}]}  '
                        "Return an empty list if there are none."
                    ),
                },
                {
                    "role": "user",
                    "content": f"World state:\n{kg_summary}\n\nNew text:\n{new_text}",
                },
            ]
            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            raw = data.get("conflicts", [])
            return [{"type": "llm", "description": c.get("description", str(c))} for c in raw]
        except Exception as exc:
            logger.warning("[ConflictDetector][_llm_check] LLM检测失败: %s", exc)
            return []


# ══════════════════════════════════════════════════════════
#  Resolution Strategies（解决策略层）
# ══════════════════════════════════════════════════════════

class ConflictResolutionStrategy(ABC):
    """冲突解决策略抽象基类。

    定义冲突解决的接口规范。
    所有具体解决策略都需要实现 resolve 方法。
    """

    @abstractmethod
    def resolve(self, conflicts: List[Dict[str, str]], kg: Any) -> List[Dict[str, str]]:
        """解决知识图谱中的冲突。

        参数：
            conflicts: 冲突列表
            kg: 知识图谱对象

        返回：
            List[Dict[str, str]]: 未能解决的冲突列表
        """
        ...


class KeepLatestResolver(ConflictResolutionStrategy):
    """保留最新策略。

    解决冲突的方法：保留较新的信息，删除较早的矛盾数据。
    对于互斥关系对，删除 last_confirmed_turn 值较小的边。
    """

    def resolve(self, conflicts: List[Dict[str, str]], kg: Any) -> List[Dict[str, str]]:
        """解决冲突列表。

        处理逻辑：
        - exclusive_relation：删除较旧的关系
        - dead_active：删除死亡实体的活跃关系
        - llm：LLM检测的冲突无法确定性解决，保留为未解决

        参数：
            conflicts: 冲突列表
            kg: 知识图谱对象

        返回：
            List[Dict[str, str]]: 未能解决的冲突列表
        """
        unresolved: List[Dict[str, str]] = []
        for conflict in conflicts:
            ctype = conflict.get("type", "")

            if ctype == "exclusive_relation":
                resolved = self._resolve_exclusive(conflict, kg)
                if not resolved:
                    unresolved.append(conflict)
            elif ctype == "dead_active":
                # 对于死亡实体的活跃关系，直接删除该关系
                src = conflict.get("source", "")
                rel = conflict.get("relation", "")
                tgt = conflict.get("target", "")
                if src and tgt and rel:
                    self._remove_relation(kg, src, tgt, rel)
                    logger.info(
                        "[KeepLatestResolver] 删除了死亡活跃关系 %s --[%s]--> %s",
                        src, rel, tgt,
                    )
            elif ctype == "llm":
                # LLM检测的冲突：无法确定性解决，保留为未解决
                unresolved.append(conflict)
            else:
                unresolved.append(conflict)

        return unresolved

    def _resolve_exclusive(self, conflict: Dict[str, str], kg: Any) -> bool:
        """解决互斥关系冲突，删除两者中较旧的关系。

        参数：
            conflict: 互斥关系冲突
            kg: 知识图谱对象

        返回：
            bool: 是否成功解决
        """
        src = conflict.get("source", "")
        tgt = conflict.get("target", "")
        rel_a = conflict.get("relation_a", "")
        rel_b = conflict.get("relation_b", "")

        if not (src and tgt and rel_a and rel_b):
            return False

        g = kg.graph
        if not g.has_edge(src, tgt):
            return False

        # 查找两条互斥关系的确认时间
        turn_a, turn_b = -1, -1
        key_a, key_b = None, None

        for k, data in g[src][tgt].items():
            if data.get("relation") == rel_a:
                turn_a = data.get("last_confirmed_turn", 0)
                key_a = k
            elif data.get("relation") == rel_b:
                turn_b = data.get("last_confirmed_turn", 0)
                key_b = k

        # 删除较旧的关系
        if turn_a >= turn_b and key_b is not None:
            g.remove_edge(src, tgt, key=key_b)
            logger.info(
                "[KeepLatestResolver] 删除了较旧关系 %s --[%s]--> %s (回合 %d < %d)",
                src, rel_b, tgt, turn_b, turn_a,
            )
            return True
        elif turn_b > turn_a and key_a is not None:
            g.remove_edge(src, tgt, key=key_a)
            logger.info(
                "[KeepLatestResolver] 删除了较旧关系 %s --[%s]--> %s (回合 %d < %d)",
                src, rel_a, tgt, turn_a, turn_b,
            )
            return True
        return False

    @staticmethod
    def _remove_relation(kg: Any, src: str, tgt: str, relation: str) -> None:
        """从图中删除指定关系的所有边。

        参数：
            kg: 知识图谱对象
            src: 源实体
            tgt: 目标实体
            relation: 关系类型
        """
        g = kg.graph
        if not g.has_edge(src, tgt):
            return
        keys_to_remove = []
        for k, data in g[src][tgt].items():
            if data.get("relation") == relation:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            g.remove_edge(src, tgt, key=k)


class LLMArbitrateResolver(ConflictResolutionStrategy):
    """LLM 仲裁解决策略。

    通过调用 LLM 来决定如何解决冲突。
    LLM 可以根据上下文选择：保留新信息、保留旧信息、删除关系、更新实体状态等。
    """

    def resolve(self, conflicts: List[Dict[str, str]], kg: Any) -> List[Dict[str, str]]:
        """解决冲突列表，每个冲突都通过 LLM 仲裁。

        对每个冲突调用 LLM 获取解决建议，然后执行相应操作。

        参数：
            conflicts: 冲突列表
            kg: 知识图谱对象

        返回：
            List[Dict[str, str]]: 未能解决的冲突列表
        """
        unresolved: List[Dict[str, str]] = []

        for conflict in conflicts:
            ctype = conflict.get("type", "")

            if ctype in ("exclusive_relation", "dead_active", "llm"):
                resolved = self._arbitrate_single(conflict, kg)
                if not resolved:
                    unresolved.append(conflict)
            else:
                unresolved.append(conflict)

        return unresolved

    def _arbitrate_single(self, conflict: Dict[str, str], kg: Any) -> bool:
        """使用 LLM 决定单个冲突的解决方法。

        LLM 可以返回以下解决方式：
        - keep_new: 删除较旧的关系
        - keep_old: 删除较新的关系
        - remove_relation: 删除特定关系
        - update_entity: 更新实体状态
        - no_action: 无需操作

        参数：
            conflict: 冲突字典
            kg: 知识图谱对象

        返回：
            bool: 是否成功解决
        """
        try:
            from src.utils.api_client import llm_client

            kg_summary = kg.to_summary()
            conflict_desc = conflict.get("description", str(conflict))

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conflict resolver for a text adventure knowledge graph.\n"
                        "Given a conflict and the current world state, decide how to resolve it.\n"
                        "Return ONLY valid JSON:\n"
                        '{"resolution": "keep_new"|"keep_old"|"remove_relation"|"update_entity"|"no_action", '
                        '"target_entity": str, "target_relation": str, "reason": str}\n'
                        "- keep_new: remove the older/conflicting relation, keep the newer one\n"
                        "- keep_old: remove the newer relation\n"
                        "- remove_relation: remove a specific relation\n"
                        "- update_entity: update an entity's status\n"
                        "- no_action: conflict is not a real problem, ignore it"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conflict: {conflict_desc}\n"
                        f"Conflict type: {conflict.get('type', 'unknown')}\n"
                        f"World state:\n{kg_summary}"
                    ),
                },
            ]

            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            resolution = data.get("resolution", "no_action")
            reason = data.get("reason", "")

            logger.info(
                "[LLMArbitrateResolver] 冲突: '%s' | 解决方案=%s | 原因=%s",
                conflict_desc[:80], resolution, reason,
            )

            # LLM 认为无需操作
            if resolution == "no_action":
                return True

            # 根据 LLM 决定执行相应操作
            if resolution in ("keep_new", "remove_relation"):
                return self._apply_remove(conflict, kg)

            if resolution == "keep_old":
                return self._apply_remove(conflict, kg, remove_newer=True)

            if resolution == "update_entity":
                return self._apply_entity_update(conflict, kg, data)

            return False

        except Exception as exc:
            logger.warning("[LLMArbitrateResolver] LLM仲裁失败: %s", exc)
            return False

    def _apply_remove(self, conflict: Dict[str, str], kg: Any, remove_newer: bool = False) -> bool:
        """根据冲突类型删除矛盾关系边。

        参数：
            conflict: 冲突字典
            kg: 知识图谱对象
            remove_newer: 是否删除较新的关系（True）或较旧的关系（False）

        返回：
            bool: 是否成功删除
        """
        ctype = conflict.get("type", "")
        g = kg.graph

        if ctype == "exclusive_relation":
            src = conflict.get("source", "")
            tgt = conflict.get("target", "")
            rel_a = conflict.get("relation_a", "")
            rel_b = conflict.get("relation_b", "")
            if not (src and tgt and rel_a and rel_b and g.has_edge(src, tgt)):
                return False

            # 查找两条互斥关系的时间戳
            turns: Dict[str, int] = {}
            keys: Dict[str, Any] = {}
            for k, data in g[src][tgt].items():
                rel = data.get("relation", "")
                if rel in (rel_a, rel_b):
                    turns[rel] = data.get("last_confirmed_turn", 0)
                    keys[rel] = k

            # 确定要删除的目标关系
            if remove_newer:
                target_rel = max(turns, key=turns.get) if turns else None
            else:
                target_rel = min(turns, key=turns.get) if turns else None

            if target_rel and target_rel in keys:
                g.remove_edge(src, tgt, key=keys[target_rel])
                logger.info("[LLMArbitrateResolver] 删除了 '%s' 边 %s→%s", target_rel, src, tgt)
                return True

        elif ctype == "dead_active":
            src = conflict.get("source", "")
            tgt = conflict.get("target", "")
            rel = conflict.get("relation", "")
            if src and tgt and rel:
                KeepLatestResolver._remove_relation(kg, src, tgt, rel)
                logger.info("[LLMArbitrateResolver] 删除了死亡活跃关系 %s→%s [%s]", src, tgt, rel)
                return True

        return False

    def _apply_entity_update(self, conflict: Dict[str, str], kg: Any, llm_data: Dict) -> bool:
        """根据 LLM 仲裁建议更新实体状态。

        参数：
            conflict: 冲突字典
            kg: 知识图谱对象
            llm_data: LLM 返回的数据，包含 target_entity 等字段

        返回：
            bool: 是否成功更新
        """
        target = llm_data.get("target_entity", "")
        if not target:
            return False
        # 标记为已解决 — 状态更新将由引擎处理
        logger.info("[LLMArbitrateResolver] 建议更新实体 '%s' 的状态", target)
        return True


# ══════════════════════════════════════════════════════════
#  Factory（工厂函数）
# ══════════════════════════════════════════════════════════

def get_resolver(mode: str = "") -> ConflictResolutionStrategy:
    """工厂函数：返回适当的冲突解决策略。

    根据 mode 参数返回对应的解决策略实例。
    如果 mode 为空，使用配置文件中的默认策略。

    参数：
        mode: 解决策略模式，可选 "keep_latest" 或 "llm_arbitrate"

    返回：
        ConflictResolutionStrategy: 解决策略实例
    """
    mode = mode or settings.KG_CONFLICT_RESOLUTION
    if mode == "keep_latest":
        logger.debug("[ConflictResolver] 使用 KeepLatestResolver")
        return KeepLatestResolver()
    elif mode == "llm_arbitrate":
        logger.debug("[ConflictResolver] 使用 LLMArbitrateResolver")
        return LLMArbitrateResolver()
    else:
        logger.warning("[ConflictResolver] 未知模式 '%s'，回退到 llm_arbitrate", mode)
        return LLMArbitrateResolver()
