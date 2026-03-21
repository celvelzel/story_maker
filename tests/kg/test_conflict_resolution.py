"""Tests for conflict_detector.py — detection + multi-strategy resolution."""
import pytest
from unittest.mock import patch, MagicMock

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.conflict_detector import (
    ConflictDetector,
    KeepLatestResolver,
    LLMArbitrateResolver,
    get_resolver,
    EXCLUSIVE_PAIRS,
)


# ══════════════════════════════════════════════════════════
#  Detection tests (backward compatible)
# ══════════════════════════════════════════════════════════

class TestConflictDetection:
    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_exclusive_pair_detected(self, kg):
        kg.add_entity("hero", "person")
        kg.add_entity("dragon", "creature")
        kg.add_relation("hero", "dragon", "ally_of")
        kg.add_relation("hero", "dragon", "enemy_of")
        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()
        assert len(conflicts) >= 1
        assert conflicts[0]["type"] == "exclusive_relation"

    def test_no_conflict_when_clean(self, kg):
        kg.add_entity("hero", "person")
        kg.add_entity("forest", "location")
        kg.add_relation("hero", "forest", "located_at")
        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()
        assert conflicts == []

    def test_check_all_combines_layers(self, kg):
        kg.add_entity("hero", "person")
        kg.add_entity("dragon", "creature")
        kg.add_relation("hero", "dragon", "alive")
        kg.add_relation("hero", "dragon", "dead")
        det = ConflictDetector(kg)
        with patch("src.utils.api_client.llm_client") as mock_llm:
            mock_llm.chat_json.return_value = {"conflicts": []}
            conflicts = det.check_all("Some new text")
        # at least the rule-based conflict
        assert len(conflicts) >= 1

    @patch("src.utils.api_client.llm_client")
    def test_llm_check_returns_conflicts(self, mock_llm):
        mock_llm.chat_json.return_value = {
            "conflicts": [{"description": "Hero cannot be in two places"}]
        }
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person")
        det = ConflictDetector(kg)
        conflicts = det._llm_check("Hero is in the forest and the cave.")
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "llm"

    @patch("src.utils.api_client.llm_client")
    def test_llm_check_handles_error(self, mock_llm):
        mock_llm.chat_json.side_effect = RuntimeError("fail")
        kg = KnowledgeGraph()
        det = ConflictDetector(kg)
        conflicts = det._llm_check("text")
        assert conflicts == []


# ══════════════════════════════════════════════════════════
#  KeepLatestResolver tests
# ══════════════════════════════════════════════════════════

class TestKeepLatestResolver:
    def test_resolves_exclusive_pair(self):
        kg = KnowledgeGraph()
        kg.add_relation("hero", "dragon", "ally_of", turn_id=3)
        kg.add_relation("hero", "dragon", "enemy_of", turn_id=7)

        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()
        assert len(conflicts) >= 1

        resolver = KeepLatestResolver()
        unresolved = resolver.resolve(conflicts, kg)

        # the older relation (ally_of at turn 3) should be removed
        rels = kg.get_relations("hero")
        out_rels = [r for r in rels if r.get("direction") == "out" and r.get("target") == "dragon"]
        labels = [r.get("relation") for r in out_rels]
        assert "ally_of" not in labels  # removed (older)
        assert "enemy_of" in labels  # kept (newer)

    def test_removes_dead_active_relation(self):
        kg = KnowledgeGraph()
        kg.add_entity("hero", "person", status={"status": "dead"}, turn_id=1)
        kg.add_entity("cave", "location", turn_id=1)
        kg.add_relation("hero", "cave", "located_at", turn_id=2)

        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()

        resolver = KeepLatestResolver()
        unresolved = resolver.resolve(conflicts, kg)

        hero_rels = [r for r in kg.get_relations("hero") if r.get("direction") == "out"]
        located_rels = [r for r in hero_rels if r.get("relation") == "located_at"]
        assert len(located_rels) == 0

    def test_no_conflict_no_action(self):
        kg = KnowledgeGraph()
        kg.add_entity("hero", "person")
        kg.add_entity("forest", "location")
        kg.add_relation("hero", "forest", "located_at")

        resolver = KeepLatestResolver()
        unresolved = resolver.resolve([], kg)
        assert unresolved == []
        assert kg.num_edges == 1


# ══════════════════════════════════════════════════════════
#  LLMArbitrateResolver tests
# ══════════════════════════════════════════════════════════

class TestLLMArbitrateResolver:
    @patch("src.utils.api_client.llm_client")
    def test_calls_llm_and_resolves(self, mock_llm):
        mock_llm.chat_json.return_value = {
            "resolution": "keep_new",
            "target_entity": "hero",
            "target_relation": "ally_of",
            "reason": "Alliance was formed more recently",
        }
        kg = KnowledgeGraph()
        kg.add_relation("hero", "dragon", "ally_of", turn_id=3)
        kg.add_relation("hero", "dragon", "enemy_of", turn_id=7)

        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()

        resolver = LLMArbitrateResolver()
        unresolved = resolver.resolve(conflicts, kg)

        # LLM chose keep_new -> should remove the older relation
        rels = kg.get_relations("hero")
        out_rels = [r for r in rels if r.get("direction") == "out" and r.get("target") == "dragon"]
        labels = [r.get("relation") for r in out_rels]
        assert "ally_of" not in labels  # removed as older

    @patch("src.utils.api_client.llm_client")
    def test_llm_no_action_considers_resolved(self, mock_llm):
        mock_llm.chat_json.return_value = {
            "resolution": "no_action",
            "reason": "Not a real conflict",
        }
        kg = KnowledgeGraph()
        kg.add_relation("hero", "dragon", "ally_of", turn_id=3)
        kg.add_relation("hero", "dragon", "enemy_of", turn_id=7)

        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()

        resolver = LLMArbitrateResolver()
        unresolved = resolver.resolve(conflicts, kg)
        # no_action means considered resolved
        assert len(unresolved) == 0

    @patch("src.utils.api_client.llm_client")
    def test_llm_error_returns_unresolved(self, mock_llm):
        mock_llm.chat_json.side_effect = RuntimeError("API down")
        kg = KnowledgeGraph()
        kg.add_relation("hero", "dragon", "ally_of", turn_id=3)
        kg.add_relation("hero", "dragon", "enemy_of", turn_id=7)

        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()

        resolver = LLMArbitrateResolver()
        unresolved = resolver.resolve(conflicts, kg)
        # on error, conflicts remain unresolved
        assert len(unresolved) >= 1


# ══════════════════════════════════════════════════════════
#  Factory tests
# ══════════════════════════════════════════════════════════

class TestGetResolver:
    def test_returns_keep_latest(self):
        resolver = get_resolver("keep_latest")
        assert isinstance(resolver, KeepLatestResolver)

    def test_returns_llm_arbitrate(self):
        resolver = get_resolver("llm_arbitrate")
        assert isinstance(resolver, LLMArbitrateResolver)

    def test_unknown_falls_back_to_llm(self):
        resolver = get_resolver("unknown_strategy")
        assert isinstance(resolver, LLMArbitrateResolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
