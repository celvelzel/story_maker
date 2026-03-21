"""Tests for KnowledgeGraph, ConflictDetector and relation_extractor."""
import pytest
from unittest.mock import patch, MagicMock

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.conflict_detector import ConflictDetector
from src.knowledge_graph import relation_extractor


# ── KnowledgeGraph ──────────────────────────────────────────────────

class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_add_entity(self, kg):
        kg.add_entity("Hero", "person")
        assert "hero" in kg.graph.nodes
        assert kg.graph.nodes["hero"]["entity_type"] == "person"

    def test_add_entity_case_insensitive(self, kg):
        kg.add_entity("HERO", "person")
        kg.add_entity("hero", "person")
        assert kg.graph.number_of_nodes() == 1

    def test_add_relation(self, kg):
        kg.add_entity("Hero", "person")
        kg.add_entity("Forest", "location")
        kg.add_relation("Hero", "Forest", "located_in")
        assert kg.graph.number_of_edges() == 1

    def test_skip_duplicate_relation(self, kg):
        kg.add_entity("Hero", "person")
        kg.add_entity("Forest", "location")
        kg.add_relation("Hero", "Forest", "located_in")
        kg.add_relation("Hero", "Forest", "located_in")
        assert kg.graph.number_of_edges() == 1

    def test_auto_create_nodes_for_relation(self, kg):
        kg.add_relation("A", "B", "knows")
        assert "a" in kg.graph.nodes
        assert "b" in kg.graph.nodes

    def test_to_summary(self, kg):
        kg.add_entity("Hero", "person")
        kg.add_entity("Sword", "item")
        kg.add_relation("Hero", "Sword", "possesses")
        summary = kg.to_summary()
        assert "hero" in summary.lower()

    def test_enforce_limit(self, kg):
        # Add many nodes and verify pruning
        for i in range(210):
            kg.add_entity(f"node{i}", "thing")
        kg._enforce_limit()
        assert kg.graph.number_of_nodes() <= 200


# ── ConflictDetector ────────────────────────────────────────────────

class TestConflictDetector:
    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_exclusive_pair_detected(self, kg):
        kg.add_entity("hero", "person")
        kg.add_entity("dragon", "creature")
        kg.add_relation("hero", "dragon", "alive")
        kg.add_relation("hero", "dragon", "dead")
        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()
        assert len(conflicts) >= 1

    def test_no_conflict_when_clean(self, kg):
        kg.add_entity("hero", "person")
        kg.add_entity("forest", "location")
        kg.add_relation("hero", "forest", "located_in")
        det = ConflictDetector(kg)
        conflicts = det._rule_based_check()
        assert conflicts == []


# ── Relation extractor (mocked LLM) ────────────────────────────────

class TestRelationExtractor:
    @patch("src.utils.api_client.llm_client")
    def test_extract_returns_dict(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [{"name": "Hero", "type": "person"}],
            "relations": [{"source": "Hero", "target": "Forest", "relation": "entered"}],
        }
        result = relation_extractor.extract("The hero entered the forest.")
        assert "entities" in result
        assert "relations" in result
        assert len(result["entities"]) == 1

    @patch("src.utils.api_client.llm_client")
    def test_extract_handles_error(self, mock_client):
        mock_client.chat_json.side_effect = RuntimeError("API down")
        result = relation_extractor.extract("broken")
        assert result == {"entities": [], "relations": []}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
