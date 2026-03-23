"""Tests for enhanced KnowledgeGraph (graph.py)."""
import pytest
from unittest.mock import patch

from src.knowledge_graph.graph import KnowledgeGraph


class TestAddEntityEnhanced:
    """Test enriched entity attributes and temporal tracking."""

    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_add_entity_with_description_and_status(self, kg):
        kg.add_entity(
            "Hero", "person",
            description="A brave warrior",
            status={"health": "full", "mood": "confident"},
            turn_id=1,
        )
        ent = kg.get_entity("hero")
        assert ent is not None
        assert ent["description"] == "A brave warrior"
        assert ent["status"] == {"health": "full", "mood": "confident"}
        assert ent["created_turn"] == 1
        assert ent["last_mentioned_turn"] == 1
        assert ent["mention_count"] == 1

    def test_add_entity_turn_tracking(self, kg):
        kg.add_entity("Hero", "person", turn_id=1)
        kg.add_entity("Hero", "person", turn_id=5)
        ent = kg.get_entity("hero")
        assert ent["created_turn"] == 1  # preserved from first creation
        assert ent["last_mentioned_turn"] == 5  # updated on second mention
        assert ent["mention_count"] == 2

    def test_add_entity_player_mentioned(self, kg):
        kg.add_entity("Sword", "item", turn_id=1, is_player_mentioned=True)
        ent = kg.get_entity("sword")
        assert ent["player_mention_count"] == 1
        # importance should be boosted by player mention
        assert ent["importance_score"] > 0.5

    def test_add_entity_merges_description(self, kg):
        kg.add_entity("Hero", "person", description="A brave warrior", turn_id=1)
        kg.add_entity("Hero", "person", description="Known for defeating dragons", turn_id=3)
        ent = kg.get_entity("hero")
        assert "brave warrior" in ent["description"]
        assert "defeating dragons" in ent["description"]

    def test_add_entity_merges_status(self, kg):
        kg.add_entity("Hero", "person", status={"health": "full"}, turn_id=1)
        kg.add_entity("Hero", "person", status={"health": "injured", "mood": "determined"}, turn_id=3)
        ent = kg.get_entity("hero")
        assert ent["status"]["health"] == "injured"
        assert ent["status"]["mood"] == "determined"

    def test_add_entity_case_insensitive(self, kg):
        kg.add_entity("HERO", "person")
        kg.add_entity("hero", "person")
        assert kg.num_nodes == 1

    def test_add_entity_importance_boost(self, kg):
        kg.add_entity("Hero", "person", turn_id=1)
        initial_score = kg.get_entity("hero")["importance_score"]
        kg.add_entity("Hero", "person", turn_id=2)
        updated_score = kg.get_entity("hero")["importance_score"]
        assert updated_score > initial_score


class TestAddRelationEnhanced:
    """Test enriched relation attributes."""

    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_add_relation_with_context_and_confidence(self, kg):
        kg.add_entity("Hero", "person", turn_id=1)
        kg.add_entity("Forest", "location", turn_id=1)
        kg.add_relation("Hero", "Forest", "located_at", context="Hero enters the forest", turn_id=1, confidence=0.9)
        rels = kg.get_relations("hero")
        assert len(rels) == 1
        assert rels[0]["context"] == "Hero enters the forest"
        assert rels[0]["confidence"] == 0.9
        assert rels[0]["created_turn"] == 1

    def test_duplicate_relation_updates_confidence(self, kg):
        kg.add_relation("Hero", "Forest", "located_at", turn_id=1, confidence=0.5)
        kg.add_relation("Hero", "Forest", "located_at", turn_id=3, confidence=0.9)
        rels = kg.get_relations("hero")
        # still only one edge
        out_rels = [r for r in rels if r.get("direction") == "out"]
        assert len(out_rels) == 1
        # confidence updated to max
        assert out_rels[0]["confidence"] == 0.9
        assert out_rels[0]["last_confirmed_turn"] == 3

    def test_add_relation_auto_creates_nodes(self, kg):
        kg.add_relation("A", "B", "knows")
        assert kg.get_entity("a") is not None
        assert kg.get_entity("b") is not None

    def test_get_relations_direction(self, kg):
        kg.add_relation("Hero", "Sword", "possesses")
        hero_rels = kg.get_relations("hero")
        sword_rels = kg.get_relations("sword")
        assert any(r["direction"] == "out" for r in hero_rels)
        assert any(r["direction"] == "in" for r in sword_rels)


class TestUpdateEntityState:
    """Test entity state update method."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", status={"health": "full"}, turn_id=1)
        return kg

    def test_update_existing_entity(self, kg):
        result = kg.update_entity_state("Hero", {"health": "injured", "location": "cave"}, turn_id=3)
        assert result is True
        ent = kg.get_entity("hero")
        assert ent["status"]["health"] == "injured"
        assert ent["status"]["location"] == "cave"
        assert ent["last_mentioned_turn"] == 3

    def test_update_nonexistent_entity(self, kg):
        result = kg.update_entity_state("Ghost", {"visibility": "visible"})
        assert result is False


class TestRefreshMentions:
    """Test batch mention refresh and decay."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        kg.add_entity("Sword", "item", turn_id=1)
        kg.add_entity("Dragon", "creature", turn_id=1)
        return kg

    def test_refresh_mentioned_entities(self, kg):
        kg.refresh_mentions(["Hero", "Sword"], turn_id=5, player_mentioned_names=["Hero"])
        hero = kg.get_entity("hero")
        sword = kg.get_entity("sword")
        dragon = kg.get_entity("dragon")

        assert hero["last_mentioned_turn"] == 5
        assert hero["mention_count"] > 1
        assert hero["player_mention_count"] >= 1
        assert sword["last_mentioned_turn"] == 5

    def test_unmentioned_entities_decay(self, kg):
        # Get initial importance
        initial_dragon = kg.get_entity("dragon")["importance_score"]
        # Refresh only Hero and Sword
        kg.refresh_mentions(["Hero", "Sword"], turn_id=5)
        decayed_dragon = kg.get_entity("dragon")["importance_score"]
        assert decayed_dragon < initial_dragon


class TestApplyDecay:
    """Test relation confidence decay and pruning."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_relation("Hero", "Forest", "located_at", turn_id=1, confidence=0.8)
        kg.add_relation("Hero", "Sword", "possesses", turn_id=5, confidence=0.9)
        return kg

    def test_old_relations_decay(self, kg):
        kg.apply_decay(turn_id=10)
        hero_rels = kg.get_relations("hero")
        forest_rel = [r for r in hero_rels if r.get("target") == "forest"]
        sword_rel = [r for r in hero_rels if r.get("target") == "sword"]
        assert len(forest_rel) == 1
        # forest relation (turn 1) should decay more than sword (turn 5)
        if sword_rel:
            assert forest_rel[0]["confidence"] < sword_rel[0]["confidence"]

    def test_weak_relations_pruned(self, kg):
        # Set very low initial confidence
        kg.add_relation("Hero", "Cave", "located_at", turn_id=1, confidence=0.1)
        kg.apply_decay(turn_id=20)
        cave_rels = [r for r in kg.get_relations("hero") if r.get("target") == "cave"]
        # should be pruned
        assert len(cave_rels) == 0


class TestRecalculateImportance:
    """Test importance recalculation."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1, is_player_mentioned=True)
        kg.add_entity("Forest", "location", turn_id=1)
        kg.add_entity("Sword", "item", turn_id=1)
        kg.add_relation("Hero", "Sword", "possesses", turn_id=1)
        kg.add_relation("Hero", "Forest", "located_at", turn_id=1)
        return kg

    def test_composite_importance(self, kg):
        kg.recalculate_importance()
        hero = kg.get_entity("hero")
        forest = kg.get_entity("forest")
        # Hero has more connections + player mention -> higher importance
        assert hero["importance_score"] >= forest["importance_score"]

    def test_degree_only_mode(self, kg):
        with patch("src.knowledge_graph.graph.settings") as mock_settings:
            mock_settings.KG_IMPORTANCE_MODE = "degree_only"
            mock_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
            mock_settings.KG_RELATION_DECAY_FACTOR = 0.90
            mock_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
            mock_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
            mock_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
            mock_settings.KG_MAX_NODES = 200
            kg.recalculate_importance()
            hero = kg.get_entity("hero")
            assert 0.0 <= hero["importance_score"] <= 1.0

    def test_incremental_mode_matches_composite(self, kg):
        # Fix _current_turn so recency_score is consistent (turns_since must be positive)
        kg._current_turn = 10

        # Baseline composite scores
        with patch("src.knowledge_graph.graph.settings") as mock_settings:
            mock_settings.KG_IMPORTANCE_MODE = "composite"
            mock_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
            mock_settings.KG_RELATION_DECAY_FACTOR = 0.90
            mock_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
            mock_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
            mock_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
            mock_settings.KG_MAX_NODES = 200
            mock_settings.KG_ENABLE_INCREMENTAL_IMPORTANCE = True
            mock_settings.KG_INCREMENTAL_FULL_RECALC_INTERVAL = 10
            kg.recalculate_importance()
            composite_scores = {
                n: kg.graph.nodes[n]["importance_score"] for n in kg.graph.nodes()
            }

        # Incremental mode should produce same score on full-recalc turn (turn 10 % 10 == 0)
        kg._dirty_nodes.clear()
        with patch("src.knowledge_graph.graph.settings") as mock_settings:
            mock_settings.KG_IMPORTANCE_MODE = "incremental"
            mock_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
            mock_settings.KG_RELATION_DECAY_FACTOR = 0.90
            mock_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
            mock_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
            mock_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
            mock_settings.KG_MAX_NODES = 200
            mock_settings.KG_ENABLE_INCREMENTAL_IMPORTANCE = True
            mock_settings.KG_INCREMENTAL_FULL_RECALC_INTERVAL = 10
            kg.recalculate_importance()

        for node, score in composite_scores.items():
            assert abs(kg.graph.nodes[node]["importance_score"] - score) < 1e-6


class TestGetTimeline:
    """Test timeline generation."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_relation("Hero", "Forest", "located_at", turn_id=1, context="Hero entered forest")
        kg.add_relation("Hero", "Dragon", "enemy_of", turn_id=3, context="Hero challenged dragon")
        kg.add_relation("Hero", "Sword", "possesses", turn_id=5, context="Hero found sword")
        return kg

    def test_timeline_sorted_by_turn(self, kg):
        timeline = kg.get_timeline(n=2)
        assert len(timeline) == 2
        assert timeline[0]["last_confirmed_turn"] >= timeline[1]["last_confirmed_turn"]

    def test_timeline_limit(self, kg):
        timeline = kg.get_timeline(n=1)
        assert len(timeline) == 1


class TestToSummary:
    """Test summary generation in both modes."""

    @pytest.fixture
    def kg_rich(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", description="A brave warrior", status={"health": "full"}, turn_id=1, is_player_mentioned=True)
        kg.add_entity("Sword", "item", description="Ancient glowing blade", turn_id=1)
        kg.add_entity("Dragon", "creature", description="A fearsome dragon", turn_id=3)
        kg.add_entity("Village", "location", turn_id=1)
        kg.add_relation("Hero", "Sword", "possesses", turn_id=1, confidence=0.95)
        kg.add_relation("Hero", "Dragon", "enemy_of", turn_id=3, context="Hero challenged dragon")
        return kg

    def test_flat_summary_backward_compat(self, kg_rich):
        with patch("src.knowledge_graph.graph.settings") as mock_settings:
            mock_settings.KG_SUMMARY_MODE = "flat"
            mock_settings.KG_MAX_TIMELINE_ENTRIES = 5
            mock_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
            mock_settings.KG_RELATION_DECAY_FACTOR = 0.90
            mock_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
            mock_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
            mock_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
            mock_settings.KG_MAX_NODES = 200
            mock_settings.KG_IMPORTANCE_MODE = "composite"
            summary = kg_rich.to_summary()
            assert "=== World State ===" in summary
            assert "=== Relations ===" in summary
            assert "hero" in summary.lower()

    def test_layered_summary_sections(self, kg_rich):
        with patch("src.knowledge_graph.graph.settings") as mock_settings:
            mock_settings.KG_SUMMARY_MODE = "layered"
            mock_settings.KG_MAX_TIMELINE_ENTRIES = 5
            mock_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
            mock_settings.KG_RELATION_DECAY_FACTOR = 0.90
            mock_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
            mock_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
            mock_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
            mock_settings.KG_MAX_NODES = 200
            mock_settings.KG_IMPORTANCE_MODE = "composite"
            summary = kg_rich.to_summary()
            assert "Core Entities" in summary or "Secondary Entities" in summary or "Background" in summary
            assert "Description:" in summary
            assert "Timeline" in summary or "Relations:" in summary

    def test_layered_summary_includes_entity_details(self, kg_rich):
        with patch("src.knowledge_graph.graph.settings") as mock_settings:
            mock_settings.KG_SUMMARY_MODE = "layered"
            mock_settings.KG_MAX_TIMELINE_ENTRIES = 5
            mock_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
            mock_settings.KG_RELATION_DECAY_FACTOR = 0.90
            mock_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
            mock_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
            mock_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
            mock_settings.KG_MAX_NODES = 200
            mock_settings.KG_IMPORTANCE_MODE = "composite"
            summary = kg_rich.to_summary()
            assert "A brave warrior" in summary
            assert "health: full" in summary

    def test_empty_graph_summary(self):
        kg = KnowledgeGraph()
        summary = kg.to_summary()
        assert "(empty)" in summary


class TestEnforceLimit:
    """Test node pruning based on importance."""

    def test_prunes_lowest_importance(self):
        kg = KnowledgeGraph()
        for i in range(210):
            kg.add_entity(f"node{i}", "thing", turn_id=1, is_player_mentioned=(i < 10))
        kg._enforce_limit()
        assert kg.num_nodes <= 200
        # high importance nodes (player mentioned) should survive
        assert kg.get_entity("node0") is not None


class TestSetTurn:
    """Test turn counter management."""

    def test_set_turn(self):
        kg = KnowledgeGraph()
        kg.set_turn(5)
        assert kg._current_turn == 5

    def test_turn_id_used_for_new_entities(self):
        kg = KnowledgeGraph()
        kg.set_turn(3)
        kg.add_entity("Hero", "person")
        assert kg.get_entity("hero")["created_turn"] == 3


class TestBackwardCompatibility:
    """Ensure original test_knowledge_graph.py tests still pass."""

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
        for i in range(210):
            kg.add_entity(f"node{i}", "thing")
        kg._enforce_limit()
        assert kg.graph.number_of_nodes() <= 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
