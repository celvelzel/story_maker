"""Tests for NLU modules: intent classifier, entity extractor, coreference."""
import pytest

from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.nlu.coreference import CoreferenceResolver


# ── Intent classifier (keyword / rule fallback) ─────────────────────

class TestIntentClassifier:
    @pytest.fixture
    def clf(self):
        return IntentClassifier()

    @pytest.mark.parametrize("text,expected", [
        ("explore the dark cave", "explore"),
        ("attack the goblin", "action"),
        ("talk to the elder", "dialogue"),
        ("use the healing potion", "use_item"),
        ("what is this place", "ask_info"),
        ("rest by the fire", "rest"),
        ("trade with the merchant", "trade"),
    ])
    def test_keyword_intents(self, clf, text, expected):
        result = clf.predict(text)
        assert result["intent"] == expected

    def test_unknown_falls_back_to_other(self, clf):
        result = clf.predict("xyzzy gibberish")
        assert result["intent"] == "other"

    def test_confidence_range(self, clf):
        result = clf.predict("explore the dungeon")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_dict(self, clf):
        result = clf.predict("look around")
        assert "intent" in result and "confidence" in result


# ── Entity extractor ────────────────────────────────────────────────

class TestEntityExtractor:
    @pytest.fixture
    def ext(self):
        return EntityExtractor()

    def test_extract_returns_list(self, ext):
        entities = ext.extract("A knight walked into the cave.")
        assert isinstance(entities, list)

    def test_entity_dict_keys(self, ext):
        entities = ext.extract("Gandalf visited the Shire.")
        for e in entities:
            assert "text" in e
            assert "type" in e

    def test_empty_input(self, ext):
        assert ext.extract("") == []


# ── Coreference resolver ────────────────────────────────────────────

class TestCoreferenceResolver:
    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_resolve_returns_string(self, resolver):
        out = resolver.resolve("He went there.", ["The knight saw a cave."])
        assert isinstance(out, str)

    def test_no_context_passthrough(self, resolver):
        text = "Open the door."
        assert resolver.resolve(text, []) == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
