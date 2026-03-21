"""Tests for DistilBERT / tokenizer compatibility fixes.

Verifies that:
1. _filter_model_inputs correctly strips unexpected keys
2. Rule-fallback remains functional when model is None
"""
import pytest

from src.nlu.intent_classifier import IntentClassifier
from src.nlu.sentiment_analyzer import SentimentAnalyzer


# ── Stubs for testing without real model files ─────────────────────────

class _FakeForwardWithKwargs:
    """Simulates a model.forward that accepts **kwargs (e.g. BERT)."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        pass


class _FakeForwardStrict:
    """Simulates a model.forward that does NOT accept **kwargs (e.g. DistilBERT)."""

    def forward(self, input_ids=None, attention_mask=None):
        pass


class _FakeTokenizerWithTTI:
    """Tokenizer that returns token_type_ids (problematic case)."""

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "token_type_ids": [[0, 0, 0]],
        }


# ── IntentClassifier: filter tests ─────────────────────────────────────

class TestIntentClassifierFilter:
    def _build_clf(self, forward_cls, tokenizer):
        clf = IntentClassifier()
        clf.model = forward_cls()
        clf.tokenizer = tokenizer()
        clf.device = "cpu"
        clf.backend = "distilbert"
        return clf

    def test_filter_removes_token_type_ids_strict_model(self):
        clf = self._build_clf(_FakeForwardStrict, _FakeTokenizerWithTTI)
        raw = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "token_type_ids": [[0, 0, 0]],
        }
        filtered = clf._filter_model_inputs(raw)
        assert "token_type_ids" not in filtered
        assert "input_ids" in filtered
        assert "attention_mask" in filtered

    def test_filter_passes_through_kwargs_model(self):
        clf = self._build_clf(_FakeForwardWithKwargs, _FakeTokenizerWithTTI)
        raw = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "token_type_ids": [[0, 0, 0]],
        }
        filtered = clf._filter_model_inputs(raw)
        assert "token_type_ids" in filtered
        assert "input_ids" in filtered
        assert "attention_mask" in filtered

    def test_filter_strips_arbitrary_extra_keys(self):
        clf = self._build_clf(_FakeForwardStrict, _FakeTokenizerWithTTI)
        raw = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "token_type_ids": [[0, 0, 0]],
            "extra_field": [[9, 9, 9]],
            "another_junk": "should be removed",
        }
        filtered = clf._filter_model_inputs(raw)
        assert set(filtered.keys()) == {"input_ids", "attention_mask"}


# ── SentimentAnalyzer: filter tests ───────────────────────────────────

class TestSentimentAnalyzerFilter:
    def _build_analyzer(self, forward_cls, tokenizer):
        analyzer = SentimentAnalyzer()
        analyzer.model = forward_cls()
        analyzer.tokenizer = tokenizer()
        analyzer.device = "cpu"
        analyzer.backend = "distilroberta"
        return analyzer

    def test_filter_removes_token_type_ids_strict_model(self):
        analyzer = self._build_analyzer(_FakeForwardStrict, _FakeTokenizerWithTTI)
        raw = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "token_type_ids": [[0, 0, 0]],
        }
        filtered = analyzer._filter_model_inputs(raw)
        assert "token_type_ids" not in filtered
        assert "input_ids" in filtered
        assert "attention_mask" in filtered

    def test_filter_passes_through_kwargs_model(self):
        analyzer = self._build_analyzer(_FakeForwardWithKwargs, _FakeTokenizerWithTTI)
        raw = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "token_type_ids": [[0, 0, 0]],
        }
        filtered = analyzer._filter_model_inputs(raw)
        assert "token_type_ids" in filtered


# ── Cross-backend fallback verification ───────────────────────────────

class TestIntentClassifierFallback:
    def test_rule_fallback_still_works_when_model_is_none(self):
        clf = IntentClassifier()
        assert clf.model is None
        result = clf.predict("attack now")
        assert result["intent"] in {
            "action", "dialogue", "explore", "use_item",
            "ask_info", "rest", "trade", "other",
        }
        assert "confidence" in result


class TestSentimentAnalyzerFallback:
    def test_rule_fallback_still_works_when_model_is_none(self):
        analyzer = SentimentAnalyzer()
        assert analyzer.model is None
        result = analyzer.analyze("I am very angry")
        assert result["emotion"] in {
            "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral",
        }
        assert "confidence" in result
        assert "scores" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
