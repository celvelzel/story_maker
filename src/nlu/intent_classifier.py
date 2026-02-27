"""Intent classification using fine-tuned RoBERTa + keyword fallback.

8 labels: action, dialogue, explore, use_item, ask_info, rest, trade, other
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify player input into one of 8 intent categories.

    When a fine-tuned RoBERTa checkpoint is available it is used;
    otherwise ``rule_fallback()`` keyword matching takes over transparently.
    """

    KEYWORD_MAP: Dict[str, List[str]] = {
        "action": ["attack", "fight", "strike", "slash", "defend", "hit", "shoot", "cast", "punch", "kick"],
        "dialogue": ["talk", "speak", "ask", "tell", "greet", "say", "chat", "discuss", "whisper", "shout"],
        "explore": ["explore", "search", "look", "investigate", "wander", "examine", "inspect", "survey", "observe"],
        "use_item": ["use", "apply", "drink", "eat", "equip", "throw", "activate", "open", "consume"],
        "ask_info": ["who", "what", "where", "why", "how", "info", "information", "explain", "describe"],
        "rest": ["rest", "sleep", "wait", "camp", "sit", "relax", "meditate", "nap"],
        "trade": ["trade", "buy", "sell", "barter", "exchange", "offer", "shop", "purchase", "bargain"],
    }

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None

    # ── model loading ─────────────────────────────────────
    def load(self) -> None:
        """Try to load a fine-tuned RoBERTa checkpoint."""
        if not self.model_path:
            logger.info("No intent model path – using rule_fallback.")
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from config import settings

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(settings.INTENT_LABELS),
            ).to(self.device)
            self.model.eval()
            logger.info("Intent classifier loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("Intent model load failed (%s) – using rule_fallback.", exc)
            self.model = None

    # ── prediction ────────────────────────────────────────
    def predict(self, text: str) -> Dict[str, object]:
        """Return ``{"intent": str, "confidence": float}``."""
        if self.model is not None and self.tokenizer is not None:
            return self._model_predict(text)
        return self.rule_fallback(text)

    def _model_predict(self, text: str) -> Dict[str, object]:
        import torch
        from config import settings

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][idx].item()
        return {"intent": settings.INTENT_LABELS[idx], "confidence": round(confidence, 4)}

    # ── keyword fallback ──────────────────────────────────
    def rule_fallback(self, text: str) -> Dict[str, object]:
        """Keyword-based intent classification (always available)."""
        text_lower = text.lower()
        best_intent = "other"
        best_score = 0.0

        for intent, keywords in self.KEYWORD_MAP.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = matches / len(keywords) if keywords else 0
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score < 0.05:
            return {"intent": "other", "confidence": 0.5}
        return {"intent": best_intent, "confidence": round(min(best_score + 0.3, 1.0), 4)}
