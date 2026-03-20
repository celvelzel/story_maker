"""Sentiment / emotion analysis for player input.

情感/情绪分析模块：分析玩家输入的情感。

6 类 Ekman 模型：anger（愤怒）, disgust（厌恶）, fear（恐惧）, joy（快乐）, sadness（悲伤）, surprise（惊讶）+ neutral（中性）。

主要方式：HuggingFace distilroberta 模型。
回退方式：基于关键词的规则匹配。
"""
from __future__ import annotations

import logging
import inspect
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# Keyword fallback rules
_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "anger": [
        "angry", "furious", "rage", "kill", "damn", "stupid", "hate",
        "destroy", "revenge", "wrath", "annoyed", "irritated", "frustrated",
        "attack", "fight", "smash", "crush", "vengeance",
    ],
    "disgust": [
        "disgusting", "gross", "vile", "repulsive", "revolting", "nasty",
        "foul", "sickening", "horrible", "terrible", "awful", "rotten",
        "filthy", "putrid", "loathsome",
    ],
    "fear": [
        "afraid", "scared", "terrified", "danger", "flee", "escape",
        "hide", "panic", "dread", "horror", "frightened", "worried",
        "nervous", "anxious", "creepy", "spooky", "threatening", "menacing",
    ],
    "joy": [
        "happy", "wonderful", "great", "excited", "love", "amazing",
        "fantastic", "excellent", "delighted", "cheerful", "glad", "pleased",
        "celebrate", "triumph", "victory", "win", "success", "beautiful",
        "brilliant", "magnificent",
    ],
    "sadness": [
        "sad", "sorry", "lost", "miss", "cry", "mourn", "grief",
        "depressed", "unhappy", "miserable", "heartbroken", "lonely",
        "hopeless", "tragic", "weep", "tears", "despair", "gloomy",
    ],
    "surprise": [
        "wow", "amazing", "unexpected", "sudden", "shocked", "astonished",
        "startled", "incredible", "unbelievable", "surprised", "stunned",
        "bewildered", "astounded",
    ],
}


class SentimentAnalyzer:
    """Analyze emotion in player input text.
    
    情感分析器：分析玩家输入文本中的情绪。
    支持 6 种 Ekman 情绪类别 + 中性。
    """

    def __init__(self) -> None:
        """初始化情感分析器。"""
        self.model = None  # distilroberta 模型
        self.tokenizer = None  # 分词器
        self.device = None  # 计算设备
        self.backend = "rule_fallback"  # 当前后端

    MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    def load(self) -> None:
        """Load the emotion classification model with retry and fallback.
        
        Attempts to load the distilroberta model up to MAX_RETRIES times.
        Falls back to rule-based keyword matching on any failure.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._check_transformers_version()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_NAME,
                ).to(self.device)
                self.model.eval()
                self.backend = "distilroberta"
                logger.info("Sentiment analyzer loaded: %s (attempt %d)", self.MODEL_NAME, attempt)
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    logger.warning(
                        "Sentiment model load attempt %d/%d failed (%s) – retrying in %.1fs",
                        attempt, self.MAX_RETRIES, exc, self.RETRY_DELAY,
                    )
                    time.sleep(self.RETRY_DELAY)
                    self._reset_model_state()

        logger.warning(
            "Sentiment model load exhausted %d attempts (%s) – using rule_fallback.",
            self.MAX_RETRIES, last_error,
        )
        self._reset_model_state()
        self.backend = "rule_fallback"

    def _check_transformers_version(self) -> None:
        """Warn if transformers version is outside tested range."""
        try:
            import transformers
            v = getattr(transformers, "__version__", "unknown")
            major = int(v.split(".")[0])
            minor = int(v.split(".")[1])
            if major > 4 or (major == 4 and minor >= 50):
                logger.warning(
                    "transformers %s detected – version >= 4.50 may have breaking changes "
                    "in model/tokenizer compatibility. Tested range: 4.40–4.49.",
                    v,
                )
        except Exception:
            pass

    def _reset_model_state(self) -> None:
        """Reset model, tokenizer, and device to clean state."""
        self.model = None
        self.tokenizer = None
        self.device = None

    def analyze(self, text: str) -> Dict[str, object]:
        """Analyze the emotion in text.

        分析文本中的情感。
        
        参数:
            text: 要分析的文本
            
        返回:
            Dict: {
                "emotion": str,          # 主导情绪标签
                "confidence": float,     # 置信度分数 (0-1)
                "scores": Dict[str, float]  # 所有情绪分数
            }
        """
        if self.model is not None and self.tokenizer is not None:
            return self._model_analyze(text)
        return self._rule_fallback(text)

    def _model_analyze(self, text: str) -> Dict[str, object]:
        """Analyze using the neural model."""
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
            return_token_type_ids=False,
        )
        inputs = self._filter_model_inputs(inputs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        scores = {}
        for i, label in enumerate(EMOTION_LABELS):
            if i < len(probs):
                scores[label] = round(probs[i].item(), 4)

        # Find dominant emotion
        max_idx = torch.argmax(probs).item()
        dominant = EMOTION_LABELS[max_idx] if max_idx < len(EMOTION_LABELS) else "neutral"
        confidence = probs[max_idx].item()

        return {
            "emotion": dominant,
            "confidence": round(confidence, 4),
            "scores": scores,
        }

    def _rule_fallback(self, text: str) -> Dict[str, object]:
        """Keyword-based emotion detection."""
        text_lower = text.lower()
        scores: Dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}

        for emotion, keywords in _EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                scores[emotion] = round(min(matches / 3.0, 1.0), 4)

        # Find dominant emotion
        max_score = 0.0
        dominant = "neutral"
        for emotion, score in scores.items():
            if score > max_score:
                max_score = score
                dominant = emotion

        if max_score < 0.1:
            dominant = "neutral"
            max_score = 0.5

        return {
            "emotion": dominant,
            "confidence": round(max_score, 4),
            "scores": scores,
        }

    def _filter_model_inputs(self, inputs):
        """Filter tokenizer outputs to parameters accepted by model.forward."""
        signature = inspect.signature(self.model.forward)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs:
            return inputs

        allowed_keys = set(parameters.keys())
        return {key: value for key, value in inputs.items() if key in allowed_keys}
