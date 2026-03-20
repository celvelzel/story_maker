"""Intent classification using fine-tuned DistilBERT + keyword fallback.

使用微调的 DistilBERT 模型 + 关键词回退进行意图分类。

8 个意图标签：
- action: 动作（攻击、防御等）
- dialogue: 对话（交谈、询问等）
- explore: 探索（搜索、调查等）
- use_item: 使用物品（使用、装备等）
- ask_info: 询问信息（谁、什么、哪里等）
- rest: 休息（休息、睡觉等）
- trade: 交易（买卖、交换等）
- other: 其他
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify player input into one of 8 intent categories.

    将玩家输入分类为 8 个意图类别之一。
    
    工作机制：
    1. 如果有微调的 DistilBERT 检查点，使用模型预测
    2. 否则使用 rule_fallback() 关键词匹配进行透明回退
    """

    # 关键词映射表：每个意图类别对应一组关键词
    KEYWORD_MAP: Dict[str, List[str]] = {
        "action": ["attack", "fight", "strike", "slash", "defend", "hit", "shoot", "cast", "punch", "kick"],
        "dialogue": ["talk", "speak", "ask", "tell", "greet", "say", "chat", "discuss", "whisper", "shout"],
        "explore": ["explore", "search", "look", "investigate", "wander", "examine", "inspect", "survey", "observe"],
        "use_item": ["use", "apply", "drink", "eat", "equip", "throw", "activate", "open", "consume"],
        "ask_info": ["who", "what", "where", "why", "how", "info", "information", "explain", "describe"],
        "rest": ["rest", "sleep", "wait", "camp", "sit", "relax", "meditate", "nap"],
        "trade": ["trade", "buy", "sell", "barter", "exchange", "offer", "shop", "purchase", "bargain"],
    }

    def __init__(self, model_path: Optional[str] = None, max_length: int = 128) -> None:
        """
        初始化意图分类器。
        
        参数:
            model_path: 模型路径（可选）
            max_length: 最大 token 长度
        """
        self.model_path = model_path  # 模型路径
        self.max_length = max_length  # 最大 token 长度
        self.model = None  # DistilBERT 模型
        self.tokenizer = None  # 分词器
        self.device = None  # 计算设备（CPU/GPU）
        self.backend = "rule_fallback"  # 当前后端："distilbert" 或 "rule_fallback"

    # ── model loading ─────────────────────────────────────
    def load(self) -> None:
        """Try to load a fine-tuned DistilBERT checkpoint from local path.
        
        尝试从本地路径加载微调的 DistilBERT 检查点。
        如果模型目录不存在或加载失败，回退到关键词匹配模式。
        """
        from pathlib import Path
        from config import settings

        configured_path = self.model_path or str(settings.INTENT_MODEL_PATH)
        model_dir = Path(configured_path)
        if not model_dir.exists():
            logger.info("Intent model directory missing (%s) – using rule_fallback.", model_dir)
            self.backend = "rule_fallback"
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                num_labels=len(settings.INTENT_LABELS),
            ).to(self.device)
            self.model.eval()
            self.backend = "distilbert"
            logger.info("Intent classifier loaded from %s", model_dir)
        except Exception as exc:
            logger.warning("Intent model load failed (%s) – using rule_fallback.", exc)
            self.model = None
            self.tokenizer = None
            self.backend = "rule_fallback"

    # ── prediction ────────────────────────────────────────
    def predict(self, text: str) -> Dict[str, object]:
        """Return ``{"intent": str, "confidence": float}``.
        
        预测文本的意图。
        如果模型已加载，使用模型预测；否则使用关键词回退。
        
        参数:
            text: 要分类的文本
            
        返回:
            Dict: {"intent": 意图标签, "confidence": 置信度}
        """
        if self.model is not None and self.tokenizer is not None:
            return self._model_predict(text)
        return self.rule_fallback(text)

    def _model_predict(self, text: str) -> Dict[str, object]:
        """使用 DistilBERT 模型进行预测。"""
        import torch
        from config import settings

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][idx].item()
        return {"intent": settings.INTENT_LABELS[idx], "confidence": round(confidence, 4)}

    # ── keyword fallback ──────────────────────────────────
    def rule_fallback(self, text: str) -> Dict[str, object]:
        """Keyword-based intent classification (always available).
        
        基于关键词的意图分类（始终可用）。
        通过匹配文本中的关键词来确定意图。
        如果匹配分数过低，返回 "other" 意图。
        """
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
