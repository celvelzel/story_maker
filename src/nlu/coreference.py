"""Coreference resolution using fastcoref FCoref.

Resolves pronouns in player input given recent story context.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class CoreferenceResolver:
    """Resolve pronouns → antecedents using fastcoref (or rule fallback)."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        try:
            from fastcoref import FCoref  # type: ignore[import-untyped]
            self.model = FCoref(device="cpu")
            logger.info("Coreference resolver loaded (fastcoref)")
        except Exception as exc:
            logger.warning("fastcoref unavailable (%s) – rule-based fallback.", exc)
            self.model = None

    # ── public API ────────────────────────────────────────
    def resolve(self, text: str, context: Optional[List[str]] = None) -> str:
        """Return *text* with pronouns replaced by antecedents when possible."""
        if not context:
            return text

        full = " ".join(context[-3:]) + " " + text

        if self.model is not None:
            return self._neural_resolve(full, text)
        return self._rule_resolve(text, context)

    # ── neural ────────────────────────────────────────────
    def _neural_resolve(self, full_context: str, original: str) -> str:
        try:
            preds = self.model.predict(texts=[full_context])
            if preds and hasattr(preds[0], "get_resolved_text"):
                resolved = preds[0].get_resolved_text()
                # Only keep the tail that corresponds to the original input
                if len(resolved) > len(original):
                    return resolved[len(resolved) - len(original) :]
                return resolved
        except Exception as exc:
            logger.warning("Neural coref failed: %s", exc)
        return original

    # ── rule fallback ─────────────────────────────────────
    @staticmethod
    def _rule_resolve(text: str, context: List[str]) -> str:
        recent = " ".join(context[-2:]) if context else ""
        names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", recent)
        if not names:
            return text
        last_name = names[-1]
        for pronoun in ("him", "her", "them", "he", "she", "they"):
            pat = rf"\b{pronoun}\b"
            if re.search(pat, text, re.IGNORECASE):
                text = re.sub(pat, last_name, text, count=1, flags=re.IGNORECASE)
                break
        return text
