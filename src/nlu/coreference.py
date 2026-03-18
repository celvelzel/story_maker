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
            # ── Compatibility patch for transformers 5.2.0 x fastcoref 2.x ──────────────────
            # Problem: transformers 5.2.0 removed the 'all_tied_weights_keys' attribute that
            #          fastcoref 2.x still expects during model loading/serialization.
            # Error:   AttributeError: 'FCorefModel' object has no attribute 'all_tied_weights_keys'
            #
            # Solution: Inject an empty dict-like object as this attribute on PreTrainedModel
            #           before fastcoref loads any models. fastcoref only checks the structure
            #           of this attribute (specifically calls .keys() on it), not its contents.
            #
            # Why dict subclass?
            #   - fastcoref code does: for key in model.all_tied_weights_keys.keys()
            #   - Need the .keys() method, not just a property or function
            #   - dict subclass provides complete dict interface automatically
            #
            # Performance: <1ms one-time patch applied at module load time
            # Safety: Patch only applies if attribute doesn't exist (no overwrites)
            from transformers.modeling_utils import PreTrainedModel
            
            class _TiedWeightsCompat(dict):
                """
                Drop-in replacement for transformers 5.2.0's missing 'all_tied_weights_keys'.
                Empty dict satisfies fastcoref's serialization checks during model loading.
                """
                def __init__(self):
                    super().__init__()
            
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = _TiedWeightsCompat()
            
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
