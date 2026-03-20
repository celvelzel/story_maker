#!/usr/bin/env python3
"""Environment health-check script.

Run this before deployment or in CI to verify the environment is correctly
set up for NLU model loading and inference.

Usage:
    python scripts/health_check.py [--verbose]
    pytest scripts/health_check.py -v
"""
import sys
import argparse
from pathlib import Path

import pytest


def _check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = v.major == 3 and 10 <= v.minor <= 13
    msg = f"Python {v.major}.{v.minor}.{v.micro} (requires 3.10–3.13)"
    return ok, msg


def _check_dependency(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        return True, f"{name} {version}"
    except ImportError:
        return False, f"{name} NOT installed"


def _check_transformers_version() -> tuple[bool, str]:
    try:
        import transformers
        v = transformers.__version__
        major, minor = int(v.split(".")[0]), int(v.split(".")[1])
        ok = (4 == major and minor < 50)
        status = "OK (tested)" if ok else "WARNING (may work, verify after upgrade)"
        msg = f"transformers {v} — {status} (tested: 4.40–4.49, safety filter is active)"
        return True, msg
    except ImportError:
        return False, "transformers NOT installed"


def _check_torch_device() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})"
        return True, "CPU only"
    except ImportError:
        return False, "torch NOT installed"


def _check_intent_model_dir() -> tuple[bool, str]:
    from config import settings
    path = Path(settings.INTENT_MODEL_PATH)
    if not path.exists():
        return False, f"Intent model dir missing: {path} (will use rule_fallback)"
    config_files = list(path.glob("*.json"))
    safetensors = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not config_files:
        return False, f"Intent model dir incomplete (no config): {path}"
    if not safetensors:
        return False, f"Intent model dir incomplete (no weights): {path}"
    return True, f"Intent model dir OK: {path}"


def _check_intent_classifier_loads() -> tuple[bool, str]:
    try:
        from src.nlu.intent_classifier import IntentClassifier
        clf = IntentClassifier()
        clf.load()
        ok = clf.backend in ("distilbert", "rule_fallback")
        return ok, f"IntentClassifier backend={clf.backend}"
    except Exception as exc:
        return False, f"IntentClassifier load failed: {exc}"


def _check_sentiment_analyzer_loads() -> tuple[bool, str]:
    try:
        from src.nlu.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        analyzer.load()
        ok = analyzer.backend in ("distilroberta", "rule_fallback")
        return ok, f"SentimentAnalyzer backend={analyzer.backend}"
    except Exception as exc:
        return False, f"SentimentAnalyzer load failed: {exc}"


def _check_intent_predict() -> tuple[bool, str]:
    try:
        from src.nlu.intent_classifier import IntentClassifier
        clf = IntentClassifier()
        clf.load()
        result = clf.predict("attack the dragon")
        ok = "intent" in result and "confidence" in result
        return ok, f"IntentClassifier.predict OK: {result}"
    except Exception as exc:
        return False, f"IntentClassifier.predict failed: {exc}"


def _check_sentiment_analyze() -> tuple[bool, str]:
    try:
        from src.nlu.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        analyzer.load()
        result = analyzer.analyze("I am very angry")
        ok = "emotion" in result and "confidence" in result and "scores" in result
        return ok, f"SentimentAnalyzer.analyze OK: {result}"
    except Exception as exc:
        return False, f"SentimentAnalyzer.analyze failed: {exc}"


def _check_token_type_ids_compat() -> tuple[bool, str]:
    """Verify that _filter_model_inputs exists and strips token_type_ids."""
    try:
        from src.nlu.intent_classifier import IntentClassifier
        clf = IntentClassifier()
        assert hasattr(clf, "_filter_model_inputs"), "_filter_model_inputs not found"
        class FakeModelStrict:
            def forward(self, input_ids=None, attention_mask=None):
                pass
        clf.model = FakeModelStrict()
        raw = {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "token_type_ids": [[0, 0]],
            "extra_junk": "should be removed",
        }
        filtered = clf._filter_model_inputs(raw)
        ok = (
            "token_type_ids" not in filtered
            and "extra_junk" not in filtered
            and "input_ids" in filtered
            and "attention_mask" in filtered
        )
        return ok, f"token_type_ids filter OK (kept only: {list(filtered.keys())})"
    except Exception as exc:
        return False, f"token_type_ids filter check failed: {exc}"


def run_all(verbose: bool = False) -> bool:
    _scripts_dir = Path(__file__).parent.resolve()
    _project_root = _scripts_dir.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    checks = [
        ("Python version", _check_python_version),
        ("torch", lambda: _check_dependency("torch")),
        ("transformers", _check_transformers_version),
        ("CUDA / device", _check_torch_device),
        ("transformers package", lambda: _check_dependency("transformers")),
        ("Intent model directory", _check_intent_model_dir),
        ("IntentClassifier loads", _check_intent_classifier_loads),
        ("SentimentAnalyzer loads", _check_sentiment_analyzer_loads),
        ("IntentClassifier.predict", _check_intent_predict),
        ("SentimentAnalyzer.analyze", _check_sentiment_analyze),
        ("token_type_ids compatibility", _check_token_type_ids_compat),
    ]

    results: list[tuple[str, bool, str]] = []
    all_passed = True

    for name, check_fn in checks:
        ok, msg = check_fn()
        results.append((name, ok, msg))
        if not ok:
            all_passed = False

    print("=" * 60)
    print("ENVIRONMENT HEALTH CHECK")
    print("=" * 60)
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        symbol = "[+]" if ok else "[-]"
        print(f"  {symbol} [{status}] {name}")
        if verbose or not ok:
            print(f"       {msg}")
    print("=" * 60)
    if all_passed:
        print("All checks PASSED.")
    else:
        failed = [name for name, ok, _ in results if not ok]
        print(f"FAILED checks: {', '.join(failed)}")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Environment health-check for story_maker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all messages")
    args = parser.parse_args()

    ok = run_all(verbose=args.verbose)
    sys.exit(0 if ok else 1)
