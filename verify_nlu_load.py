"""Verify NLU module loading status after GameEngine init."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.coreference import CoreferenceResolver
from src.nlu.entity_extractor import EntityExtractor

print("=" * 70)
print("NLU Module Load Status Verification")
print("=" * 70)

# Test 1: Initialize GameEngine and check nlu_status
print("\n[1] GameEngine Initialization Status")
engine = GameEngine(genre="fantasy")

print(f"  ✓ GameEngine initialized")
print(f"    - coref_loaded: {engine.nlu_status['coref_loaded']}")
print(f"    - intent_model_loaded: {engine.nlu_status['intent_model_loaded']}")
print(f"    - intent_backend: {engine.nlu_status['intent_backend']}")
print(f"    - entity_model_loaded: {engine.nlu_status['entity_model_loaded']}")

# Test 2: Check IntentClassifier directly
print("\n[2] IntentClassifier Status")
print(f"  - Model object: {engine.intent_clf.model is not None}")
print(f"  - Tokenizer object: {engine.intent_clf.tokenizer is not None}")
print(f"  - Backend: {engine.intent_clf.backend}")
print(f"  - Device: {engine.intent_clf.device}")

# Test 3: Test Intent prediction
print("\n[3] Intent Prediction Test")
test_inputs = [
    "attack the goblin",
    "talk to the elder",
    "explore the cave",
    "use the potion",
]
for text in test_inputs:
    result = engine.intent_clf.predict(text)
    print(f"  - '{text}'")
    print(f"    → intent: {result['intent']}, confidence: {result['confidence']}")

# Test 4: Check Coreference
print("\n[4] CoreferenceResolver Status")
print(f"  - Model object: {engine.coref.model is not None}")
test_coref = engine.coref.resolve("He attacked the enemy.", ["The warrior saw the enemy approaching."])
print(f"  - Sample resolution: '{test_coref}'")

# Test 5: Check Entity Extractor
print("\n[5] EntityExtractor Status")
print(f"  - spaCy model object: {engine.entity_ext.nlp is not None}")
entities = engine.entity_ext.extract("The knight walked into the castle and met the wizard.")
print(f"  - Sample extraction: {entities}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
all_loaded = (
    engine.nlu_status["coref_loaded"]
    and engine.nlu_status["intent_model_loaded"]
    and engine.nlu_status["entity_model_loaded"]
)
all_not_fallback = engine.nlu_status["intent_backend"] == "distilbert"

if all_loaded and all_not_fallback:
    print("✓ All NLU modules successfully loaded (NOT using fallback)")
    print("  - Coref: ✓ fastcoref active")
    print("  - Intent: ✓ distilbert-base-uncased active")
    print("  - Entity: ✓ spaCy active")
    sys.exit(0)
else:
    print("✗ Some NLU modules NOT loaded or using fallback:")
    if not engine.nlu_status["coref_loaded"]:
        print("  - Coref: ✗ using rule fallback")
    if not engine.nlu_status["intent_model_loaded"]:
        print("  - Intent: ✗ using rule fallback")
    if engine.nlu_status["intent_backend"] == "rule_fallback":
        print("  - Intent: ✗ backend is rule_fallback (distilbert not loaded)")
    if not engine.nlu_status["entity_model_loaded"]:
        print("  - Entity: ✗ using regex fallback")
    sys.exit(1)
