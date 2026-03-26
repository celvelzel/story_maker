"""Test Start New Game flow"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TESTING START NEW GAME FLOW")
print("=" * 60)

# Test 1: Create engine and start game
print("\n1. Creating GameEngine and starting game...")
from src.engine.game_engine import GameEngine

try:
    engine = GameEngine(genre="fantasy")
    print(f"   - Engine created: {engine is not None}")
    
    result = engine.start_game()
    print(f"   - Game started successfully!")
    print(f"   - story_text length: {len(result.story_text)}")
    print(f"   - options count: {len(result.options)}")
    print(f"   - kg_html length: {len(result.kg_html)}")
    print(f"   - kg_html is truthy: {bool(result.kg_html)}")
    
    # Test 2: Check kg_html content
    print(f"\n2. Checking kg_html content...")
    print(f"   - First 200 chars: {result.kg_html[:200]}")
    
    # Test 3: Simulate sidebar rendering
    print(f"\n3. Simulating sidebar rendering...")
    kg_html = result.kg_html
    
    if kg_html or engine:
        print(f"   - Outer condition: TRUE")
        if kg_html:
            print(f"   -> Would render cached kg_html")
        elif engine:
            print(f"   -> Would render kg_html from engine (fallback)")
    else:
        print(f"   - Outer condition: FALSE")
        print(f"   -> Would show fallback message")
    
    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[FAIL] Test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("TEST FAILED")
    print("=" * 60)
