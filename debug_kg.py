"""Debug script to check KG rendering"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine
from src.knowledge_graph.visualizer import render_kg_html

print("=" * 60)
print("KG RENDERING DEBUG")
print("=" * 60)

# Test 1: Check if PyVis is available
try:
    from pyvis.network import Network
    print("[PASS] PyVis is available")
except ImportError:
    print("[FAIL] PyVis is NOT available")

# Test 2: Create engine and start game
print("\nCreating GameEngine...")
engine = GameEngine(genre="fantasy")

print("Starting game...")
result = engine.start_game()

# Test 3: Check kg_html
print(f"\nkg_html length: {len(result.kg_html)}")
print(f"kg_html is empty: {not result.kg_html}")
print(f"kg_html first 200 chars: {result.kg_html[:200] if result.kg_html else 'EMPTY'}")

# Test 4: Check if render_kg_html works
print("\n\nTesting render_kg_html directly...")
kg_html = render_kg_html(engine.kg.graph)
print(f"render_kg_html result length: {len(kg_html)}")
print(f"render_kg_html is empty: {not kg_html}")
print(f"render_kg_html first 200 chars: {kg_html[:200] if kg_html else 'EMPTY'}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
