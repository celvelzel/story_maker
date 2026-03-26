"""Debug script to simulate Streamlit Start New Game flow"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock Streamlit session state
class MockSessionState:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattr__(self, name)
        return self._data.get(name, None)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
    
    def __contains__(self, key):
        return key in self._data

# Create mock session state
mock_state = MockSessionState()

# Initialize with defaults from state_manager.py
defaults = {
    "engine": None,
    "history": [],
    "consistency_history": [],
    "kg_html": "",
    "options": [],
    "nlu_debug": {},
    "processing": False,
    "kg_conflict_resolution": "warn",
    "kg_extraction_mode": "dual",
    "kg_importance_mode": "decay",
    "kg_summary_mode": "rolling",
}

for key, value in defaults.items():
    mock_state[key] = value

print("=" * 60)
print("SIMULATING START NEW GAME FLOW")
print("=" * 60)

print(f"\n1. Initial state:")
print(f"   - engine: {mock_state.engine}")
print(f"   - kg_html: '{mock_state.kg_html}' (len={len(mock_state.kg_html)})")
print(f"   - kg_html is truthy: {bool(mock_state.kg_html)}")

# Simulate Start New Game
print(f"\n2. Creating GameEngine...")
from src.engine.game_engine import GameEngine

engine = GameEngine(genre="fantasy")
mock_state.engine = engine
print(f"   - engine created: {engine is not None}")

# Try to start game (this will fail due to API issue, but we can check the flow)
print(f"\n3. Attempting to start game...")
try:
    result = engine.start_game()
    print(f"   - Game started successfully!")
    print(f"   - result.kg_html length: {len(result.kg_html)}")
    print(f"   - result.kg_html is truthy: {bool(result.kg_html)}")
    
    # Simulate app.py lines 225-229
    mock_state.kg_html = result.kg_html
    print(f"\n4. After setting kg_html:")
    print(f"   - kg_html length: {len(mock_state.kg_html)}")
    print(f"   - kg_html is truthy: {bool(mock_state.kg_html)}")
    
    # Simulate sidebar.py rendering logic
    print(f"\n5. Simulating sidebar rendering:")
    print(f"   - Condition 'kg_html or engine': {bool(mock_state.kg_html or mock_state.engine)}")
    print(f"   - Condition 'kg_html': {bool(mock_state.kg_html)}")
    
    if mock_state.kg_html or mock_state.engine:
        if mock_state.kg_html:
            print(f"   -> Should render cached kg_html")
        elif mock_state.engine:
            print(f"   -> Should render kg_html from engine (fallback path)")
    else:
        print(f"   -> Should show fallback message")
    
except Exception as e:
    print(f"   - FAILED: {e}")
    print(f"\n4. Checking fallback rendering path:")
    print(f"   - engine exists: {mock_state.engine is not None}")
    print(f"   - kg_html is empty: {not mock_state.kg_html}")
    
    # Simulate sidebar.py rendering logic with empty kg_html
    print(f"\n5. Simulating sidebar rendering (with empty kg_html):")
    print(f"   - Condition 'kg_html or engine': {bool(mock_state.kg_html or mock_state.engine)}")
    print(f"   - Condition 'kg_html': {bool(mock_state.kg_html)}")
    
    if mock_state.kg_html or mock_state.engine:
        if mock_state.kg_html:
            print(f"   -> Should render cached kg_html")
        elif mock_state.engine:
            print(f"   -> Should render kg_html from engine (fallback path)")
            # Try to render
            try:
                from src.knowledge_graph.visualizer import render_kg_html
                kg_html = render_kg_html(mock_state.engine.kg.graph)
                print(f"      - render_kg_html result length: {len(kg_html)}")
                print(f"      - render_kg_html is truthy: {bool(kg_html)}")
                mock_state.kg_html = kg_html
                print(f"      - After caching: kg_html length={len(mock_state.kg_html)}")
            except Exception as e2:
                print(f"      - render_kg_html FAILED: {e2}")
    else:
        print(f"   -> Should show fallback message")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
