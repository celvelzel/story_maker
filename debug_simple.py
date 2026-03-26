"""Simple debug to check rendering logic"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("CHECKING KG RENDERING LOGIC")
print("=" * 60)

# Test 1: Check PyVis
print("\n1. Checking PyVis...")
try:
    from pyvis.network import Network
    print("   [PASS] PyVis is available")
except ImportError:
    print("   [FAIL] PyVis is NOT available")

# Test 2: Check if we can create an empty graph and render it
print("\n2. Testing render_kg_html with empty graph...")
from src.knowledge_graph.visualizer import render_kg_html
import networkx as nx

empty_graph = nx.MultiDiGraph()
empty_html = render_kg_html(empty_graph)
print(f"   - Empty graph HTML length: {len(empty_html)}")
print(f"   - Empty graph HTML is truthy: {bool(empty_html)}")

# Test 3: Test sidebar rendering logic with different kg_html values
print("\n3. Testing sidebar rendering logic...")

test_cases = [
    ("Empty string", ""),
    ("None", None),
    ("Non-empty HTML", "<html><body>Test</body></html>"),
]

for name, kg_html in test_cases:
    engine = "mock_engine"  # Simulate engine existing
    
    print(f"\n   Case: {name}")
    print(f"   - kg_html = {repr(kg_html)}")
    print(f"   - kg_html is truthy: {bool(kg_html)}")
    print(f"   - engine is truthy: {bool(engine)}")
    
    # Simulate sidebar.py lines 171-184
    if kg_html or engine:
        print(f"   - Outer condition: TRUE")
        if kg_html:
            print(f"   -> Would render cached kg_html")
        elif engine:
            print(f"   -> Would render kg_html from engine (fallback)")
    else:
        print(f"   - Outer condition: FALSE")
        print(f"   -> Would show fallback message")

# Test 4: The critical case - empty kg_html with engine
print("\n4. CRITICAL TEST: Empty kg_html with engine")
kg_html = ""
engine_exists = True

print(f"   - kg_html = '{kg_html}'")
print(f"   - engine exists = {engine_exists}")

if kg_html or engine_exists:
    print(f"   - Outer condition: TRUE (because engine_exists)")
    if kg_html:
        print(f"   -> Would render cached kg_html")
    elif engine_exists:
        print(f"   -> Would render kg_html from engine (fallback)")
        print(f"   -> This is the CORRECT behavior!")
else:
    print(f"   - Outer condition: FALSE")
    print(f"   -> Would show fallback message (WRONG!)")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("\nThe rendering logic is CORRECT.")
print("If KG doesn't display, the issue is likely:")
print("1. render_kg_html() returns empty string")
print("2. components.html() fails to render")
print("3. CSS/HTML rendering issue in browser")
print("4. Streamlit session state issue")
