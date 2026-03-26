# Fix: Dashboard metrics crash when engine is fresh

## Problem
When clicking "Start New Game", the sidebar crashes because it tries to access `engine.kg_entity_names` and `engine.turn_conflict_counts` before these properties are properly initialized.

## Root Cause
In `sidebar.py` lines 153-154:
```python
entity_count = len(engine.kg_entity_names) if engine else 0
conflict_total = sum(engine.turn_conflict_counts) if engine else 0
```

Even though `engine` is not None, its properties `kg_entity_names` and `turn_conflict_counts` may not be accessible yet, causing an AttributeError that crashes the entire sidebar.

## Fix Required
Wrap the dashboard metric access in try-except blocks in `src/ui/sections/sidebar.py`.

## Changes
Edit `src/ui/sections/sidebar.py` lines 148-165:

Replace:
```python
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='section-title'>&#x1F39B;&#xFE0F; Dashboard</div>", unsafe_allow_html=True)
        turn_count = _story_turn_count()
        engine: GameEngine | None = st.session_state.engine
        entity_count = len(engine.kg_entity_names) if engine else 0
        conflict_total = sum(engine.turn_conflict_counts) if engine else 0
        ...
```

With:
```python
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='section-title'>&#x1F39B;&#xFE0F; Dashboard</div>", unsafe_allow_html=True)
        turn_count = _story_turn_count()
        engine: GameEngine | None = st.session_state.engine
        
        # Safely get entity count and conflict count from engine
        try:
            entity_count = len(engine.kg_entity_names) if engine else 0
        except (AttributeError, TypeError):
            entity_count = 0
        try:
            conflict_total = sum(engine.turn_conflict_counts) if engine else 0
        except (AttributeError, TypeError):
            conflict_total = 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Turns", turn_count)
        c2.metric("Entities", entity_count)
        c3.metric("Conflicts", conflict_total)

        if engine:
            try:
                st.caption(
                    f"Strategy: {engine.conflict_resolution} | {engine.extraction_mode} | "
                    f"{engine.summary_mode} | {engine.importance_mode}"
                )
            except (AttributeError, TypeError):
                pass
```

## Verification
After fix:
1. Start new game → Dashboard shows Turns=0, Entities=0, Conflicts=0
2. Make a choice → Dashboard updates correctly
3. No errors in console