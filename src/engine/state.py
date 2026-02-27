"""Game state data structures for StoryWeaver (hybrid architecture)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GameState:
    """Minimal game state carried across turns."""

    turn_id: int = 0
    genre: str = "fantasy"
    story_history: List[Dict[str, str]] = field(default_factory=list)
    # Each entry: {"role": "player"|"narrator", "text": "..."}

    def add_player_input(self, text: str) -> None:
        self.story_history.append({"role": "player", "text": text})

    def add_narration(self, text: str) -> None:
        self.story_history.append({"role": "narrator", "text": text})
        self.turn_id += 1

    def recent_history(self, n: int = 6) -> str:
        """Return the last *n* entries formatted for LLM context."""
        entries = self.story_history[-n:]
        parts: List[str] = []
        for e in entries:
            prefix = "Player" if e["role"] == "player" else "Narrator"
            parts.append(f"[{prefix}] {e['text']}")
        return "\n".join(parts)
