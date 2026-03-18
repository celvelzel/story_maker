"""Game state data structures for StoryWeaver (hybrid architecture)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GameState:
    """
    GameState Class Documentation
    A minimal game state class that manages the narrative progression of an interactive story game.
    This class maintains turn information, genre context, and the complete history of exchanges
    between the player and the narrator.
    Attributes:
        turn_id (int): Counter tracking the current turn number in the game. Increments after each narrator action.
            Default: 0
        genre (str): The genre or theme of the story (e.g., "fantasy", "sci-fi", "mystery").
            Influences the style and context of narration.
            Default: "fantasy"
        story_history (List[Dict[str, str]]): Complete record of all story exchanges.
            Each entry is a dictionary containing:
            - "role": Either "player" (for player input) or "narrator" (for story narration)
            - "text": The actual content of that exchange
            Default: Empty list
    Methods:
        add_player_input(text: str) -> None:
            Appends a player's input to the story history.
            Args:
                text (str): The player's input text to be added to the story.
        add_narration(text: str) -> None:
            Appends narrator's response to the story history and increments the turn counter.
            This method should be called after processing the narrator's action to advance the game state.
            Args:
                text (str): The narrator's response text to be added to the story.
        recent_history(n: int = 6) -> str:
            Retrieves and formats the last n entries from story history for LLM context.
            This is useful for providing the language model with recent story context.
            Args:
                n (int): Number of recent entries to retrieve. Default: 6
            Returns:
                str: Formatted string with entries separated by newlines, 
                     each prefixed with "[Player]" or "[Narrator]" tag.
    """
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
