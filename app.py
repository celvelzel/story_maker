"""StoryWeaver – Gradio chat-based text adventure UI.

Layout (gr.Blocks):
  Left column (3/5):  chat history  +  option Radio  +  free-text input
  Right column (2/5): KG HTML panel  +  NLU debug accordion
"""
from __future__ import annotations

import os
import sys
import logging

import gradio as gr

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.nlg.option_generator import StoryOption

logger = logging.getLogger(__name__)

# ── Global engine (lazy, one per process) ────────────────────────────────
_engine: GameEngine | None = None


def _get_engine(genre: str = "fantasy") -> GameEngine:
    global _engine
    if _engine is None:
        _engine = GameEngine(genre=genre)
    return _engine


# ── Helpers ──────────────────────────────────────────────────────────────

def _options_to_choices(options: list[StoryOption]) -> list[str]:
    return [f"{i+1}. {o.text}" for i, o in enumerate(options)]


def _format_nlu(nlu: dict) -> str:
    if not nlu:
        return ""
    lines = [
        f"**Resolved input:** {nlu.get('resolved_input', '')}",
        f"**Intent:** {nlu.get('intent', '?')}  (conf {nlu.get('confidence', 0):.2f})",
        f"**Entities:** {nlu.get('entities', [])}",
    ]
    return "\n".join(lines)


# ── Callbacks ────────────────────────────────────────────────────────────

def start_game(genre: str, history: list):
    engine = _get_engine(genre or "fantasy")
    result: TurnResult = engine.start_game()

    history = history or []
    history.append({"role": "assistant", "content": result.story_text})

    choices = _options_to_choices(result.options)
    return (
        history,
        gr.update(choices=choices, value=None, visible=True),  # option radio
        result.kg_html,
        "",  # nlu debug (no NLU on opening)
    )


def submit_action(user_text: str, selected_option: str | None, history: list):
    # Prefer typed text; fall back to selected option
    action = (user_text or "").strip()
    if not action and selected_option:
        # Strip the leading "1. " prefix
        action = selected_option.split(". ", 1)[-1] if ". " in selected_option else selected_option
    if not action:
        return history, gr.update(), "", ""

    engine = _get_engine()
    history = history or []
    history.append({"role": "user", "content": action})

    result: TurnResult = engine.process_turn(action)

    assistant_msg = result.story_text
    if result.conflicts:
        assistant_msg += "\n\n*⚠ World-consistency notes:*\n" + "\n".join(
            f"- {c}" for c in result.conflicts
        )
    history.append({"role": "assistant", "content": assistant_msg})

    choices = _options_to_choices(result.options)
    return (
        history,
        gr.update(choices=choices, value=None, visible=True),
        result.kg_html,
        _format_nlu(result.nlu_debug),
    )


# ── UI Layout ────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="StoryWeaver – AI Text Adventure",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=".kg-panel { border:1px solid #444; border-radius:8px; padding:8px; min-height:300px; }",
    ) as demo:
        gr.Markdown("# StoryWeaver: AI-Powered Text Adventure\n*Hybrid NLU + LLM story engine with live knowledge-graph tracking*")

        with gr.Row():
            genre_box = gr.Textbox(value="fantasy", label="Genre", scale=2)
            new_game_btn = gr.Button("New Game", variant="primary", scale=1)

        with gr.Row():
            # ── Left column ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Story", type="messages", height=480)
                option_radio = gr.Radio(
                    choices=[], label="Choose an option", interactive=True, visible=False,
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Or type a free-form action…",
                        label="Your action", scale=4, lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # ── Right column ──
            with gr.Column(scale=2):
                kg_html = gr.HTML(label="Knowledge Graph", elem_classes="kg-panel")
                with gr.Accordion("NLU Debug", open=False):
                    nlu_md = gr.Markdown("")

        # ── Wiring ──
        new_game_btn.click(
            fn=start_game,
            inputs=[genre_box, chatbot],
            outputs=[chatbot, option_radio, kg_html, nlu_md],
        )

        send_btn.click(
            fn=submit_action,
            inputs=[user_input, option_radio, chatbot],
            outputs=[chatbot, option_radio, kg_html, nlu_md],
        ).then(fn=lambda: "", outputs=user_input)

        user_input.submit(
            fn=submit_action,
            inputs=[user_input, option_radio, chatbot],
            outputs=[chatbot, option_radio, kg_html, nlu_md],
        ).then(fn=lambda: "", outputs=user_input)

        option_radio.input(
            fn=submit_action,
            inputs=[user_input, option_radio, chatbot],
            outputs=[chatbot, option_radio, kg_html, nlu_md],
        ).then(fn=lambda: "", outputs=user_input)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
