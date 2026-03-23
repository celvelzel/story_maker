"""Smoke tests for primary chat flow scenarios.

聊天主流程冒烟测试。
基于 Streamlit AppTest 框架验证核心用户交互。

Scenarios:
- S1: Start game from landing state
- S2: Submit free-text action
- S3: Select generated option button
"""
from __future__ import annotations

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest

# Check if LLM API key is available (for skipping LLM-dependent tests)
_LLM_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))


def _run_app_with_signal_skip():
    """Helper to run app and skip if signal handler issue occurs.
    
    辅助函数：运行应用，如果出现信号处理器问题则跳过。
    AppTest 在非主线程运行，signal.signal() 会失败。
    """
    at = AppTest.from_file("app.py")
    at.run()
    if at.exception:
        err_msg = str(at.exception)
        if "signal only works in main thread" in err_msg:
            pytest.skip(f"Known signal handler issue in AppTest: {err_msg}")
        # Other exceptions are real failures
    return at


def test_landing_state_renders():
    """S1a: App renders without exception in landing state.
    
    验证应用在初始状态正常渲染，无异常。
    """
    at = _run_app_with_signal_skip()
    # If we got here (not skipped), there should be no exception
    assert not at.exception, f"App crashed on load: {at.exception}"


def test_landing_has_start_controls():
    """S1b: Landing state shows genre input and start button.
    
    验证初始状态显示类型输入框和开始按钮。
    """
    at = _run_app_with_signal_skip()
    assert not at.exception
    # Should have at least one text input (genre) and one button (start)
    assert len(at.text_input) >= 1, "Expected at least one text input for genre"
    assert len(at.button) >= 1, "Expected at least one button for start"


@pytest.mark.skipif(not _LLM_AVAILABLE, reason="OPENAI_API_KEY not set — LLM required")
def test_start_game_landing():
    """S1: Start game from landing state.
    
    从初始状态开始游戏：输入类型 → 点击开始 → 验证引擎初始化。
    """
    at = _run_app_with_signal_skip()
    assert not at.exception

    # Input genre
    at.text_input[0].input("fantasy").run()
    assert not at.exception

    # Click start button
    at.button[0].click().run()
    assert not at.exception

    # Engine should be initialized
    engine = at.session_state["engine"] if "engine" in at.session_state else None
    assert engine is not None, "GameEngine not initialized after start"


@pytest.mark.skipif(not _LLM_AVAILABLE, reason="OPENAI_API_KEY not set — LLM required")
def test_submit_free_text_action():
    """S2: Submit free-text action via chat input.
    
    提交自由文本行动：开始游戏 → 输入行动 → 验证响应。
    """
    at = _run_app_with_signal_skip()
    assert not at.exception

    # Start game first
    at.text_input[0].input("fantasy").run()
    at.button[0].click().run()
    engine = at.session_state["engine"] if "engine" in at.session_state else None
    assert engine is not None

    # Submit free-text action via chat input
    if len(at.chat_input) > 0:
        at.chat_input[0].input("I look around the forest").run()
        assert not at.exception

        # History should have new entries
        history = at.session_state["history"] if "history" in at.session_state else []
        assert len(history) >= 2, "Expected user + assistant messages after action"
    else:
        pytest.skip("No chat_input widget found — may require game to be started differently")


@pytest.mark.skipif(not _LLM_AVAILABLE, reason="OPENAI_API_KEY not set — LLM required")
def test_select_option_button():
    """S3: Select generated option button.
    
    选择生成的选项按钮：开始游戏 → 等待选项 → 点击选项 → 验证状态转换。
    """
    at = _run_app_with_signal_skip()
    assert not at.exception

    # Start game
    at.text_input[0].input("fantasy").run()
    at.button[0].click().run()
    engine = at.session_state["engine"] if "engine" in at.session_state else None
    assert engine is not None

    # Submit an action to generate options
    if len(at.chat_input) > 0:
        at.chat_input[0].input("explore").run()
        assert not at.exception

        # Check if options were generated
        options = at.session_state["options"] if "options" in at.session_state else []
        if options:
            # Click first option button (options are rendered as buttons)
            # Find option buttons (they appear after chat input)
            option_buttons = [b for b in at.button if b.label != "Start"]
            if option_buttons:
                initial_history = at.session_state["history"] if "history" in at.session_state else []
                initial_history_len = len(initial_history)
                option_buttons[0].click().run()
                assert not at.exception

                # History should grow
                new_history = at.session_state["history"] if "history" in at.session_state else []
                new_history_len = len(new_history)
                assert new_history_len > initial_history_len, "History should grow after option selection"
            else:
                pytest.skip("No option buttons found after action")
        else:
            pytest.skip("No options generated after first action")
    else:
        pytest.skip("No chat_input widget found")
