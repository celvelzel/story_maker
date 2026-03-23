"""Visual consistency rules tests for StoryWeaver UI.

StoryWeaver 视觉一致性规则测试。
验证主题令牌的使用和样式注入的一致性。
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


# ── Theme token validation tests ────────────────────────────────────────────

def test_theme_tokens_import():
    """Test that theme tokens can be imported.
    
    测试主题令牌可以导入。
    """
    from src.ui.theme_tokens import DARK_TOKENS, get_tokens
    
    assert DARK_TOKENS is not None
    assert len(DARK_TOKENS) > 0
    
    # Test get_tokens function
    tokens = get_tokens("dark")
    assert tokens is not None
    assert "primary" in tokens


def test_theme_tokens_structure():
    """Test theme tokens have expected structure.
    
    测试主题令牌具有预期的结构。
    """
    from src.ui.theme_tokens import DARK_TOKENS
    
    # Check required keys exist
    assert "primary" in DARK_TOKENS
    assert "bg" in DARK_TOKENS
    assert "text" in DARK_TOKENS
    assert "muted" in DARK_TOKENS


def test_style_injector_generates_css():
    """Test that style injector generates valid CSS.
    
    测试样式注入器生成有效的 CSS。
    """
    from src.ui.style_injector import inject_styles, _build_css
    
    # Test CSS generation
    from src.ui.theme_tokens import get_tokens
    tokens = get_tokens("dark")
    css = _build_css(tokens)
    assert css is not None
    assert len(css) > 0
    assert "background" in css.lower() or "color" in css.lower()


def test_style_injector_reduced_motion():
    """Test that style injector includes reduced motion support.
    
    测试样式注入器包含减弱动画支持。
    """
    from src.ui.style_injector import _build_css
    from src.ui.theme_tokens import get_tokens
    
    tokens = get_tokens("dark")
    css = _build_css(tokens)
    
    # Should include prefers-reduced-motion media query
    assert "prefers-reduced-motion" in css


# ── App integration tests ───────────────────────────────────────────────────

def test_app_uses_theme_injection():
    """Test that app.py uses theme injection.
    
    测试 app.py 使用主题注入。
    """
    # Read app.py and check for inject_styles import
    with open("app.py", "r", encoding="utf-8") as f:
        app_content = f.read()
    
    assert "from src.ui.style_injector import inject_styles" in app_content
    assert "inject_styles(" in app_content


def test_app_uses_feedback_helpers():
    """Test that app.py uses feedback helpers.
    
    测试 app.py 使用反馈助手。
    """
    # Check that layout/section modules import feedback helpers
    with open("src/ui/layout/sidebar_view.py", "r", encoding="utf-8") as f:
        sidebar_content = f.read()
    
    with open("src/ui/layout/main_view.py", "r", encoding="utf-8") as f:
        main_content = f.read()
    
    # At least one module should import feedback helpers
    has_feedback_import = (
        "from src.ui.feedback import" in sidebar_content or
        "from src.ui.feedback import" in main_content
    )
    assert has_feedback_import, "No feedback imports found in layout modules"


def test_app_uses_session_contract():
    """Test that app.py uses session contract.
    
    测试 app.py 使用会话契约。
    """
    with open("app.py", "r", encoding="utf-8") as f:
        app_content = f.read()
    
    # Should import session contract
    assert "from src.ui.session_contract import" in app_content
    assert "initialize_session()" in app_content


# ── Component structure tests ───────────────────────────────────────────────

def test_layout_components_exist():
    """Test that layout components exist.
    
    测试布局组件存在。
    """
    import os
    
    assert os.path.exists("src/ui/layout/__init__.py")
    assert os.path.exists("src/ui/layout/sidebar_view.py")
    assert os.path.exists("src/ui/layout/main_view.py")


def test_section_components_exist():
    """Test that section components exist.
    
    测试区域组件存在。
    """
    import os
    
    assert os.path.exists("src/ui/sections/__init__.py")
    assert os.path.exists("src/ui/sections/chat_section.py")
    assert os.path.exists("src/ui/sections/kg_section.py")
    assert os.path.exists("src/ui/sections/evaluation_section.py")


def test_feedback_module_exists():
    """Test that feedback module exists.
    
    测试反馈模块存在。
    """
    import os
    
    assert os.path.exists("src/ui/feedback.py")


def test_session_contract_module_exists():
    """Test that session contract module exists.
    
    测试会话契约模块存在。
    """
    import os
    
    assert os.path.exists("src/ui/session_contract.py")


# ── Inline style detection tests ────────────────────────────────────────────

def test_no_large_inline_styles_in_app():
    """Test that app.py doesn't have large inline style blocks.
    
    测试 app.py 没有大型内联样式块。
    """
    with open("app.py", "r", encoding="utf-8") as f:
        app_content = f.read()
    
    # Count lines with style-related content
    style_lines = [
        line for line in app_content.split("\n")
        if "unsafe_allow_html=True" in line and "<style>" in line
    ]
    
    # Should have minimal or no inline style blocks
    assert len(style_lines) <= 2, f"Found {len(style_lines)} inline style blocks, expected <= 2"


def test_feedback_helpers_used_consistently():
    """Test that feedback helpers are used instead of raw st.warning/st.info.
    
    测试使用反馈助手而不是原始的 st.warning/st.info。
    """
    with open("app.py", "r", encoding="utf-8") as f:
        app_content = f.read()
    
    # Count raw st.warning/st.info calls (excluding comments)
    lines = app_content.split("\n")
    raw_feedback_calls = [
        line for line in lines
        if (
            ("st.warning(" in line or "st.info(" in line or "st.error(" in line)
            and not line.strip().startswith("#")
            and "show_warning" not in line
            and "show_info" not in line
            and "show_error" not in line
        )
    ]
    
    # Should have minimal raw feedback calls
    assert len(raw_feedback_calls) <= 2, f"Found {len(raw_feedback_calls)} raw feedback calls, expected <= 2"
