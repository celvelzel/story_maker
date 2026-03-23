"""Theme tokens for StoryWeaver cyberpunk dark style.

StoryWeaver 赛博朋克暗色主题令牌。
定义颜色、间距、排版、阴影等视觉系统常量。

Usage:
    from src.ui.theme_tokens import get_tokens, DARK_TOKENS
    
    tokens = get_tokens("dark")
    # or
    tokens = DARK_TOKENS
"""
from __future__ import annotations

from typing import Literal

# Type alias for theme modes
ThemeMode = Literal["dark", "light"]

# ── Dark Theme Tokens (Cyberpunk Neon) ──────────────────────────────────

DARK_TOKENS: dict[str, str] = {
    # Base colors
    "bg": "#06080f",
    "text": "#e0e8ff",
    "muted": "#7b8db5",
    
    # Hero/brand colors
    "hero1": "#00f0ff",
    "hero2": "#7b2fff",
    "hero3": "#ff00aa",
    "hero_shadow": "rgba(0, 240, 255, 0.18)",
    
    # Panel/container colors
    "panel": "rgba(8, 12, 28, 0.88)",
    "panel_border": "rgba(0, 240, 255, 0.12)",
    
    # Title/heading
    "title": "#00f0ff",
    
    # Primary action colors
    "primary": "#00f0ff",
    "primary_hover": "#7b2fff",
    
    # Input field colors
    "input_bg": "rgba(6, 10, 24, 0.92)",
    "input_border": "rgba(0, 240, 255, 0.22)",
    
    # Chat area
    "chat_bg": "rgba(0, 240, 255, 0.03)",
    
    # Neon accent colors
    "neon_cyan": "#00f0ff",
    "neon_magenta": "#ff00aa",
    "neon_purple": "#7b2fff",
    "neon_gold": "#ffd700",
    
    # Semantic state colors (for feedback)
    "success": "#00ff88",
    "warning": "#ffd700",
    "error": "#ff4444",
    "info": "#00f0ff",
    
    # Surface levels (for hierarchy)
    "surface_1": "rgba(8, 12, 28, 0.88)",
    "surface_2": "rgba(12, 18, 36, 0.92)",
    "surface_3": "rgba(16, 24, 48, 0.95)",
    
    # Spacing scale (in px)
    "space_xs": "4px",
    "space_sm": "8px",
    "space_md": "16px",
    "space_lg": "24px",
    "space_xl": "32px",
    
    # Border radius
    "radius_sm": "4px",
    "radius_md": "8px",
    "radius_lg": "16px",
    "radius_xl": "24px",
    
    # Shadows
    "shadow_sm": "0 0 6px rgba(0, 240, 255, 0.15)",
    "shadow_md": "0 0 12px rgba(0, 240, 255, 0.25)",
    "shadow_lg": "0 0 24px rgba(0, 240, 255, 0.35)",
    "shadow_glow": "0 0 8px rgba(0, 240, 255, 0.3), 0 0 20px rgba(0, 240, 255, 0.1)",
}

# ── Light Theme Tokens (Fallback — still cyberpunk-tinted) ──────────────

LIGHT_TOKENS: dict[str, str] = {
    # Base colors
    "bg": "#06080f",
    "text": "#e0e8ff",
    "muted": "#7b8db5",
    
    # Hero/brand colors
    "hero1": "#00f0ff",
    "hero2": "#7b2fff",
    "hero3": "#ff00aa",
    "hero_shadow": "rgba(0, 240, 255, 0.18)",
    
    # Panel/container colors
    "panel": "rgba(8, 12, 28, 0.88)",
    "panel_border": "rgba(0, 240, 255, 0.12)",
    
    # Title/heading
    "title": "#00f0ff",
    
    # Primary action colors
    "primary": "#00f0ff",
    "primary_hover": "#7b2fff",
    
    # Input field colors
    "input_bg": "rgba(6, 10, 24, 0.92)",
    "input_border": "rgba(0, 240, 255, 0.22)",
    
    # Chat area
    "chat_bg": "rgba(0, 240, 255, 0.03)",
    
    # Neon accent colors
    "neon_cyan": "#00f0ff",
    "neon_magenta": "#ff00aa",
    "neon_purple": "#7b2fff",
    "neon_gold": "#ffd700",
    
    # Semantic state colors (for feedback)
    "success": "#00ff88",
    "warning": "#ffd700",
    "error": "#ff4444",
    "info": "#00f0ff",
    
    # Surface levels (for hierarchy)
    "surface_1": "rgba(8, 12, 28, 0.88)",
    "surface_2": "rgba(12, 18, 36, 0.92)",
    "surface_3": "rgba(16, 24, 48, 0.95)",
    
    # Spacing scale (in px)
    "space_xs": "4px",
    "space_sm": "8px",
    "space_md": "16px",
    "space_lg": "24px",
    "space_xl": "32px",
    
    # Border radius
    "radius_sm": "4px",
    "radius_md": "8px",
    "radius_lg": "16px",
    "radius_xl": "24px",
    
    # Shadows
    "shadow_sm": "0 0 6px rgba(0, 240, 255, 0.15)",
    "shadow_md": "0 0 12px rgba(0, 240, 255, 0.25)",
    "shadow_lg": "0 0 24px rgba(0, 240, 255, 0.35)",
    "shadow_glow": "0 0 8px rgba(0, 240, 255, 0.3), 0 0 20px rgba(0, 240, 255, 0.1)",
}


def get_tokens(mode: ThemeMode = "dark") -> dict[str, str]:
    """Return theme tokens for the specified mode.
    
    返回指定模式的主题令牌。
    
    Args:
        mode: Theme mode ("dark" or "light")
        
    Returns:
        Dictionary of token name → value mappings
    """
    if mode == "dark":
        return DARK_TOKENS.copy()
    return LIGHT_TOKENS.copy()


def get_token(mode: ThemeMode, key: str, fallback: str = "") -> str:
    """Get a single token value by key.
    
    通过键获取单个令牌值。
    
    Args:
        mode: Theme mode
        key: Token key name
        fallback: Fallback value if key not found
        
    Returns:
        Token value or fallback
    """
    tokens = get_tokens(mode)
    return tokens.get(key, fallback)
