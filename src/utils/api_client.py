"""Singleton OpenAI wrapper with retry, JSON mode, and cost tracking.

OpenAI API 单例封装器模块。

提供以下核心功能：
- ``chat()``：发送聊天补全请求，返回纯文本响应
- ``chat_json()``：发送聊天补全请求，返回 JSON 模式解析后的字典
- 自动重试机制：遇到临时性错误时自动重试最多 3 次（指数退避策略）
- 会话级用量追踪：记录输入/输出 token 数量和 USD 成本

使用单例模式确保整个应用共享同一个 API 客户端实例。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pricing per 1 M tokens for gpt-4o-mini (as of 2025-06)
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class LLMClient:
    """OpenAI 聊天补全单例封装器。

    核心功能：
    - ``chat()``：发送聊天补全请求，返回纯文本响应
    - ``chat_json()``：发送聊天补全请求，返回 JSON 模式解析后的字典
    - 自动重试机制：遇到临时性错误时自动重试最多 3 次（指数退避策略）
    - 会话级用量追踪：记录输入/输出 token 数量和 USD 成本

    使用单例模式确保整个应用共享同一个 API 客户端实例。
    采用懒加载方式初始化 OpenAI 客户端，只在首次使用时创建。

    属性：
        _instance: 类级别的单例实例
        _initialised: 是否已完成初始化标志
        _client: OpenAI 客户端实例（懒加载）
        _total_input_tokens: 会话内累计输入 token 数
        _total_output_tokens: 会话内累计输出 token 数
    """

    _instance: Optional["LLMClient"] = None

    def __new__(cls) -> "LLMClient":
        """创建或返回单例实例。

        确保整个应用只有一个 LLMClient 实例。

        返回：
            LLMClient: 单例实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        """初始化客户端配置。

        仅在首次创建时执行初始化，后续调用会直接返回。
        加载配置、初始化 token 计数器。
        """
        if self._initialised:
            return
        from config import settings

        self._settings = settings
        self._client: Any = None  # OpenAI 客户端实例（懒加载）
        self._total_input_tokens: int = 0  # 累计输入 token 数
        self._total_output_tokens: int = 0  # 累计输出 token 数
        self._initialised = True

    # ── lazy OpenAI client ────────────────────────────────
    @property
    def client(self) -> Any:
        """懒加载的 OpenAI 客户端属性。

        首次访问时创建 OpenAI 客户端实例，支持自定义 base_url
        以兼容 OpenAI 兼容的第三方服务。

        返回：
            OpenAI: 已配置的 OpenAI 客户端实例

        异常：
            Exception: 如果创建客户端失败，抛出异常
        """
        if self._client is None:
            try:
                from openai import OpenAI
                # 构建客户端参数
                kwargs: Dict[str, Any] = {"api_key": self._settings.OPENAI_API_KEY}
                # 如果配置了自定义 base_url，使用它（支持第三方 OpenAI 兼容服务）
                if self._settings.OPENAI_BASE_URL:
                    kwargs["base_url"] = self._settings.OPENAI_BASE_URL
                self._client = OpenAI(**kwargs)
            except Exception as exc:
                logger.error("创建 OpenAI 客户端失败: %s", exc)
                raise
        return self._client

    # ── public API ────────────────────────────────────────
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """发送聊天补全请求并返回助手消息文本。

        发送消息列表到 OpenAI API，支持配置温度和最大 token 数。
        遇到临时性错误时自动重试最多 3 次（指数退避策略）。

        参数：
            messages: 消息列表，每条消息包含 "role" 和 "content"
            temperature: 生成温度参数，控制随机性（默认使用配置值）
            max_tokens: 最大生成 token 数（默认使用配置值）
            json_mode: 是否使用 JSON 响应模式

        返回：
            str: 助手的回复文本

        异常：
            RuntimeError: 3 次重试全部失败时抛出
        """
        # 使用配置默认值或显式传入的值
        temperature = temperature if temperature is not None else self._settings.OPENAI_TEMPERATURE
        max_tokens = max_tokens or self._settings.OPENAI_MAX_TOKENS

        # 构建 API 请求参数
        kwargs: Dict[str, Any] = {
            "model": self._settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # JSON 模式：要求模型返回合法的 JSON 对象
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # 重试循环：最多 3 次，指数退避
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(**kwargs)
                # 追踪 token 使用量
                usage = response.usage
                if usage:
                    self._total_input_tokens += usage.prompt_tokens
                    self._total_output_tokens += usage.completion_tokens
                # 返回助手的回复内容
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                # 指数退避：等待 2^attempt 秒
                wait = 2 ** attempt
                logger.warning("LLM 调用第 %d 次失败 (%s)。%d 秒后重试…", attempt, exc, wait)
                time.sleep(wait)

        # 所有重试均失败
        raise RuntimeError(f"LLM 调用 3 次重试全部失败: {last_exc}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """发送聊天补全请求并返回解析后的 JSON 字典。

        与 ``chat()`` 类似，但自动启用 JSON 模式并将响应解析为字典。
        适用于需要结构化输出的场景（如知识图谱提取、选项生成等）。

        参数：
            messages: 消息列表
            temperature: 生成温度参数
            max_tokens: 最大生成 token 数

        返回：
            Dict[str, Any]: 解析后的 JSON 响应字典

        异常：
            json.JSONDecodeError: 如果响应不是有效 JSON
        """
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        return json.loads(raw)

    # ── cost tracking ─────────────────────────────────────
    @property
    def total_input_tokens(self) -> int:
        """获取会话内累计输入 token 数量。

        返回：
            int: 累计输入 token 数
        """
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """获取会话内累计输出 token 数量。

        返回：
            int: 累计输出 token 数
        """
        return self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        """计算会话内累计 API 调用成本（USD）。

        基于配置的模型价格和实际使用的 token 数量计算成本。
        默认使用 gpt-4o-mini 的定价。

        返回：
            float: 累计成本（美元）
        """
        # 获取当前模型的价格表，默认使用 gpt-4o-mini
        pricing = _PRICING.get(self._settings.OPENAI_MODEL, _PRICING["gpt-4o-mini"])
        # 计算总成本：输入 token 费用 + 输出 token 费用
        return (
            self._total_input_tokens * pricing["input"] / 1_000_000
            + self._total_output_tokens * pricing["output"] / 1_000_000
        )

    def reset_cost(self) -> None:
        """重置 token 计数器和成本追踪。

        将输入/输出 token 计数归零，用于开始新的计费周期。
        """
        self._total_input_tokens = 0
        self._total_output_tokens = 0


# Convenience module-level singleton
llm_client = LLMClient()
