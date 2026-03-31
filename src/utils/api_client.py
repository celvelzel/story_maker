"""Singleton OpenAI wrapper with retry, JSON mode, and usage tracking.

OpenAI API 单例封装器模块。支持混合模式（hybrid）：
- 创意任务（story）使用本地 Qwen3-4B 模型
- 结构化任务（option, relation）使用 Mimo v2 Flash API

提供以下核心功能：
- ``chat()``：发送聊天补全请求，返回纯文本响应
- ``chat_json()``：发送聊天补全请求，返回 JSON 模式解析后的字典
- ``get_client_for_task()``：根据任务类型和 NLG_MODE 返回合适的客户端
- 自动重试机制：遇到临时性错误时自动重试最多 3 次（指数退避策略）
- 会话级用量追踪：记录输入/输出 token 数量

使用单例模式确保整个应用共享同一个 API 客户端实例。
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Hybrid Mode Client Manager ────────────────────────────────
class HybridClientManager:
    """Manage multiple LLM clients for hybrid mode routing.
    
    在混合模式下管理多个 LLM 客户端：
    - 本地客户端：用于创意任务（story generation）
    - API 客户端：用于结构化任务（option, relation, JSON）
    """
    
    _instance: Optional["HybridClientManager"] = None
    _local_client: Optional[LLMClient] = None
    _api_client: Optional[LLMClient] = None
    
    def __new__(cls) -> "HybridClientManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_client_for_task(cls, task_type: str = "default") -> LLMClient:
        """Get the appropriate client based on NLG_MODE and task type.
        
        根据 NLG_MODE 和任务类型返回合适的客户端：
        - task_type="story" → 创意任务，使用本地客户端（如果可用）
        - task_type="option"/"relation"/"json" → 结构化任务，使用 API 客户端（如果可用）
        
        Args:
            task_type: Task type ("story", "option", "relation", "json", "default")
            
        Returns:
            LLMClient: The appropriate client for the given task
        """
        from config import settings
        
        mode = getattr(settings, "NLG_MODE", "api").lower()
        
        if mode == "local":
            # Always use local client
            return LLMClient(client_type="local")
        elif mode == "api":
            # Always use API client
            return LLMClient(client_type="api")
        elif mode == "hybrid":
            # Route based on task type
            if task_type == "story":
                # Creative tasks use local Qwen3
                return LLMClient(client_type="local")
            else:
                # Structured tasks (option, relation, json) use Mimo API
                return LLMClient(client_type="api")
        else:
            # Default to API for unknown modes
            logger.warning(f"Unknown NLG_MODE: {mode}. Defaulting to API client.")
            return LLMClient(client_type="api")


class LLMClient:
    """OpenAI 聊天补全单例封装器（支持混合模式）。

    核心功能：
    - ``chat()``：发送聊天补全请求，返回纯文本响应
    - ``chat_json()``：发送聊天补全请求，返回 JSON 模式解析后的字典
    - 自动重试机制：遇到临时性错误时自动重试最多 3 次（指数退避策略）
    - 会话级用量追踪：记录输入/输出 token 数量
    - 混合模式支持：可为本地客户端或 API 客户端

    使用单例模式确保整个应用共享同一个 API 客户端实例。
    采用懒加载方式初始化 OpenAI 客户端，只在首次使用时创建。

    属性：
        _instance: 类级别的单例实例
        _initialised: 是否已完成初始化标志
        _client: OpenAI 客户端实例（懒加载）
        _total_input_tokens: 会话内累计输入 token 数
        _total_output_tokens: 会话内累计输出 token 数
        _client_type: 客户端类型 ("local" 或 "api")
    """

    _instance: Optional["LLMClient"] = None
    _instances_by_type: Dict[str, "LLMClient"] = {}

    def __new__(cls, client_type: str = "api") -> "LLMClient":
        """创建或返回单例实例（支持多类型）。

        确保每种客户端类型都有一个单例实例。

        参数：
            client_type: "local" 或 "api"

        返回：
            LLMClient: 单例实例
        """
        client_type = client_type.lower()
        if client_type not in cls._instances_by_type:
            instance = super().__new__(cls)
            instance._initialised = False
            cls._instances_by_type[client_type] = instance
            if cls._instance is None:
                cls._instance = instance
        return cls._instances_by_type[client_type]

    def __init__(self, client_type: str = "api") -> None:
        """初始化客户端配置。

        仅在首次创建时执行初始化，后续调用会直接返回。
        加载配置、初始化 token 计数器。
        
        参数：
            client_type: "local" 使用 OPENAI_BASE_URL；"api" 使用 MIMO_* 设置
        """
        if self._initialised:
            return
        from config import settings

        self._settings = settings
        self._client_type = client_type.lower()
        self._client: Any = None
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        
        import httpx
        self._timeout = httpx.Timeout(
            self._settings.OPENAI_TIMEOUT_CONNECT, 
            read=self._settings.OPENAI_TIMEOUT_READ
        )
        self._initialised = True

    # ── lazy OpenAI client ────────────────────────────────
    @property
    def client(self) -> Any:
        """懒加载的 OpenAI 客户端属性。

        首次访问时创建 OpenAI 客户端实例，支持自定义 base_url
        以兼容 OpenAI 兼容的第三方服务。

        对于混合模式：
        - client_type="local": 使用 OPENAI_BASE_URL（本地 Qwen3 llama.cpp 服务器）
        - client_type="api": 使用 MIMO_* 设置（Mimo v2 Flash 云 API）

        返回：
            OpenAI: 已配置的 OpenAI 客户端实例

        异常：
            Exception: 如果创建客户端失败，抛出异常
        """
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs: Dict[str, Any] = {
                    "timeout": self._timeout,
                }
                
                if self._client_type == "local":
                    # Local Qwen3-4B via llama.cpp
                    kwargs["api_key"] = self._settings.OPENAI_API_KEY or "not-needed-for-local"
                    if self._settings.OPENAI_BASE_URL:
                        kwargs["base_url"] = self._settings.OPENAI_BASE_URL
                    logger.info(f"Creating local LLM client: {self._settings.OPENAI_BASE_URL}")
                else:
                    # API client: use MIMO_* settings if available, fallback to OPENAI_*
                    mimo_api_key = getattr(self._settings, "MIMO_API_KEY", None)
                    mimo_base_url = getattr(self._settings, "MIMO_BASE_URL", None)
                    
                    if mimo_api_key:
                        kwargs["api_key"] = mimo_api_key
                    else:
                        kwargs["api_key"] = self._settings.OPENAI_API_KEY
                    
                    if mimo_base_url:
                        kwargs["base_url"] = mimo_base_url
                    elif self._settings.OPENAI_BASE_URL:
                        kwargs["base_url"] = self._settings.OPENAI_BASE_URL
                    
                    logger.info(f"Creating API LLM client: {kwargs.get('base_url', 'default')}")
                
                self._client = OpenAI(**kwargs)
            except Exception as exc:
                logger.error(f"Failed to create OpenAI client ({self._client_type}): {exc}")
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

        # === 演示模式日志：请求接收 ===
        print(f"\n{'='*50}")
        print(f"[LLM] RECEIVED request")
        print(f"{'='*50}")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # 截断显示
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{role}]: {preview}")
        print(f"{'='*50}")

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

        print(f"[LLM] PROCESSING (model={self._settings.OPENAI_MODEL}, temp={temperature}, max_tokens={max_tokens})")

        # 重试循环：最多 3 次，快速退避（1s, 2s, 3s）
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                # 追踪 token 使用量
                usage = response.usage
                if usage:
                    self._total_input_tokens += usage.prompt_tokens
                    self._total_output_tokens += usage.completion_tokens
                    print(f"[LLM] DONE (elapsed: {elapsed:.1f}s, input: {usage.prompt_tokens} tokens, output: {usage.completion_tokens} tokens)")
                else:
                    print(f"[LLM] DONE (elapsed: {elapsed:.1f}s)")

                # 返回助手的回复内容
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                # 快速退避：等待 attempt 秒（1, 2, 3）
                wait = attempt
                print(f"[LLM] WARNING: attempt {attempt} failed ({exc}). retry in {wait}s...")
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
            json.JSONDecodeError: 如果响应在修复与严格重试后仍不是有效 JSON
        """
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        parsed, last_error = self._parse_json_with_repair(raw)
        if parsed is not None:
            return parsed

        strict_instruction = (
            "Return a strictly valid JSON object only. "
            "Do not include markdown fences, commentary, or trailing commas."
        )
        strict_messages = list(messages) + [{"role": "user", "content": strict_instruction}]
        retry_tokens = max(max_tokens or 0, self._settings.OPENAI_MAX_TOKENS)
        strict_raw = self.chat(
            strict_messages,
            temperature=0.0,
            max_tokens=retry_tokens,
            json_mode=True,
        )
        strict_parsed, strict_error = self._parse_json_with_repair(strict_raw)
        if strict_parsed is not None:
            return strict_parsed

        logger.error("JSON parse failed after strict retry. Initial raw:\n%s\nStrict raw:\n%s", raw, strict_raw)
        if strict_error is not None:
            raise strict_error
        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("Expecting value", strict_raw, 0)

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if not lines:
            return stripped

        # Remove opening fence (``` or ```json)
        lines = lines[1:]
        # Remove closing fence when present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_balanced_json_objects(text: str) -> List[str]:
        objects: List[str] = []
        depth = 0
        start_idx: Optional[int] = None
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                if depth == 0:
                    start_idx = idx
                depth += 1
                continue

            if ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    objects.append(text[start_idx : idx + 1])
                    start_idx = None

        return objects

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        result: List[str] = []
        in_string = False
        escape = False
        idx = 0
        length = len(text)

        while idx < length:
            ch = text[idx]

            if in_string:
                result.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                idx += 1
                continue

            if ch == '"':
                in_string = True
                result.append(ch)
                idx += 1
                continue

            if ch == ",":
                lookahead = idx + 1
                while lookahead < length and text[lookahead].isspace():
                    lookahead += 1
                if lookahead < length and text[lookahead] in "}]":
                    idx += 1
                    continue

            result.append(ch)
            idx += 1

        return "".join(result)

    def _parse_json_with_repair(self, raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[json.JSONDecodeError]]:
        candidates: List[str] = []
        seen: set[str] = set()

        def _push(candidate: str) -> None:
            value = candidate.strip()
            if value and value not in seen:
                seen.add(value)
                candidates.append(value)

        _push(raw)
        _push(self._strip_markdown_fence(raw))

        for base in list(candidates):
            for obj in self._extract_balanced_json_objects(base):
                _push(obj)

        expanded: List[str] = []
        expanded_seen: set[str] = set()
        for candidate in candidates:
            if candidate not in expanded_seen:
                expanded_seen.add(candidate)
                expanded.append(candidate)

            repaired = self._remove_trailing_commas(candidate)
            if repaired and repaired not in expanded_seen:
                expanded_seen.add(repaired)
                expanded.append(repaired)

        last_error: Optional[json.JSONDecodeError] = None
        for candidate in expanded:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, last_error
                last_error = json.JSONDecodeError("Top-level JSON is not an object", candidate, 0)
            except json.JSONDecodeError as exc:
                last_error = exc

        return None, last_error

    # ── usage tracking ────────────────────────────────────
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

    def usage_snapshot(self) -> Dict[str, float | int]:
        """Capture current aggregate token usage counters.

        Returns:
            Dict[str, float | int]: snapshot containing input/output tokens.
        """
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }

    def usage_delta(
        self,
        before: Dict[str, float | int],
        after: Dict[str, float | int],
    ) -> Dict[str, float]:
        """Compute usage delta between two snapshots."""
        return {
            "input_tokens": float(after.get("input_tokens", 0)) - float(before.get("input_tokens", 0)),
            "output_tokens": float(after.get("output_tokens", 0)) - float(before.get("output_tokens", 0)),
        }


# Convenience module-level singleton and hybrid routing
llm_client = LLMClient()


def get_client_for_task(task_type: str = "default") -> LLMClient:
    """Get the appropriate LLM client for a given task type.
    
    Routes requests based on NLG_MODE configuration:
    - hybrid mode: creative tasks (story) → local, structured (option/relation) → API
    - local mode: always local
    - api mode: always API
    
    参数：
        task_type: Task type ("story", "option", "relation", "json", "default")
    
    返回：
        LLMClient: The appropriate client for the task
    """
    return HybridClientManager.get_client_for_task(task_type)
