"""Tests for resilient JSON parsing in LLMClient.chat_json."""

import json

import pytest

from src.utils.api_client import LLMClient


class TestLLMClientChatJsonRepair:
    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient()

    def test_parse_json_with_markdown_fence(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = ['```json\n{"entities": [], "relations": []}\n```']

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        data = client.chat_json([{"role": "user", "content": "x"}])

        assert data == {"entities": [], "relations": []}

    def test_parse_json_with_trailing_commas(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = ['{"entities": [{"name": "Hero",}], "relations": [],}']

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        data = client.chat_json([{"role": "user", "content": "x"}])

        assert data["entities"][0]["name"] == "Hero"

    def test_retry_with_strict_json_instruction(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = ["{\"entities\": [", '{"entities": [], "relations": []}']
        calls = []

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            calls.append(
                {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "json_mode": json_mode,
                }
            )
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        data = client.chat_json([{"role": "user", "content": "extract"}], max_tokens=256)

        assert data == {"entities": [], "relations": []}
        assert len(calls) == 2
        assert calls[0]["json_mode"] is True
        assert calls[1]["json_mode"] is True
        assert calls[1]["temperature"] == 0.0
        strict_message = calls[1]["messages"][-1]["content"]
        assert "strictly valid JSON object" in strict_message

    def test_strict_retry_uses_non_decreasing_token_budget(
        self,
        client: LLMClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        responses = ["{\"entities\": [", '{"entities": [], "relations": []}']
        calls = []

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            calls.append(
                {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "json_mode": json_mode,
                }
            )
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        data = client.chat_json([{"role": "user", "content": "extract"}], max_tokens=256)

        assert data == {"entities": [], "relations": []}
        assert len(calls) == 2
        assert calls[0]["max_tokens"] == 256
        assert calls[1]["max_tokens"] == max(256, client._settings.OPENAI_MAX_TOKENS)

    def test_raise_when_repair_and_retry_both_fail(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = ["not-json", "still-not-json"]

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        with pytest.raises(json.JSONDecodeError):
            client.chat_json([{"role": "user", "content": "extract"}])

    def test_raise_propagates_real_strict_retry_error(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = ["{\"entities\": [", "still-not-json"]

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        with pytest.raises(json.JSONDecodeError) as exc_info:
            client.chat_json([{"role": "user", "content": "extract"}])

        assert exc_info.value.doc == "still-not-json"

    def test_trailing_comma_repair_keeps_string_content(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [
            '{"entities": [{"name": "Hero", "description": "Signal, } remains",}], "relations": [],}'
        ]

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        data = client.chat_json([{"role": "user", "content": "extract"}])

        assert data["entities"][0]["description"] == "Signal, } remains"

    def test_extracts_balanced_json_from_noise(self, client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [
            'analysis prefix {"bad": } then payload {"entities": [], "relations": []} trailing'
        ]

        def fake_chat(messages, *, temperature=None, max_tokens=None, json_mode=False):
            return responses.pop(0)

        monkeypatch.setattr(client, "chat", fake_chat)
        data = client.chat_json([{"role": "user", "content": "extract"}])

        assert data == {"entities": [], "relations": []}
