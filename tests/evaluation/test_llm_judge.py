from __future__ import annotations

from types import SimpleNamespace

from src.evaluation import llm_judge


def _mock_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_judge_success_with_glm5_json(monkeypatch) -> None:
    captured: dict = {}

    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return _mock_response(
                '{"narrative_quality":8,"consistency":9,"player_agency":7,'
                '"creativity":8,"pacing":7,"option_relevance":9,'
                '"causal_link":8,"local_coherence":9}'
            )

    class _FakeOpenAI:
        def __init__(self, api_key: str, base_url: str):
            assert api_key == "test-key"
            assert base_url == "https://open.bigmodel.cn/api/paas/v4"
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(llm_judge, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_MODEL", "glm-5")
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_TEMPERATURE", 0.3)
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_MAX_TOKENS", 256)

    scores = llm_judge.judge("A coherent multi-turn transcript")

    assert scores["narrative_quality"] == 8
    assert scores["consistency"] == 9
    assert scores["player_agency"] == 7
    assert scores["average"] == 8.12

    assert captured["model"] == "glm-5"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_judge_missing_or_invalid_dimensions_become_zero(monkeypatch) -> None:
    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            return _mock_response(
                '{"narrative_quality":"10","consistency":12,"player_agency":-1,'
                '"creativity":"abc","pacing":4}'
            )

    class _FakeOpenAI:
        def __init__(self, api_key: str, base_url: str):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(llm_judge, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_API_KEY", "test-key")

    scores = llm_judge.judge("A transcript")

    assert scores["narrative_quality"] == 10
    assert scores["consistency"] == 10
    assert scores["player_agency"] == 0
    assert scores["creativity"] == 0
    assert scores["option_relevance"] == 0


def test_judge_returns_zeros_when_response_has_no_dimensions(monkeypatch) -> None:
    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            return _mock_response('{"foo": 1, "bar": 2}')

    class _FakeOpenAI:
        def __init__(self, api_key: str, base_url: str):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(llm_judge, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(llm_judge.settings, "EVAL_LLM_API_KEY", "test-key")

    scores = llm_judge.judge("A transcript")
    assert all(scores[d] == 0 for d in llm_judge.DIMENSIONS)
    assert scores["average"] == 0.0
