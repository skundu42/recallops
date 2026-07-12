from __future__ import annotations

import pytest

from recallops import llm as llm_module
from recallops.llm import (
    DEFAULT_CHAT_MODEL,
    OpenAIChatLLM,
    estimate_generation_cost,
    get_llm,
    openai_chat_request_body,
    parse_openai_chat,
)


def test_request_body_and_parse():
    body = openai_chat_request_body("gpt-4o-mini", "Write one question.")
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "Write one question."}]
    assert body["temperature"] == 0
    payload = {"choices": [{"message": {"content": "What is a widget?"}}]}
    assert parse_openai_chat(payload) == "What is a widget?"


def test_get_llm_parses_spec():
    assert get_llm("openai").model == DEFAULT_CHAT_MODEL
    assert get_llm("openai:gpt-4o").model == "gpt-4o"
    with pytest.raises(ValueError):
        get_llm("anthropic")
    with pytest.raises(ValueError):
        get_llm("openai:gpt-9-nonexistent")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIChatLLM()("prompt")


def test_call_goes_through_post_json(monkeypatch):
    calls = []

    def fake_post(url, body, headers, timeout=None):
        calls.append((url, body, headers))
        return {"choices": [{"message": {"content": " A question? "}}]}

    monkeypatch.setattr(llm_module, "_post_json", fake_post)
    out = OpenAIChatLLM(api_key="sk-test")("the prompt")
    assert out == " A question? "
    (url, body, headers), = calls
    assert "chat/completions" in url
    assert headers["Authorization"] == "Bearer sk-test"
    assert body["messages"][0]["content"] == "the prompt"


def test_estimate_scales_with_n_and_is_positive():
    small = estimate_generation_cost("gpt-4o-mini", 10, avg_chunk_tokens=200.0)
    large = estimate_generation_cost("gpt-4o-mini", 100, avg_chunk_tokens=200.0)
    assert 0 < small < large
    assert large == pytest.approx(small * 10)
    with pytest.raises(KeyError):
        estimate_generation_cost("gpt-9-nonexistent", 10, avg_chunk_tokens=200.0)
