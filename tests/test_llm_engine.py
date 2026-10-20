import pytest

from src import llm_engine
from src.llm_engine import LLMError, build_user_prompt, get_llm_response, rewrite_query


def test_build_user_prompt_numbers_the_sources():
    prompt = build_user_prompt("What is X?", ["first chunk", "second chunk"])
    assert "Source 1: first chunk" in prompt
    assert "Source 2: second chunk" in prompt
    assert prompt.endswith("What is X?")


def test_unknown_provider_raises():
    with pytest.raises(LLMError):
        get_llm_response("q", ["c"], provider="nope")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_llm_response("q", ["c"], provider="claude")


def test_rewrite_query_without_history_makes_no_call(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should not call the LLM")
    monkeypatch.setattr(llm_engine, "_generate", fail)
    assert rewrite_query("What is X?", []) == "What is X?"


def test_rewrite_query_keeps_first_line_and_falls_back_on_error(monkeypatch):
    history = [("What accuracy on MNIST?", "98.65%")]

    monkeypatch.setattr(llm_engine, "_generate", lambda *a, **k: "What accuracy on CIFAR10?\nextra chatter")
    assert rewrite_query("and CIFAR10?", history) == "What accuracy on CIFAR10?"

    def broken(*args, **kwargs):
        raise LLMError("network down")
    monkeypatch.setattr(llm_engine, "_generate", broken)
    assert rewrite_query("and CIFAR10?", history) == "and CIFAR10?"
