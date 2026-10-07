import pytest
from app.cache_key import CacheRequest


def req(user="What is Python?", system=None, model="gpt-4o-mini", temperature=None, max_tokens=None, history=()):
    messages = [{"role": "system", "content": system}] if system else []
    messages += list(history) + [{"role": "user", "content": user}]
    return CacheRequest(model, messages, temperature, max_tokens)


def test_same_request_same_scope():
    assert req().scope_hash == req().scope_hash


def test_different_user_text_same_scope():
    # The user text is compared by MEANING, so it must NOT be part of the scope.
    assert req("What is Python?").scope_hash == req("Explain Python to me").scope_hash


def test_system_prompt_changes_scope():
    assert req(system="Be brief").scope_hash != req(system="Be a pirate").scope_hash
    assert req(system="Be brief").system_prompt_hash != req(system="Be a pirate").system_prompt_hash


@pytest.mark.parametrize("change", [
    {"temperature": 1.0}, {"max_tokens": 50}, {"model": "gpt-4o"},
])
def test_params_change_scope(change):
    assert req().scope_hash != req(**change).scope_hash


def test_chat_history_changes_scope():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    assert req().scope_hash != req(history=history).scope_hash


def test_query_text_is_last_user_message():
    assert req("  Hello there ").query_text == "Hello there"


def test_no_user_message_raises():
    with pytest.raises(ValueError):
        CacheRequest("m", [{"role": "system", "content": "x"}]).query_text