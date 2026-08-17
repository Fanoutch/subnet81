"""Task 2 du port v4 : encode_prompt raw-completion.

Sous v4 le prompt est encodé BRUT (add_special_tokens=False), jamais via le
chat template — même si le tokenizer en déclare un (piège Qwen3-4B-Base : le
repo livre le template famille sans l'avoir appris). Parité upstream 8c38992.
"""
import pytest

from tests.v4helpers import reload_constants


@pytest.fixture(autouse=True)
def _restore_constants(monkeypatch):
    yield
    reload_constants(monkeypatch)


class _ChatTokenizer:
    """Tokenizer factice qui DÉCLARE un chat template (piège Qwen3-4B-Base)."""

    chat_template = "{{ messages }} enable_thinking"

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "v4 doit passer add_special_tokens=False"
        return [len(text), 42]

    def apply_chat_template(self, messages, **kwargs):
        raise AssertionError("v4 ne doit JAMAIS appeler apply_chat_template")


def test_v4_encode_prompt_is_raw(monkeypatch):
    monkeypatch.setattr("reliquary.constants.RAW_COMPLETION_PROMPTS", True)
    from reliquary.protocol.tokens import encode_prompt

    assert encode_prompt(_ChatTokenizer(), "abc") == [3, 42]


def test_v3_encode_prompt_still_uses_template(monkeypatch):
    monkeypatch.setattr("reliquary.constants.RAW_COMPLETION_PROMPTS", False)
    from reliquary.protocol.tokens import encode_prompt

    class _Tok(_ChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]

    assert encode_prompt(_Tok(), "abc") == [1, 2, 3]
