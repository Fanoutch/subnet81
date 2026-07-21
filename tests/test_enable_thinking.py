from reliquary.protocol.tokens import encode_prompt


class _Tok:
    chat_template = "…{% if enable_thinking %}…"

    def apply_chat_template(self, messages, **kw):
        _Tok.seen = kw
        return [1, 2, 3]

    def encode(self, *a, **k):
        return [9]


def test_thinking_enabled():
    encode_prompt(_Tok(), "hi")
    assert _Tok.seen.get("enable_thinking") is True
