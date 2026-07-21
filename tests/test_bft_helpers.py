from reliquary.shared.modeling import (
    think_close_token_ids, force_close_token_ids, has_think_close)


class _Tok:
    def convert_tokens_to_ids(self, t):
        return 999 if t == "</think>" else -1

    def encode(self, s, add_special_tokens=False):
        return [40, 41]  # tail ids


def test_think_close_atomic():
    assert think_close_token_ids(_Tok()) == [999]


def test_force_close_is_close_plus_tail():
    assert force_close_token_ids(_Tok()) == [999, 40, 41]


def test_has_think_close():
    assert has_think_close([1, 999, 2], {999}) is True
    assert has_think_close([1, 2], {999}) is False
