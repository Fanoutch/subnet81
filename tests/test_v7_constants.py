import reliquary.constants as c


def test_v7_values():
    assert c.GRAIL_PROOF_VERSION == "v7"
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 32768
    assert c.BFT_ENABLED is True
    assert c.BFT_THINKING_BUDGET == 2048 and c.BFT_ANSWER_BUDGET == 512
    assert c.BFT_FORCE_TEMPLATE == "</think>\n\nFinal Answer: \\boxed{"
    assert c.T_PROTO == 0.6 and c.TOP_P_PROTO == 0.95 and c.TOP_K_PROTO == 20
    assert c.DEFAULT_BASE_MODEL == "Qwen/Qwen3.5-2B"
    assert c.TOKEN_AUTH_THRESHOLD == 1e-8
