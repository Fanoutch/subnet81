from reliquary.environment.openmathinstruct import _answers_equal


def test_latex_frac_equals_decimal():
    assert _answers_equal("\\frac{5721}{5}", "1144.2") is True
    assert _answers_equal("\\frac{1}{2}", "0.5") is True


def test_latex_sqrt_and_cdot():
    assert _answers_equal("\\sqrt{4}", "2") is True
    assert _answers_equal("2\\cdot3", "6") is True


def test_numeric_still_works():  # #85 regression
    assert _answers_equal("82.50", "82.5") is True
    assert _answers_equal("1/2", "0.5") is True


def test_non_equal_values():
    assert _answers_equal("\\frac{1}{3}", "0.5") is False
    assert _answers_equal("7", "8") is False


def test_free_symbols_algebraic_equivalence():
    # validator e7155f2: algebraic reorderings now match via the symbolic-equality
    # path (was a string fall-back before; our grader is ported byte-exact).
    assert _answers_equal("x+1", "1+x") is True
    assert _answers_equal("x+1", "x+1") is True
    assert _answers_equal("x+1", "x+2") is False  # genuinely different → still False


def test_adversarial_payloads_rejected_fast():
    # Must return False WITHOUT hanging (power tower / factorial guards).
    assert _answers_equal("9^9^9^9", "1") is False
    assert _answers_equal("100!", "1") is False
    assert _answers_equal("2^99999", "1") is False
