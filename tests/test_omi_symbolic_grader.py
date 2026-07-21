"""OMI symbolic-grader parity with validator e7155f2 (allow_symbols +
algebraic equivalence + expand DoS bound). Math is NOT validator-authoritative
→ our reward must match the validator's grade or the submission is
REWARD_MISMATCH-rejected."""


def test_expr_str_guard_symbols_flag():
    from reliquary.environment.openmathinstruct import _expr_str_is_safe
    assert _expr_str_is_safe("2 - b") is False
    assert _expr_str_is_safe("2 + 3*(4)") is True
    assert _expr_str_is_safe("2 - b", allow_symbols=True) is True
    assert _expr_str_is_safe("x**2 + 2*x + 1", allow_symbols=True) is True
    assert _expr_str_is_safe("5!", allow_symbols=True) is False
    assert _expr_str_is_safe("exp(x)", allow_symbols=True) is False
    assert _expr_str_is_safe("gamma(3)", allow_symbols=True) is False


def test_expr_struct_guard_symbols_flag():
    import sympy
    from reliquary.environment.openmathinstruct import _expr_is_safe
    x, b = sympy.symbols("x b")
    assert _expr_is_safe(x) is False
    assert _expr_is_safe(x, allow_symbols=True) is True
    assert _expr_is_safe((x + 1) ** 2, allow_symbols=True) is True
    assert _expr_is_safe(2 - b, allow_symbols=True) is True
    assert _expr_is_safe(x ** 50, allow_symbols=True) is False
    assert _expr_is_safe(sympy.Pow(2, x, evaluate=False), allow_symbols=True) is False


def test_answers_equal_algebraic_equivalence():
    from reliquary.environment.openmathinstruct import _answers_equal
    # algebraic reorder / expansion the numeric path misses
    assert _answers_equal("2-b", "-b+2") is True
    assert _answers_equal("(x+1)**2", "x**2+2*x+1") is True
    # genuinely different expressions still False
    assert _answers_equal("x+1", "x+2") is False
    # pure numbers untouched (numeric path)
    assert _answers_equal("1144.2", "5721/5") is True


def test_expand_term_bound_caps():
    import sympy
    from reliquary.environment.openmathinstruct import _expand_term_bound
    a, b, c, d, e, f = sympy.symbols("a b c d e f")
    assert _expand_term_bound(a + b + c, 2000) == 3
    # (a+b+c+d+e+f)**10 expands to comb(15,10)=3003 monomials > cap → short-circuit
    assert _expand_term_bound((a + b + c + d + e + f) ** 10, 2000) > 2000
