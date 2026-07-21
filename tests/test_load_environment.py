"""load_environment routing — incl. opencode registration (multi-env Phase 1).

Envs load their dataset eagerly in __init__, so we no-op __init__ to test the
registry routing offline (the envs have their own tests for internals).
"""
import pytest

from reliquary.environment import load_environment
from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment


def test_routing(monkeypatch):
    monkeypatch.setattr(OpenMathInstructEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)
    assert isinstance(load_environment("openmathinstruct"), OpenMathInstructEnvironment)
    assert isinstance(load_environment("opencodeinstruct"), OpenCodeInstructEnvironment)


def test_unknown_raises():
    with pytest.raises(ValueError):
        load_environment("nope")
