"""RELIQUARY_DRAND_URLS (2026-08-15) : un miroir drand mort gèle la boucle
~15-20 s par tirage malchanceux (flips ratés 28968-28970) — la liste doit
être surchageable sans toucher au code."""
import sys


def test_env_overrides_urls(monkeypatch):
    monkeypatch.setenv("RELIQUARY_DRAND_URLS",
                       "https://a.example, https://b.example")
    sys.modules.pop("reliquary.infrastructure.drand", None)
    import reliquary.infrastructure.drand as dr
    assert dr.DRAND_URLS == ["https://a.example", "https://b.example"]


def test_default_list_untouched(monkeypatch):
    monkeypatch.delenv("RELIQUARY_DRAND_URLS", raising=False)
    sys.modules.pop("reliquary.infrastructure.drand", None)
    import reliquary.infrastructure.drand as dr
    assert "https://api.drand.sh" in dr.DRAND_URLS
    assert len(dr.DRAND_URLS) == 5
