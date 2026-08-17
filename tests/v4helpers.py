"""Helper partagé des tests v4 : recharge reliquary.constants sous un
RELIQUARY_PROTOCOL_VERSION donné.

Les constantes sont évaluées à l'import ; les tests de dérivation doivent
recharger le module avec l'env posé. Les modules CONSOMMATEURS n'ont pas
besoin de reload : ils lisent les constantes par import paresseux (pattern
upstream) et se monkeypatchent directement
(``monkeypatch.setattr("reliquary.constants.X", ...)``).
"""
import importlib


def reload_constants(monkeypatch, version=None):
    """Recharge ``reliquary.constants`` avec RELIQUARY_PROTOCOL_VERSION=version.

    ``version=None`` → env nettoyé (défaut live v3). Renvoie le module
    rechargé. Le caller DOIT re-appeler ``reload_constants(monkeypatch)`` en
    fin de test (fixture autouse) pour restaurer l'état module global.
    """
    if version is None:
        monkeypatch.delenv("RELIQUARY_PROTOCOL_VERSION", raising=False)
    else:
        monkeypatch.setenv("RELIQUARY_PROTOCOL_VERSION", str(version))
    import reliquary.constants as c
    importlib.reload(c)
    return c
