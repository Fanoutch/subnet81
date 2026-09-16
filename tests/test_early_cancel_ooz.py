"""Annuler la vérification précoce EOS des groupes jetés hors zone (16/09).

``_on_rollout`` (engine.py) envoie CHAQUE rollout à la réplique dès sa sortie du
décodeur, donc avant le grading du groupe. Or 46-54 % des groupes sont ensuite
jetés ``out_of_zone``, et la réplique est une file FIFO à 2 workers : les
vérifications de groupes condamnés passent devant celles des groupes utiles.
La réparation, elle, est déjà évitée pour ces groupes (pregrade avant
réparation, fen 45896) — mais pas la vérification précoce.

Correctif : au moment du rejet hors zone, annuler les vérifications de ce
groupe qui sont ENCORE EN FILE (un ``Future`` non démarré s'annule ; un appel
déjà parti à la réplique n'est pas touché). Dormant par défaut :
``RELIQUARY_EARLY_CANCEL_OOZ=1`` pour l'activer.
"""
from __future__ import annotations

import threading
import types

from reliquary.miner.early_terminal import EarlyTerminalVerifier

EOS = 99
CTX = ("rand", "ckpt", "/snap")


def _blocked_verifier():
    """1 worker bloqué sur un verrou : tout ce qui suit reste EN FILE."""
    gate = threading.Event()
    started = threading.Event()
    calls = []

    def verify(ctx, prompt_idx, items):
        calls.append((prompt_idx, items[0]["rollout"]))
        started.set()
        gate.wait(5.0)
        return [{"ok": True, "pick": EOS, "cdf_miss": 0.0}]

    v = EarlyTerminalVerifier(verify, eos_ids=[EOS], max_workers=1)
    return v, gate, started, calls


def _toks(r):
    return [1, 2, 10 + r, EOS]


def test_annule_les_verifications_en_file_du_groupe():
    v, gate, started, calls = _blocked_verifier()
    v.submit(CTX, 7, 0, _toks(0), prompt_len=2)          # démarre et bloque
    assert started.wait(5.0)
    for r in (1, 2, 3):
        v.submit(CTX, 7, r, _toks(r), prompt_len=2)      # en file
    assert v.cancel_prompt(7) == 3
    gate.set(); v.close()
    assert calls == [(7, 0)], "les rollouts annulés ne doivent jamais partir"


def test_ne_touche_pas_les_autres_groupes():
    v, gate, started, calls = _blocked_verifier()
    v.submit(CTX, 7, 0, _toks(0), prompt_len=2)
    assert started.wait(5.0)
    v.submit(CTX, 7, 1, _toks(1), prompt_len=2)
    v.submit(CTX, 8, 0, _toks(0), prompt_len=2)
    assert v.cancel_prompt(7) == 1
    gate.set()
    assert v.lookup(CTX, 8, 0, _toks(0), timeout=5.0) == {"ok": True, "pick": EOS, "cdf_miss": 0.0}
    v.close()


def test_verdict_d_un_rollout_annule_vaut_none_sans_exception():
    v, gate, started, calls = _blocked_verifier()
    v.submit(CTX, 7, 0, _toks(0), prompt_len=2)
    assert started.wait(5.0)
    v.submit(CTX, 7, 1, _toks(1), prompt_len=2)
    v.cancel_prompt(7)
    assert v.lookup(CTX, 7, 1, _toks(1), timeout=1.0) is None
    gate.set(); v.close()


def test_rien_a_annuler_quand_tout_est_deja_parti():
    v, gate, started, calls = _blocked_verifier()
    v.submit(CTX, 7, 0, _toks(0), prompt_len=2)
    assert started.wait(5.0)
    assert v.cancel_prompt(7) == 0                        # en cours : intouchable
    assert v.cancel_prompt(12345) == 0                    # inconnu
    gate.set(); v.close()


# ------------------------------------------------------------ câblage moteur
def _prebake(monkeypatch, flag):
    from reliquary.miner import engine

    monkeypatch.setenv("RELIQUARY_REPLICA_SOCKET", "/tmp/replica.sock")
    monkeypatch.setenv("RELIQUARY_MIN_ROLLOUT_LEN", "0")
    if flag is None:
        monkeypatch.delenv("RELIQUARY_EARLY_CANCEL_OOZ", raising=False)
    else:
        monkeypatch.setenv("RELIQUARY_EARLY_CANCEL_OOZ", flag)
    eng = engine.MiningEngine.__new__(engine.MiningEngine)
    eng._eos_ids = [EOS]
    eng._cached_randomness = "ab" * 32
    eng._local_hash = "c" * 40
    eng.tokenizer = types.SimpleNamespace(decode=lambda toks: "t")
    gens = [{"tokens": [1, 2, 10 + r, EOS], "prompt_length": 2}
            for r in range(engine.M_ROLLOUTS)]
    eng._generate_m_rollouts = lambda problem, rand, env, prompt_idx: gens
    cancelled = []
    eng.__dict__["_early_terminal"] = types.SimpleNamespace(
        cancel_prompt=lambda pi: cancelled.append(pi) or 5)
    # groupe unanime -> hors zone dès le pregrade
    monkeypatch.setattr(engine, "grade_group_parallel_ex",
                        lambda env, pairs, max_workers=8: ([1.0] * len(pairs),
                                                           [False] * len(pairs)))
    monkeypatch.setattr(engine, "dump_group_sample", lambda **k: None)
    drops = []
    eng._record_drop = lambda dropped, reason=None: drops.append(reason)
    eng._sz_note = lambda *a, **k: None
    return eng, cancelled, drops


def test_rejet_hors_zone_annule_si_active(monkeypatch):
    eng, cancelled, drops = _prebake(monkeypatch, "1")
    out = eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                              env=types.SimpleNamespace(name="opencodeinstruct"))
    assert out is None and drops == ["out_of_zone"]
    assert cancelled == [7]


def test_dormant_par_defaut(monkeypatch):
    eng, cancelled, drops = _prebake(monkeypatch, None)
    out = eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                              env=types.SimpleNamespace(name="opencodeinstruct"))
    assert out is None and drops == ["out_of_zone"]
    assert cancelled == []


def test_sans_verificateur_ne_casse_rien(monkeypatch):
    eng, cancelled, drops = _prebake(monkeypatch, "1")
    eng.__dict__.pop("_early_terminal")
    assert eng._pre_bake_entry(7, {"prompt": "p"}, 1,
                               env=types.SimpleNamespace(name="opencodeinstruct")) is None
    assert drops == ["out_of_zone"]
