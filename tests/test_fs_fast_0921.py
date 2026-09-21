"""21/09 : processeur forced-seed RAPIDE (``RELIQUARY_FS_FAST=1``).

Banc H100 du 21/09 : le forced-seed coûte +44 % par pas à 160 séquences
(12,06 contre 8,37 ms) — 3,7 ms dont ~0,64 de boucle CPU ``u_at`` et le reste
en passes pleine largeur sur des tenseurs [160, 151 936] (≈ 97 Mo chacun).

Le chemin rapide garde le MÊME calcul du token forcé et supprime le gaspillage :
1. ``u`` : préfixe SHA-256 haché une fois par séquence (``hashlib.copy``), seul
   ``t`` est ajouté à chaque pas ; valeurs écrites en une copie groupée.
2. pas de division par 1,0 (x / 1,0 == x en IEEE : identique au bit).
3. lignes forcées contiguës 0..n-1 -> vue, pas de copie ``index_select``.
4. masque CREUX : +1e30 sur le seul token forcé au lieu de -inf sur toute la
   ligne. vLLM échantillonne en glouton (``argmax``) sous temperature=0 :
   même token choisi, par construction.

Contrat verrouillé ici : le token que choisirait ``argmax`` est IDENTIQUE au
chemin actuel, ligne par ligne, et les lignes non forcées ne sont pas touchées.
"""
from __future__ import annotations

import random

import pytest
import torch

from reliquary.environment import forced_sampling as fsmp
from reliquary.environment.forced_sampling import force_rows_batched, u_at
from reliquary.miner import vllm_forced_seed as vfs
from reliquary.miner.vllm_forced_seed import ForcedRowsState, batched_force_mask

RND, CKPT = "0badf00d" * 8, "fast-gate"


@pytest.fixture(autouse=True)
def _protocole_plat(monkeypatch):
    """Protocole V1 (T=1, top_k=0, top_p=1) pour TOUS les tests : sans ça,
    l'environnement de test tourne en v3 (T=0,6, top_k=20) et le chemin
    rapide ne s'active pas — les tests passeraient sans l'exercer."""
    monkeypatch.setattr(vfs, "T_PROTO", 1.0)
    monkeypatch.setattr(vfs, "TOP_K_PROTO", 0)
    monkeypatch.setattr(vfs, "TOP_P_PROTO", 1.0)


def _fs(prompt_idx, rollout, base=0):
    return {"randomness": RND, "prompt_idx": prompt_idx,
            "checkpoint_hash": CKPT, "rollout_index": rollout,
            "base_offset": base, "start_len": 0}


# ───────────────────────────── u par préfixe ─────────────────────────────────
def test_u_par_prefixe_identique_au_bit():
    rng = random.Random(0)
    for _ in range(300):
        p, r, t = rng.randrange(10**7), rng.randrange(16), rng.randrange(9000)
        rnd = "%064x" % rng.getrandbits(256)
        pre = fsmp.u_at_prefix(rnd, p, CKPT, r)
        assert fsmp.u_from_prefix(pre, t) == u_at(rnd, p, CKPT, r, t)
        # le préfixe est réutilisable (copie interne)
        assert fsmp.u_from_prefix(pre, t + 1) == u_at(rnd, p, CKPT, r, t + 1)


# ───────────────────────── choix du token (sans masque) ───────────────────────
def test_pick_sans_division_identique_au_chemin_de_reference():
    g = torch.Generator().manual_seed(3)
    for n, vocab in ((1, 50), (7, 1000), (33, 32000)):
        lg = torch.randn(n, vocab, generator=g) * 4
        lg[0, :5] = 2.5                                   # égalités
        us = torch.rand(n, generator=g)
        ref = force_rows_batched(lg.clone(), us, t=1.0, top_k=0, top_p=1.0)
        got = vfs.fast_pick_t1(lg, us)
        assert torch.equal(got, ref)


def test_pick_ne_modifie_pas_les_logits():
    lg = torch.randn(4, 500)
    ref = lg.clone()
    vfs.fast_pick_t1(lg, torch.rand(4))
    assert torch.equal(lg, ref)


# ───────────────────────────── processeur complet ─────────────────────────────
def _state(monkeypatch, fast: bool, req: dict, device="cpu"):
    monkeypatch.setenv("RELIQUARY_FS_FAST", "1" if fast else "0")
    st = ForcedRowsState()
    st.rebuild(req, device=device)
    assert st._fast_now is fast          # le chemin visé est bien emprunté
    return st


def _req_contigu(n, outs):
    return {i: (_fs(1000 + i // 16, i % 16, base=i % 3), outs[i]) for i in range(n)}


def test_argmax_identique_au_chemin_actuel_lignes_contigues(monkeypatch):
    g = torch.Generator().manual_seed(7)
    n, vocab = 40, 4000
    outs = [[0] * (i * 3) for i in range(n)]
    req = _req_contigu(n, outs)
    logits = torch.randn(n, vocab, generator=g) * 3
    ref = _state(monkeypatch, False, req).apply(logits.clone())
    got = _state(monkeypatch, True, req).apply(logits.clone())
    assert torch.equal(ref.argmax(-1), got.argmax(-1))


def test_lignes_non_forcees_intactes_et_argmax_identique(monkeypatch):
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(6, 3000, generator=g)
    outs = {1: [1, 2], 3: [], 4: [7, 8, 9]}
    req = {1: (_fs(11, 0), outs[1]), 3: (_fs(11, 3), outs[3]),
           4: (_fs(42, 7, base=5), outs[4])}
    ref = _state(monkeypatch, False, req).apply(logits.clone())
    got = _state(monkeypatch, True, req).apply(logits.clone())
    forced = torch.tensor([1, 3, 4])
    assert torch.equal(ref[forced].argmax(-1), got[forced].argmax(-1))
    for row in (0, 2, 5):                                  # non forcées
        assert torch.equal(got[row], logits[row])


def test_suit_la_sortie_qui_grandit_sans_rebuild(monkeypatch):
    g = torch.Generator().manual_seed(9)
    out = [5]
    st = _state(monkeypatch, True, {0: (_fs(7, 2), out)})
    for step in range(5):
        l = torch.randn(1, 2000, generator=g)
        exp = batched_force_mask(l.clone(), [(0, u_at(RND, 7, CKPT, 2, len(out)))])
        assert int(st.apply(l.clone()).argmax(-1)) == int(exp.argmax(-1))
        out.append(step)


def test_rebuild_apres_changement_de_composition(monkeypatch):
    g = torch.Generator().manual_seed(10)
    a, b = [1, 2, 3], [4]
    st = _state(monkeypatch, True, {0: (_fs(1, 0), a), 1: (_fs(1, 1), b)})
    st.rebuild({0: (_fs(9, 5), b)}, device="cpu")
    l = torch.randn(2, 1500, generator=g)
    exp = batched_force_mask(l.clone(), [(0, u_at(RND, 9, CKPT, 5, len(b)))])
    got = st.apply(l.clone())
    assert int(got[0].argmax()) == int(exp[0].argmax())


def test_desactive_par_defaut(monkeypatch):
    monkeypatch.delenv("RELIQUARY_FS_FAST", raising=False)
    assert ForcedRowsState()._fast is False


def test_repli_si_protocole_non_plat(monkeypatch):
    """Le chemin rapide suppose T=1, top_k=0, top_p=1 (protocole V1). Hors de
    ce cas il ne s'active pas (calcul différent)."""
    monkeypatch.setenv("RELIQUARY_FS_FAST", "1")
    monkeypatch.setattr(vfs, "TOP_K_PROTO", 20)
    assert ForcedRowsState()._fast is False


# ───────────── sélection PAR REQUÊTE (banc : un seul moteur, 3 variantes) ─────
def test_chemin_rapide_par_requete_sans_variable(monkeypatch):
    monkeypatch.delenv("RELIQUARY_FS_FAST", raising=False)
    g = torch.Generator().manual_seed(11)
    outs = [[0] * i for i in range(8)]
    req_ref = {i: (_fs(5, i), outs[i]) for i in range(8)}
    req_fast = {i: ({**_fs(5, i), "fast": True}, outs[i]) for i in range(8)}
    logits = torch.randn(8, 2500, generator=g)
    st_ref = ForcedRowsState(); st_ref.rebuild(req_ref, device="cpu")
    st_fast = ForcedRowsState(); st_fast.rebuild(req_fast, device="cpu")
    assert st_ref._fast_now is False and st_fast._fast_now is True
    ref = st_ref.apply(logits.clone())
    got = st_fast.apply(logits.clone())
    assert torch.equal(ref.argmax(-1), got.argmax(-1))
    assert not torch.isinf(got).any()                # masque creux utilisé


def test_lot_mixte_reste_sur_le_chemin_actuel(monkeypatch):
    monkeypatch.delenv("RELIQUARY_FS_FAST", raising=False)
    req = {0: ({**_fs(5, 0), "fast": True}, []), 1: (_fs(5, 1), [])}
    st = ForcedRowsState(); st.rebuild(req, device="cpu")
    assert st._fast_now is False
