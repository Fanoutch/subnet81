"""Fenêtre glissante de prompts dans un bake (25/09, mineur math).

Mesuré sur la lane math (32 fenêtres, nuit 24→25/09) : la lane se ferme à
~45 s, or le bake suivant attend la traîne du bake courant et ne démarre qu'à
67 s — GPU quasi à l'arrêt pendant que 2-3 rollouts longs finissent.

``max_live_prompts=K`` : le bake reçoit une liste plus longue (ordre du
classement) mais n'a jamais plus de K prompts en vol ; chaque groupe livré
libère une place pour le prompt SUIVANT de la liste. Les groupes lents ne sont
pas perdus, le GPU ne se vide plus.

Verrouillé ici :
- au départ, K prompts seulement en vol ;
- une livraison enfile le prompt suivant (ordre du classement) ;
- tous les groupes finissent par être livrés, contrat de retour inchangé ;
- K=0 (défaut) ⇒ comportement strictement identique à avant ;
- au flip, seuls les prompts ENFILÉS sont avortés ; les autres sont absents
  du retour ;
- avec sprint, le balayage respecte aussi la limite K.
"""
from __future__ import annotations

from tests.test_sprint_then_scan import _FakeEngine, _backend, _install_fake_vllm


def _run(b, n_prompts, m=2, live=0, sprint=0, on_group=None, should_abort=None):
    return b.generate_forced_phase1_multi_stream(
        [[1, 2, 3]] * n_prompts,
        prompt_indices=list(range(100, 100 + n_prompts)),
        randomness="ab" * 32, checkpoint_hash="ck",
        m_rollouts=m, max_tokens=64,
        stop_token_ids=[9], primary_eos_id=9,
        on_group=on_group, should_abort=should_abort,
        sprint_size=sprint, sprint_max_wait_s=999.0,
        max_live_prompts=live,
    )


def test_only_k_prompts_in_flight_then_refilled_in_rank_order(monkeypatch):
    _install_fake_vllm(monkeypatch)
    # 4 prompts x 2 rollouts, K=2 : p0,p1 d'abord ; p0 livré -> p2 ; p1 -> p3
    plan = [["p0-r0", "p0-r1"], ["p1-r0"], ["p1-r1", "p2-r0"],
            ["p2-r1", "p3-r0"], ["p3-r1"]]
    e = _FakeEngine(plan, {})
    got = _run(_backend(e), 4, live=2)
    assert e.added_at_step[0] == 4, "2 prompts x 2 rollouts au départ"
    assert e.added_at_step[1] == 6, "p0 livré -> p2 enfilé"
    assert e.added_at_step[3] == 8, "p1 livré -> p3 enfilé"
    rids = [rid for rid, _, _ in e.added]
    assert rids == ["p0-r0", "p0-r1", "p1-r0", "p1-r1",
                    "p2-r0", "p2-r1", "p3-r0", "p3-r1"]
    assert len(got) == 4 and all(len(g) == 2 for g in got)


def test_zero_keeps_legacy_all_at_once(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1", "p1-r0", "p1-r1", "p2-r0", "p2-r1"]]
    e = _FakeEngine(plan, {})
    got = _run(_backend(e), 3, live=0)
    assert e.added_at_step[0] == 6
    assert len(got) == 3 and all(len(g) == 2 for g in got)


def test_k_larger_than_list_is_all_at_once(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1", "p1-r0", "p1-r1"]]
    e = _FakeEngine(plan, {})
    _run(_backend(e), 2, live=8)
    assert e.added_at_step[0] == 4


def test_on_group_called_for_refilled_prompts(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1"], ["p1-r0", "p1-r1"], ["p2-r0", "p2-r1"]]
    e = _FakeEngine(plan, {})
    seen = []
    _run(_backend(e), 3, live=1,
         on_group=lambda pos, idx, g: seen.append((pos, idx)))
    assert seen == [(0, 100), (1, 101), (2, 102)]


def test_abort_only_touches_enqueued_prompts(monkeypatch):
    _install_fake_vllm(monkeypatch)
    # K=2 sur 4 prompts ; flip au 2e step : p0 livré (p2 enfilé), p1/p2 en vol
    plan = [["p0-r0", "p0-r1"], ["p1-r0"], ["p1-r1"], ["p2-r0", "p2-r1"]]
    e = _FakeEngine(plan, {})
    calls = {"n": 0}

    def _abort():
        calls["n"] += 1
        return calls["n"] >= 3   # après 2 steps

    got = _run(_backend(e), 4, live=2, should_abort=_abort)
    assert not any(r.startswith("p3") for r in e.aborted), \
        "un prompt jamais enfilé ne s'avorte pas"
    assert not any(r.startswith("p3") for r, _, _ in e.added)
    assert got[0] and len(got[0]) == 2
    assert got[3] == []


def test_sprint_scan_respects_the_live_cap(monkeypatch):
    _install_fake_vllm(monkeypatch)
    # sprint=1, K=2, 4 prompts : p0 seul ; livré -> balayage limité à 2 en vol
    plan = [["p0-r0", "p0-r1"], ["p1-r0", "p1-r1"], ["p2-r0", "p2-r1"],
            ["p3-r0", "p3-r1"]]
    e = _FakeEngine(plan, {})
    got = _run(_backend(e), 4, live=2, sprint=1)
    assert e.added_at_step[0] == 2, "sprint : p0 seul"
    assert e.added_at_step[1] == 6, "p0 livré : balayage p1,p2 (2 en vol)"
    assert e.added_at_step[2] == 8, "p1 livré : p3 enfilé"
    assert len(got) == 4 and all(len(g) == 2 for g in got)
