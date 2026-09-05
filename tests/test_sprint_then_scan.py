"""Sprint puis balayage : les têtes de classement décodent SEULES d'abord.

Le départage entre k=2 à égalité de score (0.325) est
``tokens / (round_arrivée - round_ouverture)`` — arriver au round 6 au lieu du
round 9 vaut +50%. Or les 128 séquences d'un lot se partagent le GPU : le n°1
du classement finit en ~22-25 s, EN MÊME TEMPS que le n°16.

Sprint : les ``sprint_size`` premiers prompts (= le haut du classement, l'ordre
d'entrée EST l'ordre du prédicteur) sont enfilés SEULS — ~2x moins de séquences
en vol, décodage ~2x plus rapide. Le balayage (le reste) démarre dès que le
sprint est livré, OU après ``sprint_max_wait_s`` si un prompt du sprint traîne
vers le plafond (un tronqueur ne doit pas retenir la couverture de la fenêtre).

Comportement verrouillé ici :
- pendant le sprint, SEULES les requêtes du sprint sont dans le moteur ;
- le balayage part à la livraison du dernier groupe du sprint ;
- ... ou au délai, même si le sprint n'a pas fini (ses séquences continuent) ;
- sprint_size=0 => comportement strictement identique à avant (tout d'un coup) ;
- le contrat de retour et le callback par groupe sont inchangés ;
- les extra_args forced-seed du balayage sont identiques au chemin non-sprinté.
"""
from __future__ import annotations

import types

from reliquary.miner.vllm_backend import VLLMBackend


class _Out:
    def __init__(self, request_id, token_ids, finished=True):
        self.request_id = request_id
        self.finished = finished
        self.outputs = [types.SimpleNamespace(
            token_ids=list(token_ids), stop_reason=None, finish_reason="stop")]


class _FakeEngine:
    """Scriptable : ``plan[i]`` = request_id qui FINISSENT au step i.

    Enregistre aussi, à chaque step, le nombre de requêtes ajoutées jusqu'ici
    (``added_at_step``) — c'est la preuve de qui était en vol quand.
    """

    def __init__(self, plan, tokens_by_rid):
        self._plan = list(plan)
        self._tokens = tokens_by_rid
        self.added = []
        self.added_at_step = []
        self.aborted = []

    def add_request(self, request_id, prompt, params):
        self.added.append((request_id, prompt, params))

    def has_unfinished_requests(self):
        return bool(self._plan)

    def step(self):
        self.added_at_step.append(len(self.added))
        if not self._plan:
            return []
        rids = self._plan.pop(0)
        return [_Out(rid, self._tokens.get(rid, [7, 9])) for rid in rids]

    def abort_request(self, request_ids):
        self.aborted.extend(request_ids if isinstance(request_ids, list)
                            else [request_ids])


def _install_fake_vllm(monkeypatch):
    class _SP:
        def __init__(self, **kw):
            self.kw = kw

    class _TP:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    monkeypatch.setitem(__import__("sys").modules, "vllm",
                        types.SimpleNamespace(SamplingParams=_SP))
    monkeypatch.setitem(__import__("sys").modules, "vllm.inputs",
                        types.SimpleNamespace(TokensPrompt=_TP))


def _backend(engine):
    b = VLLMBackend.__new__(VLLMBackend)
    b._llm = types.SimpleNamespace(llm_engine=engine)
    b._loaded = True
    b._ensure_loaded = lambda: None
    b._stream_request_id = lambda pos, r: f"p{pos}-r{r}"
    return b


def _run(b, n_prompts=4, m=2, sprint=2, wait=999.0, on_group=None):
    return b.generate_forced_phase1_multi_stream(
        [[1, 2, 3]] * n_prompts,
        prompt_indices=list(range(100, 100 + n_prompts)),
        randomness="ab" * 32, checkpoint_hash="ck",
        m_rollouts=m, max_tokens=64,
        stop_token_ids=[9], primary_eos_id=9,
        on_group=on_group,
        sprint_size=sprint, sprint_max_wait_s=wait,
    )


def test_during_sprint_only_sprint_requests_are_in_flight(monkeypatch):
    _install_fake_vllm(monkeypatch)
    # 4 prompts x 2 rollouts ; sprint = 2 prompts (p0, p1 = têtes de classement)
    plan = [["p0-r0"], ["p0-r1", "p1-r0"], ["p1-r1"],
            ["p2-r0", "p2-r1"], ["p3-r0", "p3-r1"]]
    e = _FakeEngine(plan, {})
    _run(_backend(e), sprint=2)
    # steps 1-3 : seules les 4 requêtes du sprint (2 prompts x 2 rollouts)
    assert e.added_at_step[0] == 4, "le balayage ne doit PAS être en vol au step 1"
    assert e.added_at_step[2] == 4, "toujours seul en vol tant que le sprint court"
    # dès le sprint livré (fin step 3), le balayage entre
    assert e.added_at_step[3] == 8, "balayage enfilé dès la livraison du sprint"


def test_scan_prompts_are_the_remaining_ranked_order(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1", "p1-r0", "p1-r1"],
            ["p2-r0", "p2-r1", "p3-r0", "p3-r1"]]
    e = _FakeEngine(plan, {})
    got = _run(_backend(e), sprint=2)
    # l'ordre des requêtes = p0,p1 (sprint) puis p2,p3 (balayage), prompt-major
    rids = [rid for rid, _, _ in e.added]
    assert rids == ["p0-r0", "p0-r1", "p1-r0", "p1-r1",
                    "p2-r0", "p2-r1", "p3-r0", "p3-r1"]
    assert len(got) == 4 and all(len(g) == 2 for g in got)


def test_timeout_releases_the_scan_even_if_sprint_drags(monkeypatch):
    """Un prompt de sprint qui file vers le plafond ne retient pas la fenêtre."""
    _install_fake_vllm(monkeypatch)
    # p0 ne finit JAMAIS pendant les premiers steps (tronqueur)
    plan = [[], [], ["p1-r0", "p1-r1"], ["p0-r0"], ["p0-r1"],
            ["p2-r0", "p2-r1", "p3-r0", "p3-r1"]]
    e = _FakeEngine(plan, {})
    _run(_backend(e), sprint=2, wait=0.0)      # délai immédiat
    # dès le premier tour de boucle, le délai est écoulé -> balayage enfilé
    assert e.added_at_step[1] == 8, (
        "après le délai, le balayage doit entrer même si le sprint court encore"
    )


def test_sprint_zero_is_byte_identical_to_the_old_behaviour(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1", "p1-r0", "p1-r1"]]
    e = _FakeEngine(plan, {})
    _run(_backend(e), n_prompts=2, sprint=0)
    assert e.added_at_step[0] == 4, "sprint=0 => tout enfilé d'un coup, comme avant"


def test_groups_still_stream_out_at_their_own_completion(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p1-r0", "p1-r1"], ["p0-r0", "p0-r1"],
            ["p3-r0", "p3-r1"], ["p2-r0", "p2-r1"]]
    e = _FakeEngine(plan, {})
    order = []
    _run(_backend(e), sprint=2,
         on_group=lambda pos, pidx, g: order.append(pos))
    assert order == [1, 0, 3, 2], "livraison à la complétion, sprint comme balayage"


def test_scan_extra_args_match_the_unsprinted_path(monkeypatch):
    """Les flux forcés du balayage doivent être IDENTIQUES au chemin sans
    sprint : un start_len/base_offset décalé = SEED_MISMATCH garanti."""
    _install_fake_vllm(monkeypatch)
    from reliquary.miner.vllm_forced_seed import FORCED_SEED_EXTRA_KEY

    plan = [["p0-r0", "p0-r1"], ["p1-r0", "p1-r1"]]
    e1 = _FakeEngine(list(plan), {})
    _run(_backend(e1), n_prompts=2, sprint=1)
    e2 = _FakeEngine(list(plan), {})
    _run(_backend(e2), n_prompts=2, sprint=0)
    ex1 = [p.kw["extra_args"][FORCED_SEED_EXTRA_KEY] for _, _, p in e1.added]
    ex2 = [p.kw["extra_args"][FORCED_SEED_EXTRA_KEY] for _, _, p in e2.added]
    assert ex1 == ex2
