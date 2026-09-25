"""C1 — BAKE GLISSANT : réinjecter un prompt dès qu'un groupe est livré.

Mesuré le 25/09 sur 19 fenêtres à lane pleine (R2 + journal) : la durée d'un
bake est fixée par sa séquence la PLUS LONGUE — corr(durée, plus long rollout)
= **+0,926** — et comme le rollout moyen d'un groupe ne fait que ~60 % du plus
long, la fin du bake décode de moins en moins de séquences. Conséquence
chiffrée : le débit tombe de 6 153 à 4 617 tok/s (−25 %) entre une fenêtre à
queue courte et une à queue longue, et corr(débit, payés) = +0,585. On tourne à
**44,8 séquences en vol en moyenne sur 128 nominales**.

Le coût d'un pas de décodage s'écrit `a + b·n_live` (ajusté sur 1 030 bakes
réels : a = 5,50 ms, b = 61,2 µs, R² 0,897) — donc un pas à 16 séquences coûte
6,48 ms contre 13,33 ms à 128 : **49 % du prix pour 12,5 % du travail.**

Contrefactuel (bras de référence validé contre le réel : 54,9 payés simulés
contre 54,2 réels, heures de livraison à 1,36 s près) : **+4,50 payés/fenêtre**.

⚠️ CE QUI DISTINGUE C1 DE BAKE 14, SPRINT 3 ET SCAN_HOLDOFF — tous rejetés pour
la même raison : ils faisaient MONTER le nombre de séquences en vol et volaient
du débit aux têtes qui paient. Ici la largeur ne dépasse JAMAIS `max_in_flight`,
la largeur déjà tenue en début de bake, et les `max_in_flight` premiers prompts
partent ensemble comme aujourd'hui — donc la tête est livrée à la milliseconde
près (vérifié en simulation : +0,000 s). Ces deux propriétés sont la raison
d'être du fix ; ce fichier les verrouille.

Repli : `RELIQUARY_BAKE_IN_FLIGHT=0` (défaut) = comportement strictement
inchangé, y compris l'ordre des requêtes.
"""
from __future__ import annotations

import types

import pytest

from reliquary.miner.vllm_backend import VLLMBackend


class _Out:
    def __init__(self, request_id, token_ids, finished=True):
        self.request_id = request_id
        self.finished = finished
        self.outputs = [types.SimpleNamespace(
            token_ids=list(token_ids), stop_reason=None, finish_reason="stop")]


class _FakeEngine:
    """``plan[i]`` = les request_id qui FINISSENT au step i.

    ``added_at_step[i]`` = nombre de requêtes ajoutées au moteur à l'entrée du
    step i : c'est la preuve de ce qui était en vol, et quand.
    """

    def __init__(self, plan, tokens_by_rid=None):
        self._plan = list(plan)
        self._tokens = tokens_by_rid or {}
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

    import sys
    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=_SP))
    monkeypatch.setitem(sys.modules, "vllm.inputs",
                        types.SimpleNamespace(TokensPrompt=_TP))
    monkeypatch.setitem(
        sys.modules, "vllm.sampling_params",
        types.SimpleNamespace(
            RequestOutputKind=types.SimpleNamespace(FINAL_ONLY="final_only")))


def _backend(engine):
    b = VLLMBackend.__new__(VLLMBackend)
    b._llm = types.SimpleNamespace(llm_engine=engine)
    b._loaded = True
    b._ensure_loaded = lambda: None
    b._stream_request_id = lambda pos, r: f"p{pos}-r{r}"
    return b


def _run(b, n_prompts, m=2, in_flight=0, on_group=None, should_admit=None,
         sprint=0):
    return b.generate_forced_phase1_multi_stream(
        [[1, 2, 3]] * n_prompts,
        prompt_indices=list(range(100, 100 + n_prompts)),
        randomness="ab" * 32, checkpoint_hash="ck",
        m_rollouts=m, max_tokens=64,
        stop_token_ids=[9], primary_eos_id=9,
        on_group=on_group,
        sprint_size=sprint, sprint_max_wait_s=999.0,
        max_in_flight=in_flight, should_admit=should_admit,
    )


# ───────────────── la largeur ne dépasse JAMAIS max_in_flight ────────────────
def test_seuls_les_premiers_prompts_partent_au_depart(monkeypatch):
    """Les `max_in_flight` premiers prompts — donc la TÊTE du classement —
    partent ensemble, exactement comme le premier bake d'aujourd'hui."""
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1"], ["p1-r0", "p1-r1"], ["p2-r0", "p2-r1"],
            ["p3-r0", "p3-r1"]]
    e = _FakeEngine(plan)
    _run(_backend(e), 4, m=2, in_flight=2)
    assert e.added_at_step[0] == 4, (
        "2 prompts x 2 rollouts au premier step, pas plus : la tête ne doit "
        "jamais partager le GPU avec plus de séquences qu'aujourd'hui"
    )


def test_un_prompt_entre_exactement_quand_un_groupe_sort(monkeypatch):
    _install_fake_vllm(monkeypatch)
    plan = [["p0-r0", "p0-r1"], ["p1-r0", "p1-r1"], ["p2-r0", "p2-r1"],
            ["p3-r0", "p3-r1"]]
    e = _FakeEngine(plan)
    _run(_backend(e), 4, m=2, in_flight=2)
    # step 0 : p0+p1 en vol (4). p0 livré à la fin du step 0 -> p2 entre.
    assert e.added_at_step[1] == 6, "p2 doit entrer dès la livraison de p0"
    assert e.added_at_step[2] == 8, "p3 doit entrer dès la livraison de p1"


def test_la_largeur_ne_depasse_jamais_le_plafond(monkeypatch):
    """LA propriété de sûreté. Bake 14 et sprint 3 ont échoué en élargissant la
    file ; C1 ne doit jamais tenir plus de `max_in_flight` prompts vivants."""
    _install_fake_vllm(monkeypatch)
    n, m, w = 9, 2, 3
    # un groupe livré par step, dans l'ordre
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(n)]
    e = _FakeEngine(plan)
    livres = []
    _run(_backend(e), n, m=m, in_flight=w,
         on_group=lambda pos, idx, g: livres.append(pos))
    # à l'entrée de chaque step : ajoutés - livrés <= plafond (en prompts)
    for i, added in enumerate(e.added_at_step):
        en_vol = added / m - len(livres[:i])
        assert en_vol <= w, (
            f"step {i} : {en_vol} prompts en vol pour un plafond de {w}"
        )


def test_les_prompts_entrent_dans_l_ordre_du_classement(monkeypatch):
    _install_fake_vllm(monkeypatch)
    n = 5
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(n)]
    e = _FakeEngine(plan)
    _run(_backend(e), n, m=2, in_flight=2)
    assert [rid for rid, _, _ in e.added] == [
        f"p{i}-r{r}" for i in range(n) for r in range(2)
    ], "l'ordre d'entrée EST l'ordre du prédicteur — ne jamais le réordonner"


# ───────────────── la garde de fenêtre peut fermer le robinet ────────────────
def test_should_admit_faux_arrete_la_reinjection(monkeypatch):
    """Près du flip, injecter un prompt qui ne finira pas est du GPU jeté : la
    garde doit pouvoir fermer le robinet sans avorter ce qui est en vol."""
    _install_fake_vllm(monkeypatch)
    n = 6
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(n)]
    e = _FakeEngine(plan)
    _run(_backend(e), n, m=2, in_flight=2, should_admit=lambda: False)
    assert len(e.added) == 4, (
        "aucune réinjection quand la garde refuse ; les 2 premiers prompts "
        "restent en vol et sont livrés normalement"
    )


def test_should_admit_qui_se_referme_en_cours(monkeypatch):
    """La garde est consultée à CHAQUE réinjection, pas une fois pour toutes :
    dès qu'elle se referme, plus rien n'entre, et ce qui est en vol est livré."""
    _install_fake_vllm(monkeypatch)
    n = 6
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(n)]
    e = _FakeEngine(plan)
    appels = {"n": 0}

    def admit():
        appels["n"] += 1
        return appels["n"] <= 1        # une seule réinjection autorisée

    livres = []
    _run(_backend(e), n, m=2, in_flight=2, should_admit=admit,
         on_group=lambda pos, idx, g: livres.append(pos))
    assert len(e.added) == 6, (
        f"2 prompts au départ + 1 réinjection = 3 prompts admis, "
        f"vu {len(e.added) / 2}"
    )
    assert appels["n"] >= 2, "la garde doit être re-consultée après son refus"
    assert livres[:3] == [0, 1, 2], (
        "les 3 prompts admis sont livrés normalement : refuser une réinjection "
        "n'avorte rien"
    )


def test_la_reinjection_precede_le_callback(monkeypatch):
    """ORDRE VOULU : le slot GPU se remplit AVANT `on_group`, qui déclenche
    grade + preuve GRAIL et peut durer plusieurs secondes. L'inverse rendrait
    la carte oisive pendant tout le callback — exactement ce que C1 corrige."""
    _install_fake_vllm(monkeypatch)
    n = 4
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(n)]
    e = _FakeEngine(plan)
    vu = []
    _run(_backend(e), n, m=2, in_flight=2,
         on_group=lambda pos, idx, g: vu.append(len(e.added)))
    assert vu[0] == 6, (
        "au moment du 1er callback, le prompt suivant doit DÉJÀ être dans le "
        f"moteur (6 requêtes), vu {vu[0]}"
    )


# ───────────────── repli : 0 = comportement strictement inchangé ─────────────
@pytest.mark.parametrize("in_flight", [0, 4, 9])
def test_plafond_nul_ou_plus_grand_que_le_lot_est_le_comportement_actuel(
        monkeypatch, in_flight):
    """0 = éteint ; >= n = tout d'un coup. Dans les deux cas l'ordre ET le
    nombre de requêtes doivent être ceux d'aujourd'hui, au bit près."""
    _install_fake_vllm(monkeypatch)
    n, m = 4, 2
    plan = [[f"p{i}-r{r}" for i in range(n) for r in range(m)]]
    e = _FakeEngine(plan)
    got = _run(_backend(e), n, m=m, in_flight=in_flight)
    assert e.added_at_step[0] == n * m, (
        "tout le lot doit être en vol au premier step"
    )
    assert len(got) == n and all(len(g) == m for g in got)


def test_le_contrat_de_retour_est_inchange(monkeypatch):
    """Retour parallèle aux prompts, [] pour un groupe non livré."""
    _install_fake_vllm(monkeypatch)
    n = 4
    # p3 ne finit jamais : le moteur n'a plus rien après 3 groupes
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(3)]
    e = _FakeEngine(plan)
    got = _run(_backend(e), n, m=2, in_flight=2)
    assert len(got) == n
    assert all(len(got[i]) == 2 for i in range(3))
    assert got[3] == [], "un groupe non livré rend une liste vide"


def test_le_plafond_desactive_le_sprint(monkeypatch):
    """Deux ordonnanceurs qui se disputent la même file, c'est la recette des
    échecs passés : quand C1 est armé, le sprint est ignoré."""
    _install_fake_vllm(monkeypatch)
    n = 6
    plan = [[f"p{i}-r0", f"p{i}-r1"] for i in range(n)]
    e = _FakeEngine(plan)
    _run(_backend(e), n, m=2, in_flight=3, sprint=2)
    assert e.added_at_step[0] == 6, (
        "3 prompts (le plafond C1) au premier step, pas 2 (le sprint)"
    )


# ═══════════════════ câblage moteur : l'env var arrive au backend ════════════
import asyncio  # noqa: E402

from reliquary.miner.engine import MiningEngine, bake_in_flight  # noqa: E402


def test_lecteur_env_defaut_eteint(monkeypatch):
    """Défaut 0 = éteint. Un déploiement qui oublie la variable ne change RIEN."""
    monkeypatch.delenv("RELIQUARY_BAKE_IN_FLIGHT", raising=False)
    assert bake_in_flight() == 0
    monkeypatch.setenv("RELIQUARY_BAKE_IN_FLIGHT", "8")
    assert bake_in_flight() == 8
    monkeypatch.setenv("RELIQUARY_BAKE_IN_FLIGHT", "-3")
    assert bake_in_flight() == 0, "jamais négatif"
    monkeypatch.setenv("RELIQUARY_BAKE_IN_FLIGHT", "bruit")
    assert bake_in_flight() == 0, "une valeur illisible retombe sur éteint"


class _BackendC1:
    """Backend factice qui DÉCLARE `max_in_flight` : enregistre ce qu'il reçoit."""

    def __init__(self):
        self.vu = {}

    def generate_forced_phase1_multi_stream(
        self, prompts_tokens, *, prompt_indices, randomness, checkpoint_hash,
        m_rollouts, max_tokens, stop_token_ids, primary_eos_id,
        on_group=None, should_abort=None, sprint_size=0,
        sprint_max_wait_s=20.0, scan_holdoff_s=0.0,
        max_in_flight=0, should_admit=None,
    ):
        self.vu = dict(max_in_flight=max_in_flight, should_admit=should_admit,
                       sprint_size=sprint_size)
        groups = [[[7, 9]] * m_rollouts for _ in prompt_indices]
        for pos in range(len(prompt_indices)):
            if on_group is not None:
                on_group(pos, prompt_indices[pos], groups[pos])
        return groups


class _BackendSansC1:
    """Backend d'avant C1 : ne déclare PAS `max_in_flight` (rollback partiel)."""

    def __init__(self):
        self.appele = False

    def generate_forced_phase1_multi_stream(
        self, prompts_tokens, *, prompt_indices, randomness, checkpoint_hash,
        m_rollouts, max_tokens, stop_token_ids, primary_eos_id,
        on_group=None, should_abort=None,
    ):
        self.appele = True
        groups = [[[7, 9]] * m_rollouts for _ in prompt_indices]
        for pos in range(len(prompt_indices)):
            if on_group is not None:
                on_group(pos, prompt_indices[pos], groups[pos])
        return groups


def _moteur(backend):
    e = MiningEngine.__new__(MiningEngine)
    e._vllm_backend = backend
    e._cached_randomness = "ab" * 32
    e._local_hash = "ck"
    e.max_new_tokens = 64
    e.tokenizer = types.SimpleNamespace()
    e._eos_ids = [9]
    e._primary_eos_id = lambda: 9

    async def _fake_grade(chunk_pairs, entries, *, expected_ckpt_n, env):
        for prompt_idx, _p in chunk_pairs:
            entries.append({"prompt_idx": prompt_idx})

    e._grade_chunk_streaming = _fake_grade
    return e


def _drive(monkeypatch, backend, n=3):
    monkeypatch.setattr(
        "reliquary.miner.engine.encode_prompt", lambda tok, p: [1, 2, 3])
    monkeypatch.setattr("reliquary.constants.FORCED_SEED_ENFORCE", True)
    monkeypatch.setenv("RELIQUARY_VLLM_FORCED_SEED", "1")
    asyncio.run(_moteur(backend)._bake_stream_fire(
        [{"prompt": f"p{i}"} for i in range(n)],
        list(range(100, 100 + n)), expected_ckpt_n=1, env=None))


def test_l_env_var_arrive_bien_au_backend(monkeypatch):
    monkeypatch.setenv("RELIQUARY_BAKE_IN_FLIGHT", "8")
    b = _BackendC1()
    _drive(monkeypatch, b)
    assert b.vu["max_in_flight"] == 8
    assert callable(b.vu["should_admit"]), (
        "la garde de fenêtre doit être passée, sinon on injecte au flip"
    )
    assert b.vu["should_admit"]() is True, "fenêtre vivante => robinet ouvert"


def test_eteint_par_defaut_rien_n_est_passe(monkeypatch):
    monkeypatch.delenv("RELIQUARY_BAKE_IN_FLIGHT", raising=False)
    b = _BackendC1()
    _drive(monkeypatch, b)
    assert b.vu["max_in_flight"] == 0, "défaut = comportement historique"
    assert b.vu["should_admit"] is None


def test_backend_sans_c1_reste_utilisable(monkeypatch):
    """Repli partiel : un backend d'avant C1 ne doit pas lever TypeError."""
    monkeypatch.setenv("RELIQUARY_BAKE_IN_FLIGHT", "8")
    b = _BackendSansC1()
    _drive(monkeypatch, b)
    assert b.appele, "le bake doit tourner même si le backend ignore C1"
