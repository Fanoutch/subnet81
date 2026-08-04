"""Les chemins forced-phase1 vLLM doivent INCLURE le token d'arrêt (EOS) dans
les tokens retournés — ``include_stop_str_in_output=True``.

Verdicts live 2026-08-03 (fenêtre 27585, 4B/v3) : 3/3 submissions ingress-OK
mais verdict final ``bad_termination``. Cause : ``generate_forced_phase1`` et
``generate_forced_phase1_multi`` construisent leurs SamplingParams inline SANS
``include_stop_str_in_output`` → vLLM s'arrête sur l'EOS mais l'EXCLUT des
token_ids → le rollout soumis ne finit pas par un EOS → le validateur le
classe non-terminé. Le chemin générique ``_build_sampling_params`` du même
fichier documente et évite précisément ce piège — les chemins forced doivent
faire pareil.
"""
import sys
import types

import pytest


class _RecordingSamplingParams:
    def __init__(self, **kw):
        self.kw = kw

    def __getattr__(self, k):
        try:
            return self.__dict__["kw"][k]
        except KeyError as e:
            raise AttributeError(k) from e


class _FakeOut:
    """Simule le comportement vLLM observé live (fenêtres 27585/27586) : la
    séquence s'arrête sur un stop-token SPÉCIAL, vLLM le RETIRE des token_ids
    et le signale via ``stop_reason``. include_stop_str_in_output n'y change
    rien pour les tokens spéciaux."""

    def __init__(self, token_ids=(1, 2, 3), stop_reason=151645,
                 finish_reason="stop"):
        self.token_ids = list(token_ids)
        self.stop_reason = stop_reason
        self.finish_reason = finish_reason


class _FakeROut:
    def __init__(self, out=None):
        self.outputs = [out or _FakeOut()]


class _FakeLLM:
    def __init__(self, recorded):
        self._recorded = recorded

    def generate(self, prompts, sampling_params=None):
        sps = sampling_params if isinstance(sampling_params, list) \
            else [sampling_params] * len(prompts)
        self._recorded.extend(sps)
        return [_FakeROut() for _ in prompts]


@pytest.fixture()
def stub_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = _RecordingSamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    vllm.inputs = inputs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    yield


def _mk_backend(recorded):
    from reliquary.miner.vllm_backend import VLLMBackend
    b = VLLMBackend.__new__(VLLMBackend)
    b._llm = _FakeLLM(recorded)
    b._ensure_loaded = lambda: None
    return b


def test_phase1_multi_includes_stop_token(stub_vllm):
    recorded = []
    b = _mk_backend(recorded)
    b.generate_forced_phase1_multi(
        [[10, 11], [12, 13, 14]], prompt_indices=[5, 6],
        randomness="ab" * 32, checkpoint_hash="ck", m_rollouts=2,
        max_tokens=64, stop_token_ids=[151645],
    )
    assert recorded, "SamplingParams capturés"
    for sp in recorded:
        assert sp.kw.get("include_stop_str_in_output") is True, (
            "phase-1 multi : l'EOS d'arrêt doit rester dans les token_ids "
            "(sinon bad_termination au verdict)"
        )
        assert sp.kw.get("stop_token_ids") == [151645]


def test_phase1_single_includes_stop_token(stub_vllm):
    recorded = []
    b = _mk_backend(recorded)
    b.generate_forced_phase1(
        [10, 11, 12], randomness="ab" * 32, prompt_idx=5,
        checkpoint_hash="ck", m_rollouts=2, max_tokens=64,
        stop_token_ids=[151645],
    )
    assert recorded
    for sp in recorded:
        assert sp.kw.get("include_stop_str_in_output") is True


def test_phase1_multi_reappends_stripped_stop_token(stub_vllm):
    """Le stop-token retiré par vLLM (stop_reason) doit être RÉ-APPENDU :
    c'est le pick forcé réellement généré, le validateur exige exactement un
    EOS en dernière position (admission._classify_termination)."""
    recorded = []
    b = _mk_backend(recorded)
    groups = b.generate_forced_phase1_multi(
        [[10, 11]], prompt_indices=[5], randomness="ab" * 32,
        checkpoint_hash="ck", m_rollouts=2, max_tokens=64,
        stop_token_ids=[151645],
    )
    for rollout in groups[0]:
        assert rollout[-1] == 151645, (
            f"stop-token absent de la fin: {rollout}"
        )


def test_phase1_single_reappends_stripped_stop_token(stub_vllm):
    recorded = []
    b = _mk_backend(recorded)
    rollouts = b.generate_forced_phase1(
        [10, 11, 12], randomness="ab" * 32, prompt_idx=5,
        checkpoint_hash="ck", m_rollouts=2, max_tokens=64,
        stop_token_ids=[151645],
    )
    for r in rollouts:
        assert r[-1] == 151645


def test_no_double_append_when_vllm_already_included(stub_vllm):
    """Si vLLM inclut déjà le stop-token (versions/configs qui le font), on ne
    doit PAS le doubler — deux EOS = bad_termination aussi (règle 'exactement
    un, en dernière position')."""
    recorded = []
    b = _mk_backend(recorded)
    b._llm.generate = lambda prompts, sampling_params=None: [
        _FakeROut(_FakeOut(token_ids=[1, 2, 151645], stop_reason=151645))
        for _ in prompts
    ]
    groups = b.generate_forced_phase1_multi(
        [[10, 11]], prompt_indices=[5], randomness="ab" * 32,
        checkpoint_hash="ck", m_rollouts=1, max_tokens=64,
        stop_token_ids=[151645],
    )
    assert groups[0][0] == [1, 2, 151645], "pas de double EOS"


def test_phase1_paths_set_ignore_eos(stub_vllm):
    """ignore_eos=True sur les chemins forced : l'arrêt EOS du MOTEUR vLLM
    (stop_reason=None, token strippé, rien à ré-appendre) est désactivé pour
    que seul stop_token_ids stoppe — stop_reason porte alors TOUJOURS l'ID du
    token d'arrêt et la restauration est infaillible. Verdicts 27590 : le
    ré-append seul n'a pas suffi car l'arrêt passait par l'EOS moteur."""
    recorded = []
    b = _mk_backend(recorded)
    b.generate_forced_phase1_multi(
        [[10, 11]], prompt_indices=[5], randomness="ab" * 32,
        checkpoint_hash="ck", m_rollouts=1, max_tokens=64,
        stop_token_ids=[151645],
    )
    b.generate_forced_phase1(
        [10, 11], randomness="ab" * 32, prompt_idx=5,
        checkpoint_hash="ck", m_rollouts=1, max_tokens=64,
        stop_token_ids=[151645],
    )
    assert recorded
    for sp in recorded:
        assert sp.kw.get("ignore_eos") is True, (
            "ignore_eos manquant → l'EOS moteur strippe le token sans "
            "stop_reason et le rollout part sans EOS final"
        )


def test_reconstructs_model_eos_when_stop_reason_is_none(stub_vllm):
    """CAS CONSTRUCTIF : vLLM dit « je me suis arrêté » (finish_reason='stop')
    mais ne dit pas sur quoi (stop_reason=None) → c'est l'EOS du MODÈLE qui a
    stoppé, et il a été retiré. Son identifiant est connu et unique pour ce
    checkpoint (generation_config), donc on peut le RECONSTRUIRE.

    Sans ça, ces rollouts n'ont aucun EOS : la troncature n'a rien à couper et
    la garde locale les jette → le mineur ne soumet plus RIEN (panne
    silencieuse). C'est la différence entre « on n'envoie pas de mauvais » et
    « on envoie du bon »."""
    import os
    recorded = []
    b = _mk_backend(recorded)
    b._llm.generate = lambda prompts, sampling_params=None: [
        _FakeROut(_FakeOut(token_ids=[1, 2, 3], stop_reason=None,
                           finish_reason="stop"))
        for _ in prompts
    ]
    call = lambda: b.generate_forced_phase1_multi(
        [[10, 11]], prompt_indices=[5], randomness="ab" * 32,
        checkpoint_hash="ck", m_rollouts=1, max_tokens=64,
        stop_token_ids=[248044, 248046], primary_eos_id=248044,
    )
    # DÉFAUT = OFF : on n'invente rien tant que le probe n'a pas identifié le
    # token qui arrête réellement (2 EOS déclarés → apposer le mauvais serait
    # PIRE qu'un groupe jeté : token jamais généré → SEED_MISMATCH).
    os.environ.pop("RELIQUARY_RECONSTRUCT_EOS", None)
    assert call()[0][0] == [1, 2, 3], "défaut OFF : aucune reconstruction"

    os.environ["RELIQUARY_RECONSTRUCT_EOS"] = "1"
    try:
        assert call()[0][0] == [1, 2, 3, 248044], (
            "activé : l'EOS modèle doit être reconstruit"
        )
    finally:
        os.environ.pop("RELIQUARY_RECONSTRUCT_EOS", None)


def test_no_reconstruction_without_primary_eos_hint(stub_vllm):
    """Sans indice explicite on n'invente RIEN : mieux vaut jeter le rollout
    que deviner un token qui ferait échouer le teacher-forcing."""
    recorded = []
    b = _mk_backend(recorded)
    b._llm.generate = lambda prompts, sampling_params=None: [
        _FakeROut(_FakeOut(token_ids=[1, 2, 3], stop_reason=None,
                           finish_reason="stop"))
        for _ in prompts
    ]
    groups = b.generate_forced_phase1_multi(
        [[10, 11]], prompt_indices=[5], randomness="ab" * 32,
        checkpoint_hash="ck", m_rollouts=1, max_tokens=64,
        stop_token_ids=[248044, 248046],
    )
    assert groups[0][0] == [1, 2, 3]


def test_no_append_on_length_finish(stub_vllm):
    """Cap atteint (finish_reason='length', stop_reason=None) : ne rien
    ajouter — le rollout est tronqué, la garde locale le jettera."""
    recorded = []
    b = _mk_backend(recorded)
    b._llm.generate = lambda prompts, sampling_params=None: [
        _FakeROut(_FakeOut(token_ids=[1, 2, 3], stop_reason=None,
                           finish_reason="length"))
        for _ in prompts
    ]
    groups = b.generate_forced_phase1_multi(
        [[10, 11]], prompt_indices=[5], randomness="ab" * 32,
        checkpoint_hash="ck", m_rollouts=1, max_tokens=64,
        stop_token_ids=[151645],
    )
    assert groups[0][0] == [1, 2, 3], "rien ajouté sur un arrêt par cap"
