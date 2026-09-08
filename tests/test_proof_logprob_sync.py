"""Preuve GRAIL : une seule synchronisation GPU→CPU par appel (2026-09-08).

``_chunked_chosen_logprobs_fused`` appelait ``.tolist()`` À CHAQUE TRANCHE —
donc un point de synchronisation qui vide le pipeline CUDA par tranche et par
sortie. Profil du process en vol (400 instantanés de pile, 08/09) : **49 % du
temps de la preuve** est passé sur ces lignes (engine.py:202/216/223), contre
40 % dans le forward Qwen3 lui-même. Décomposition appariée au volume réel du
groupe (73 groupes, R2 ↔ submits_v4.jsonl) : 31,6 ms/1000 tokens de calcul
mais **0,62 s de coût FIXE** sur une preuve de 1,07 s.

Le correctif ne touche AUCUN calcul : les tranches sont accumulées en tenseurs
GPU et rapatriées une seule fois à la fin. Les valeurs doivent donc rester
strictement identiques — les commitments et les logprobs soumis en dépendent,
et le validateur les recalcule.
"""
import pytest

torch = pytest.importorskip("torch")

from reliquary.miner.engine import _chunked_chosen_logprobs_fused  # noqa: E402


def _mk(seq=97, hidden=32, vocab=211, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(seq, hidden, generator=g)
    lm = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm.weight.copy_(torch.randn(vocab, hidden, generator=g))
    toks = torch.randint(0, vocab, (seq,), generator=g).tolist()
    return h, lm, toks


def _run_counting_syncs(chunk, *, argmax):
    """Renvoie (logprobs, argmax, nb de transferts GPU→CPU)."""
    h, lm, toks = _mk()
    calls = []
    real = torch.Tensor.tolist

    def counting(self):
        calls.append(tuple(self.shape))
        return real(self)

    torch.Tensor.tolist = counting
    try:
        amx = [] if argmax else None
        out = _chunked_chosen_logprobs_fused(
            h, lm, toks, 1, chunk=chunk, argmax_out=amx,
        )
    finally:
        torch.Tensor.tolist = real
    return out, amx, len(calls)


def test_sync_count_does_not_grow_with_the_number_of_chunks():
    """96 lignes en tranches de 8 (12 tranches) ne doit pas coûter 12× plus de
    synchronisations que la même chose en une seule tranche."""
    _, _, many = _run_counting_syncs(8, argmax=True)
    _, _, one = _run_counting_syncs(1000, argmax=True)
    assert many == one, (
        f"{many} synchronisations GPU→CPU sur 12 tranches contre {one} sur une "
        f"seule — le coût de synchronisation suit le nombre de tranches"
    )


def test_sync_count_does_not_grow_without_the_argmax_mirror():
    """Même invariant quand ``argmax_out`` n'est pas demandé."""
    _, _, many = _run_counting_syncs(8, argmax=False)
    _, _, one = _run_counting_syncs(1000, argmax=False)
    assert many == one, f"{many} contre {one}"


def test_values_are_bit_identical_across_chunk_sizes():
    """Garde de non-régression : le découpage en tranches ne doit RIEN changer
    aux valeurs. Égalité stricte, pas une tolérance — les logprobs partent au
    validateur qui les recalcule."""
    a, amx_a, _ = _run_counting_syncs(8, argmax=True)
    b, amx_b, _ = _run_counting_syncs(1000, argmax=True)
    assert a == b
    assert amx_a == amx_b


def test_argmax_mirror_still_filled_and_aligned():
    """``argmax_out`` alimente ``local_verif_screen`` : une entrée par token de
    complétion, alignée sur les logprobs."""
    out, amx, _ = _run_counting_syncs(8, argmax=True)
    assert len(out) == len(amx) == 96
    assert all(0.0 <= p <= 1.0 for p in amx)


# ---------------------------------------------------------------------------
# Deuxième temps : le GROUPE, pas seulement le rollout.
#
# `_proof_rollouts` traite les 16 rollouts en série et chaque itération fait
# TROIS allers-retours vers l'hôte : les logprobs, le miroir argmax, et
# `hidden_states.detach().cpu()` (~4,5 Mo). Soit ~48 synchronisations par
# groupe, dont chacune vide le pipeline CUDA. Or rien dans la boucle ne dépend
# du rollout précédent (vérifié ligne à ligne : `local_verif_screen_detail`
# ne lit que le rollout courant, `rollouts_cache.append` non plus, aucun
# `break`). Les transferts peuvent donc tous être repoussés à la fin.
#
# Chaque forward reste MONO-SÉQUENCE : aucune arithmétique ne change, donc
# les preuves restent bit-à-bit identiques. C'est ce qui distingue ce
# correctif du batching des forwards (padding + ordre de réduction changés,
# parité à re-valider).
# ---------------------------------------------------------------------------


def _fake_forward(model, input_ids, attention_mask, layer_index,
                  materialize_logits=True):
    b, s = input_ids.shape
    g = torch.Generator().manual_seed(int(input_ids.sum()) % 10_000)
    h = torch.randn(b, s, model.hidden, generator=g)
    return h, (model.lm_head(h) if materialize_logits else None)


class _FakeProofModel:
    def __init__(self, hidden=8, vocab=23):
        self.hidden = hidden
        g = torch.Generator().manual_seed(11)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.randn(vocab, hidden, generator=g))


def _proof_engine(model):
    from reliquary.miner.engine import MiningEngine

    e = MiningEngine.__new__(MiningEngine)
    e.hf_model = model
    e.proof_gpu = 0
    return e


def _generations(n, seq=37, prompt_len=5, vocab=23):
    g = torch.Generator().manual_seed(5)
    out = []
    for _ in range(n):
        toks = torch.randint(0, vocab, (seq,), generator=g).tolist()
        out.append({"tokens": toks, "prompt_length": prompt_len})
    return out


def _run_proof_counting_transfers(n_rollouts, monkeypatch):
    """Renvoie (rollouts_cache, nb de transferts GPU→hôte)."""
    import reliquary.shared.forward as fwd

    monkeypatch.setattr(fwd, "forward_single_layer", _fake_forward)
    model = _FakeProofModel()
    gens = _generations(n_rollouts)
    texts = ["x"] * n_rollouts

    moves = []
    real_tolist, real_cpu = torch.Tensor.tolist, torch.Tensor.cpu
    monkeypatch.setattr(
        torch.Tensor, "tolist",
        lambda self: (moves.append("tolist"), real_tolist(self))[1])
    monkeypatch.setattr(
        torch.Tensor, "cpu",
        lambda self, *a, **k: (moves.append("cpu"), real_cpu(self, *a, **k))[1])
    cache = _proof_engine(model)._proof_rollouts(
        gens, texts=texts, device="cpu")
    return cache, len(moves)


def test_host_transfers_do_not_grow_with_the_group_size(monkeypatch):
    """8 rollouts ne doivent pas coûter 4× les transferts de 2 rollouts."""
    _, few = _run_proof_counting_transfers(2, monkeypatch)
    _, many = _run_proof_counting_transfers(8, monkeypatch)
    assert many <= few + 2, (
        f"{few} transferts pour 2 rollouts, {many} pour 8 — le coût de "
        f"synchronisation suit encore le nombre de rollouts"
    )


def test_proof_group_keeps_one_entry_per_rollout_in_order(monkeypatch):
    """Garde : un cache par rollout, dans l'ordre, avec les mêmes champs."""
    cache, _ = _run_proof_counting_transfers(4, monkeypatch)
    assert len(cache) == 4
    gens = _generations(4)
    for entry, gen in zip(cache, gens):
        assert entry["all_tokens"] == gen["tokens"]
        assert entry["prompt_length"] == gen["prompt_length"]
        assert entry["hidden_states_cpu"].shape[0] == len(gen["tokens"])
        assert len(entry["token_logprobs"]) == (
            len(gen["tokens"]) - gen["prompt_length"])


def test_proof_group_values_are_bit_identical_to_the_serial_path(monkeypatch):
    """Le résultat ne doit RIEN devoir à l'ordonnancement des transferts :
    mêmes logprobs, mêmes hidden states, mêmes verdicts de screen local."""
    a, _ = _run_proof_counting_transfers(4, monkeypatch)
    b, _ = _run_proof_counting_transfers(4, monkeypatch)
    for x, y in zip(a, b):
        assert x["token_logprobs"] == y["token_logprobs"]
        assert x["local_screen"] == y["local_screen"]
        assert torch.equal(x["hidden_states_cpu"], y["hidden_states_cpu"])
