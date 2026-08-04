"""Protocol-fixed sampler shared by miner (generation) and validator (verification).

The per-position draw is a public deterministic function of window randomness, so
there is exactly one legal generation per (miner, prompt, rollout, window). A rollout
not generated from this draw is detectable by teacher-forced consistency.
"""
from __future__ import annotations

import hashlib

import torch

from reliquary.constants import FORCED_SEED_DOMAIN


def warp(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature -> top-k -> top-p, returned in canonical (token-id ascending) order."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.numel())
        kth = torch.topk(lg, k).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))  # include crossing token
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum()


def warp_fast(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Bit-exact top_k-restricted equivalent of :func:`warp` (single row).

    Same result as ``warp`` but the ``O(V log V)`` top-p sort runs on the ~top_k
    survivors instead of the full ``V``-token vocab (V≈151k, only ~20 non-zero
    after top_k). This is the throughput lever: the full-vocab sort was ~90% of
    the forced-seed processor cost on the 2B.

    PARITY — must equal ``warp`` BIT-FOR-BIT (shared with the validator):
      * softmax denominator + final renorm sum stay FULL-vocab (reordering a
        float reduction changes it at the ULP — the trap the 3 prior attempts
        hit). We only shrink the SORT and its sequential cumsum, which are
        order-preserving on the non-zeros.
      * survivors = ``probs > 0`` (exactly ``lg >= kth``, ties at the threshold
        included) so we never drop a token ``warp`` kept.
    ⚠️ Verified bit-exact on CPU; GPU reduction kernels differ → RE-VERIFY on the
    target GPU before shipping.
    """
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.numel())
        kth = torch.topk(lg, k).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        surv_idx = (probs > 0).nonzero(as_tuple=True)[0]      # ~top_k survivors
        surv = probs.index_select(0, surv_idx)
        sp, order = torch.sort(surv, descending=True)          # sort ~20, not V
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))
        kept_ids = surv_idx.index_select(0, order)
        probs = torch.zeros_like(probs).scatter(-1, kept_ids, sp)
    return probs / probs.sum()


def pick(probs: torch.Tensor, u: float) -> int:
    """First token id whose cumulative probability exceeds u (inverse-CDF)."""
    cdf = torch.cumsum(probs, dim=-1)
    u_tensor = torch.tensor(float(u), device=cdf.device, dtype=cdf.dtype)
    idx = int(torch.searchsorted(cdf, u_tensor, right=True))
    return min(idx, probs.numel() - 1)


def force_rows_batched(logits: torch.Tensor, us, *, t: float, top_k: int,
                       top_p: float) -> torch.Tensor:
    """Vectorised forced pick for a WHOLE batch — same token as
    ``pick(warp(row), u)`` per row, but in one tensor pass instead of a Python
    loop, and the sort/cumsum restricted to the top_k non-zero entries.

    Returns a LongTensor of forced token-ids, one per row of ``logits``
    ([n, vocab]). ``us`` is a length-n sequence of floats.

    Equivalence with warp+pick (must stay bit-exact — shared with the
    validator):
      * warp: temperature, then top_k mask, softmax, top_p mask keeping the
        crossing token, renormalise. In canonical (token-id) order the CDF that
        ``pick`` walks is the same whether we scatter back to full vocab or keep
        the survivors sorted by ascending token-id.
      * pick: first token id whose cumulative prob (ascending token-id order)
        exceeds u. We reproduce it on the top_k survivors sorted by token-id.
    """
    n, vocab = logits.shape
    device = logits.device

    # EXACTEMENT les mêmes opérations que warp(), mais sur [n, vocab] au lieu
    # d'une ligne — dim=-1 partout, aucune arithmétique réordonnée. Le gain vient
    # de faire tomber la boucle Python sur n, pas de changer le calcul par ligne.
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, vocab)
        kth = torch.topk(lg, k, dim=-1).values[..., -1:]          # [n, 1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)                              # [n, vocab]
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    probs = probs / probs.sum(dim=-1, keepdim=True)               # [n, vocab]

    # pick() vectorisé : inverse-CDF en ordre token-id croissant, une recherche
    # binaire par ligne. cumsum sur le vocab complet reste identique à pick().
    cdf = torch.cumsum(probs, dim=-1)                              # [n, vocab]
    # ``us`` accepte aussi un tenseur float32 déjà résident sur le device
    # (processeur device-resident §3.2 : staging épinglé + copy_ non bloquante)
    # — même arrondi f64→f32 que le chemin liste, donc mêmes picks au bit près.
    if isinstance(us, torch.Tensor):
        u_t = us.to(device=device, dtype=cdf.dtype).unsqueeze(-1)  # [n, 1]
    else:
        u_t = torch.as_tensor([float(x) for x in us], device=device,
                              dtype=cdf.dtype).unsqueeze(-1)       # [n, 1]
    idx = torch.searchsorted(cdf, u_t, right=True).squeeze(-1)     # [n]
    return idx.clamp(max=vocab - 1)


def force_rows_batched_fast(logits: torch.Tensor, us, *, t: float, top_k: int,
                            top_p: float) -> torch.Tensor:
    """Bit-exact fast path for :func:`force_rows_batched` — the per-step
    [n, vocab] sort and cdf-cumsum are restricted to the ~top_k survivors.

    Parity recipe (the warp_fast lessons, batched):
      * every REDUCTION stays full-vocab (softmax denominator, renorm sum) —
        reordering a float reduction shifts ULPs;
      * survivors come from ``torch.topk`` (FIXED-size output → no ``nonzero``,
        no CPU-GPU sync). ``topk``'s descending order equals the reference
        sort's non-zero prefix when there are no ties;
      * the restricted ascending-token-id cumsum is bit-equal to the full cdf
        at survivor positions because the interleaved zeros add exactly 0.0;
      * the reference's ``clamp(vocab-1)`` overflow behaviour (u beyond the
        accumulated mass) is reproduced explicitly;
      * ANY tie at the top_k threshold or inside the top_k values → wholesale
        fallback to the reference (bf16 logits do collide occasionally; the
        one boolean ``.any()`` guard costs a single tiny sync per call).
    """
    n, vocab = logits.shape
    if not (top_k and 0 < top_k < vocab):
        return force_rows_batched(logits, us, t=t, top_k=top_k, top_p=top_p)
    lg = logits.float() / float(t)
    k = min(top_k, vocab)
    topv, topi = torch.topk(lg, k, dim=-1)                     # [n, k] desc
    kth = topv[..., -1:]
    keep = lg >= kth                                           # [n, vocab]
    boundary_tie = (keep.sum(dim=-1) > k).any()
    inner_tie = (topv[..., 1:] == topv[..., :-1]).any()
    if bool(boundary_tie) or bool(inner_tie):
        return force_rows_batched(logits, us, t=t, top_k=top_k, top_p=top_p)
    lg = torch.where(~keep, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)                          # full-V denominator
    if top_p and top_p < 1.0:
        sp = probs.gather(-1, topi)                            # == ref sort prefix
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(-1, topi, sp)
    probs = probs / probs.sum(dim=-1, keepdim=True)            # full-V sum
    # Restricted inverse-CDF over survivors in ascending token-id order.
    asc_i, _ = torch.sort(topi, dim=-1)                        # sort of k ints
    asc_p = probs.gather(-1, asc_i)
    cdf_k = torch.cumsum(asc_p, dim=-1)                        # [n, k]
    u_t = torch.as_tensor([float(x) for x in us], device=logits.device,
                          dtype=cdf_k.dtype).unsqueeze(-1)
    pos = torch.searchsorted(cdf_k, u_t, right=True).squeeze(-1)   # [n] in 0..k
    overflow = pos >= k
    picks = asc_i.gather(-1, pos.clamp(max=k - 1).unsqueeze(-1)).squeeze(-1)
    return torch.where(overflow, torch.full_like(picks, vocab - 1), picks)


def _lp(b: bytes) -> bytes:
    return len(b).to_bytes(2, "big") + b


def u_at(randomness: str, prompt_idx: int, checkpoint_hash: str,
         rollout_index: int, t: int) -> float:
    """Public uniform in [0, 1) for rollout `rollout_index`, completion position `t`.

    v2 (difficulty-auction, validator 6f32673): the hotkey is deliberately NOT
    hashed. The forced group for a prompt is therefore identical for every miner
    in the window, so one operator's N hotkeys yield N copies of the same draw —
    variance farming (best-of-N distinct draws) is impossible. Anti-pregeneration
    still holds: `randomness` is unknown until the window opens. Keyed on the v2
    domain (`FORCED_SEED_DOMAIN`) so v1 and v2 streams never collide."""
    msg = (FORCED_SEED_DOMAIN.encode()
           + _lp(randomness.encode())
           + int(prompt_idx).to_bytes(8, "big")
           + _lp(checkpoint_hash.encode())
           + int(rollout_index).to_bytes(4, "big")
           + int(t).to_bytes(4, "big"))
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big") / 2.0**64


def _warp_batch(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Row-batched ``warp``: logits [n, vocab] -> probs [n, vocab], bit-identical
    per row to the 1-D ``warp`` (each op is independent along dim=-1) but with no
    per-row Python loop."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.shape[-1])
        kth = torch.topk(lg, k, dim=-1).values[..., -1:]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum(dim=-1, keepdim=True)


def seed_consistency(logits: torch.Tensor, token_ids: list[int], u_values: list[float], *,
                     t: float, top_k: int, top_p: float,
                     stochastic_threshold: float) -> tuple[int, int]:
    """Teacher-forced check. logits is [n, vocab] predicting token_ids[i] at u_values[i].
    Counts stochastic positions (max_prob < threshold) and how many match the forced pick.

    Fully vectorized: one batched warp + one batched inverse-CDF over all n
    positions, with a single device sync for the two returned counts (the
    per-position loop used to force a GPU->CPU sync at every step, on the serial
    GRAIL-verify hot path)."""
    n = min(len(token_ids), len(u_values), int(logits.shape[0]))
    if n == 0:
        return 0, 0
    probs = _warp_batch(logits[:n], t=t, top_k=top_k, top_p=top_p)         # [n, vocab]
    stochastic = probs.max(dim=-1).values < stochastic_threshold           # [n] bool
    cdf = torch.cumsum(probs, dim=-1)
    u = torch.tensor([float(x) for x in u_values[:n]],
                     device=cdf.device, dtype=cdf.dtype).unsqueeze(-1)     # [n, 1]
    picks = torch.searchsorted(cdf, u, right=True).squeeze(-1).clamp(max=probs.shape[-1] - 1)
    toks = torch.tensor([int(x) for x in token_ids[:n]],
                        device=picks.device, dtype=picks.dtype)            # [n]
    matched = stochastic & (picks == toks)
    return int(stochastic.sum().item()), int(matched.sum().item())
