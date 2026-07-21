"""Budget-Forced Termination (BFT) rollout assembly — v7 / cot-2b.

Ported VERBATIM from the validator reference miner
(``reliquary/miner/engine.py``: ``_bft_assemble_rollouts`` + ``_rollout_metadata``
@ d9471f2). Byte-exact behaviour is required so the validator carve-out
(``validate_force_span``) and the GRPO trainer mask locate the injected FORCE
span at the identical indices. Do NOT edit the logic.
"""


def bft_applicable(env_name) -> bool:
    """BFT (thinking-budget force-termination) applies to the math env only.
    The validator carve-out for forced rollouts is scoped to openmathinstruct
    (validator ca3ac67 / #103), so we must NOT force-terminate code rollouts.
    ``None`` = single-env deployment → treated as math."""
    from reliquary.constants import BFT_ENABLED

    return BFT_ENABLED and (env_name is None or env_name == "openmathinstruct")


def phase1_max_new_tokens(max_new_tokens: int, env_name) -> int:
    """Phase-1 (thinking) token budget.

    For a BFT env this is EXACTLY BFT_THINKING_BUDGET — NOT
    ``min(max_new_tokens, BFT_THINKING_BUDGET)``. The validator pins a forced
    rollout's FORCE span to ``prompt_len + BFT_THINKING_BUDGET`` and rejects any
    other offset as TOKEN_TAMPERED, so a smaller phase-1 cap (e.g. the
    vllm_backend ``max_tokens=1500`` default) would place the span short and get
    100 % of forced rollouts rejected. Non-BFT (code) envs keep the miner's
    configured max_new_tokens."""
    from reliquary.constants import BFT_THINKING_BUDGET

    if bft_applicable(env_name):
        return BFT_THINKING_BUDGET
    return max_new_tokens


def bft_rollouts_from_completions(
    completions, prompt_tokens, *, model, think_close_ids, force_ids,
    eos_ids, answer_budget, randomness, hotkey, prompt_idx, checkpoint_hash,
    gen_kwargs=None,
):
    """Adapter for the vLLM/list generation paths: pad variable-length phase-1
    ``completions`` (each = ``prompt_tokens + gen`` as a token list) into one
    batch tensor and run :func:`bft_assemble_rollouts`.

    INVARIANT (caller's responsibility): a completion that did NOT hit EOS must
    be at the batch's max length (it ran to the phase-1 cap), so it carries no
    trailing pad into the natural-close / forced branch — only EOS-finished rows
    are padded, and those are trimmed at their real first EOS.
    """
    import torch

    pad = min(eos_ids) if eos_ids else 0
    width = max(len(c) for c in completions)
    rows = [list(c) + [pad] * (width - len(c)) for c in completions]
    phase1_tensor = torch.tensor(rows)
    return bft_assemble_rollouts(
        model=model, phase1_tensor=phase1_tensor, prompt_tokens=prompt_tokens,
        think_close_ids=think_close_ids, force_ids=force_ids, eos_ids=eos_ids,
        answer_budget=answer_budget, randomness=randomness, hotkey=hotkey,
        prompt_idx=prompt_idx, checkpoint_hash=checkpoint_hash, gen_kwargs=gen_kwargs,
    )


def bft_assemble_rollouts(
    *, model, phase1_tensor, prompt_tokens, think_close_ids, force_ids,
    eos_ids, answer_budget, randomness, hotkey, prompt_idx, checkpoint_hash,
    gen_kwargs=None,
):
    """Budget-Forced Termination assembly.

    Rows that hit EOS are kept as-is (truncated at first EOS). Rows that emitted
    ``</think>`` but did not hit EOS are naturally closed and continue sampling
    the answer for ``answer_budget`` tokens. Rows that did not close thinking are
    *forced*: ``force_ids`` are appended and the same phase-2 generation samples
    the boxed answer. Returns one rollout dict per row with ``forced`` and, for
    forced rows, ``force_span`` = (start, end) of the injected ids within
    ``tokens`` (so the validator carve-out and trainer mask can locate them).

    Phase-2 answer tokens are drawn from the same protocol forced-seed stream as
    phase-1, resuming at each row's own completion offset (its primed length past
    the prompt). The injected ``force_ids`` are not sampled and the validator
    excludes that span from the seed-consistency check.
    """
    import torch

    from reliquary.miner.forced_seed_sampler import (
        ForcedSeedLogitsProcessor, forced_seed_generate_kwargs, phase2_base_offsets,
    )
    from reliquary.shared.modeling import first_eos_index, has_think_close

    plen = len(prompt_tokens)
    n = int(phase1_tensor.shape[0])
    close_set = {int(t) for t in think_close_ids}
    force_ids = [int(t) for t in force_ids]

    out: list = [None] * n
    unfinished_idx: list[int] = []
    unfinished_primed: list[list[int]] = []
    unfinished_force_spans: list[tuple[int, int] | None] = []
    for i in range(n):
        seq = phase1_tensor[i].tolist()
        gen = seq[plen:]
        fe = first_eos_index(gen, eos_ids)
        if fe is not None:
            # Finished on EOS: trim padding/trailing garbage and keep as-is.
            gen = gen[: fe + 1]
            out[i] = {"tokens": prompt_tokens + gen,
                      "prompt_length": plen, "forced": False}
        elif has_think_close(gen, close_set):
            # Naturally closed thinking but did not EOS within phase-1. Continue
            # into the answer phase without injecting FORCE and without a carve.
            unfinished_idx.append(i)
            unfinished_primed.append(seq)
            unfinished_force_spans.append(None)
        else:
            force_start = len(seq)
            primed = seq + force_ids
            unfinished_idx.append(i)
            unfinished_primed.append(primed)
            unfinished_force_spans.append((force_start, force_start + len(force_ids)))

    if unfinished_primed:
        width = max(len(p) for p in unfinished_primed)
        pad = min(eos_ids) if eos_ids else 0
        rows = [[pad] * (width - len(p)) + p for p in unfinished_primed]
        mask = [[0] * (width - len(p)) + [1] * len(p) for p in unfinished_primed]
        device = getattr(model, "device", "cpu")
        proc = ForcedSeedLogitsProcessor(
            randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_indices=list(unfinished_idx),
            base_offsets=phase2_base_offsets(
                [len(p) for p in unfinished_primed], plen,
            ),
            start_len=width,
        )
        ans = model.generate(
            torch.tensor(rows, device=device),
            attention_mask=torch.tensor(mask, device=device),
            max_new_tokens=answer_budget,
            **forced_seed_generate_kwargs(gen_kwargs or {}, proc),
        )
        for k, i in enumerate(unfinished_idx):
            primed = unfinished_primed[k]
            tail = ans[k].tolist()[width:]
            fe = first_eos_index(tail, eos_ids)
            tail = tail[: fe + 1] if fe is not None else tail
            forced_span = unfinished_force_spans[k]
            rollout = {"tokens": primed + tail, "prompt_length": plen,
                       "forced": forced_span is not None}
            if forced_span is not None:
                rollout["force_span"] = forced_span
            out[i] = rollout
    return out


def rollout_metadata(generation: dict, token_logprobs: list) -> dict:
    """Per-rollout metadata embedded in the GRAIL commit. Carries the BFT
    ``forced`` flag and ``force_span`` so the validator carve-out and trainer
    mask can locate the injected span."""
    prompt_length = int(generation["prompt_length"])
    all_tokens = generation["tokens"]
    force_span = generation.get("force_span")
    return {
        "prompt_length": prompt_length,
        "completion_length": len(all_tokens) - prompt_length,
        "success": True,
        "total_reward": 0.0,
        "advantage": 0.0,
        "token_logprobs": token_logprobs,
        "forced": bool(generation.get("forced", False)),
        "force_span": list(force_span) if force_span else None,
    }
