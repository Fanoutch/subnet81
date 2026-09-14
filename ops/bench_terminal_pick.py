"""Banc « EOS final » : accord vLLM (chemin de prod) ↔ forward HF, par réglage.

Pourquoi (14/09) : le validateur exige depuis #253 que le dernier token stop de
chaque rollout soit EXACTEMENT le pick forced-seed recalculé sur SON forward
HF. En prod ~7 % de nos rollouts échouent (garde ``local_terminal_pick``).
Ce banc mesure ce taux hors prod pour comparer des réglages vLLM, sur le même
chemin que le mineur (``VLLMBackend.generate_forced_phase1_multi_stream``,
vrais prompts code enveloppés v6, lots de N×16, génération jusqu'à l'EOS).

Sorties (une ligne JSON ``[bench] RESULT``) :
  terminal_exact  part des rollouts finis sur stop dont le pick HF == token
  group_ok        part des groupes dont les 16 rollouts passent
  pos_match       accord moyen sur les positions stochastiques (= la gate)
  edges_fail      quantiles de la distance au bord CDF des rollouts ratés

Usage (mineur ARRÊTÉ, env de prod) :
  BENCH_PROMPTS=8 BENCH_LABEL=base python ops/bench_terminal_pick.py
Réglages testés par variables : RELIQUARY_VLLM_DISABLE_CASCADE,
VLLM_BATCH_INVARIANT, RELIQUARY_VLLM_CUDA_GRAPHS, RELIQUARY_FS_GRAPH,
BENCH_HF_ATTN (sdpa|eager).
"""
from __future__ import annotations

import gc
import json
import os
import time

import torch


def _prompt_indices(n: int, env_name: str) -> list[int]:
    path = os.environ.get("BENCH_SAMPLES", "/workspace/samples_v4.jsonl")
    seen: list[int] = []
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 40_000_000))
        lines = fh.read().decode("utf-8", "ignore").splitlines()[1:]
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("env") != env_name or not d.get("in_zone"):
            continue
        idx = int(d["prompt_idx"])
        if idx not in seen:
            seen.append(idx)
        if len(seen) >= n:
            break
    return seen


def main() -> None:
    from huggingface_hub import snapshot_download

    from reliquary.environment import load_environment
    from reliquary.environment.forced_sampling import seed_consistency, u_at
    from reliquary.miner import engine as eng
    from reliquary.miner.vllm_backend import VLLMBackend
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        load_text_generation_model, load_tokenizer, resolve_eos_token_ids,
    )
    from reliquary import constants as c

    label = os.environ.get("BENCH_LABEL", "base")
    n_prompts = int(os.environ.get("BENCH_PROMPTS", "8"))
    max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "4096"))
    repo = os.environ.get("BENCH_REPO", "ReliquaryForge/qwen3-4b-base-dapo-v4")
    rev = os.environ.get("BENCH_REV")          # défaut : HEAD
    local = snapshot_download(repo, revision=rev)
    ckpt_hash = rev or os.path.basename(local.rstrip("/"))
    randomness = os.environ.get("BENCH_RANDOMNESS", "5a" * 32)

    env = load_environment("opencodeinstruct")
    tok = load_tokenizer(local)
    idxs = _prompt_indices(n_prompts, "opencodeinstruct")
    prompts = [encode_prompt(tok, env.get_problem(i)["prompt"]) for i in idxs]

    backend = VLLMBackend(
        model_path=local, gpu_memory_utilization=float(
            os.environ.get("BENCH_GPU_FRAC", "0.55")),
        max_model_len=int(os.environ.get("BENCH_MAX_MODEL_LEN", "10240")),
        forced_seed=True,
    )
    model_for_eos = load_text_generation_model(
        local, torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("BENCH_HF_ATTN", "sdpa"),
    ).to("cuda").eval()
    eos_ids = sorted(resolve_eos_token_ids(model_for_eos, tok))
    gen_cfg = getattr(model_for_eos, "generation_config", None)
    primary = getattr(gen_cfg, "eos_token_id", None)
    primary = primary if isinstance(primary, int) else None

    t0 = time.time()
    groups = backend.generate_forced_phase1_multi_stream(
        prompts, prompt_indices=idxs, randomness=randomness,
        checkpoint_hash=ckpt_hash, m_rollouts=c.M_ROLLOUTS,
        max_tokens=max_tokens, stop_token_ids=eos_ids, primary_eos_id=primary,
    )
    t_gen = time.time() - t0
    backend._llm = None
    gc.collect()
    torch.cuda.empty_cache()

    eos_set = set(eos_ids)
    dump_rows: list[dict] = []
    n_term = n_term_ok = n_groups_ok = 0
    pos_s = pos_m = 0
    edges_fail: list[float] = []
    for gi, (pidx, ptoks) in enumerate(zip(idxs, prompts)):
        group_ok = True
        for r, gen in enumerate(groups[gi]):
            full = list(ptoks) + list(gen)
            with torch.no_grad():
                logits = model_for_eos(
                    torch.tensor([full], device="cuda")).logits[0]
            plen = len(ptoks)
            step = logits[plen - 1: plen - 1 + len(gen)].float()
            us = [u_at(randomness, pidx, ckpt_hash, r, j) for j in range(len(gen))]
            ns, nm = seed_consistency(
                step, list(gen), us, t=c.T_PROTO, top_k=c.TOP_K_PROTO,
                top_p=c.TOP_P_PROTO,
                stochastic_threshold=c.FORCED_SEED_STOCHASTIC_MAXPROB)
            pos_s += ns
            pos_m += nm
            tj = eng.terminal_pick_inputs(full, plen, eos_set)
            if tj is None:
                continue
            n_term += 1
            ok, edge = eng.terminal_pick_verdict(
                logits[tj[0] - 1], int(full[-1]), us[tj[1]], margin=0.0)
            dump_rows.append({"prompt_idx": pidx, "rollout": r,
                              "prompt_len": plen, "tokens": full,
                              "local_ok": bool(ok), "local_edge": edge})
            if ok:
                n_term_ok += 1
            else:
                group_ok = False
                edges_fail.append(edge)
            del logits
        n_groups_ok += int(group_ok)

    def q(vals, p):
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(p * len(s)))], 6) if s else None

    out = {
        "label": label,
        "cascade_disabled": os.environ.get("RELIQUARY_VLLM_DISABLE_CASCADE", "0"),
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "cuda_graphs": os.environ.get("RELIQUARY_VLLM_CUDA_GRAPHS", ""),
        "fs_graph": os.environ.get("RELIQUARY_FS_GRAPH", ""),
        "hf_attn": os.environ.get("BENCH_HF_ATTN", "sdpa"),
        "prompts": len(idxs), "gen_s": round(t_gen, 1),
        "rollouts_eos": n_term,
        "terminal_exact": round(n_term_ok / n_term, 4) if n_term else None,
        "group_ok": round(n_groups_ok / len(idxs), 4) if idxs else None,
        "pos_match": round(pos_m / pos_s, 4) if pos_s else None,
        "edges_fail": {"n": len(edges_fail), "p10": q(edges_fail, .1),
                       "p50": q(edges_fail, .5), "p90": q(edges_fail, .9)},
    }
    print("[bench] RESULT " + json.dumps(out), flush=True)
    dump = os.environ.get("BENCH_DUMP")
    if dump:
        with open(dump, "w") as fh:
            json.dump({"randomness": randomness, "checkpoint_hash": ckpt_hash,
                       "model_path": local, "rows": dump_rows}, fh)


if __name__ == "__main__":
    main()
