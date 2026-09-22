"""Banc de décomposition du coût du forced-seed (21-22/09, H100).

Question : sur CETTE carte, où partent les ~3,8 ms par pas du forced-seed
(12,15 contre 8,36 ms à 160 séquences, 21/09) ?

UN moteur par « famille », construit exactement comme le mineur
(``VLLMBackend``, même fraction, même ``max_model_len``, lancer avec l'environ
du mineur). Modes (``MODES``) :
- famille ``nu`` (``forced_seed=False``, aucun processeur) :
  ``nu_greedy`` (T=0), ``nu_t1`` (T=1) ;
- famille ``fs`` (``forced_seed=True``, processeur enregistré) :
  ``libre_greedy`` / ``libre_t1`` (aucun extra_args : processeur vide),
  ``noop`` (extra_args + ``noop`` : processeur rempli mais inactif),
  ``fs`` (chemin de prod), ``fast`` (extra_args + ``fast`` : chemin rapide).
Écarts à lire : enregistrement = libre_greedy − nu_greedy ; mécanisme =
noop − libre_greedy ; notre calcul = fs − noop ; gain rapide = fast − fs.
Chaque séquence génère ``len`` tokens EXACTEMENT (``ignore_eos``).
Conformité : taux de tokens identiques (``identity_rates``) contre le 1er
passage ``fs`` de la même (taille, longueur) — le bruit de fond est ``fs``
contre ``fs``. Une ligne JSON par passage dans ``BENCH_OUT``.
"""
from __future__ import annotations

import json
import os
import sys
import time

MODES = {
    "nu_greedy": ("nu", 0.0, None),
    "nu_t1": ("nu", 1.0, None),
    "libre_greedy": ("fs", 0.0, None),
    "libre_t1": ("fs", 1.0, None),
    "noop": ("fs", 0.0, "noop"),
    "fs": ("fs", 0.0, ""),
    "fast": ("fs", 0.0, "fast"),
}


def identity_rates(ref, other) -> tuple[float, float]:
    """(part des séquences identiques, part moyenne des positions identiques
    avant la 1re divergence)."""
    same_seq, pos_frac = 0, 0.0
    for a, b in zip(ref, other):
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                 min(len(a), len(b)))
        same_seq += int(list(a) == list(b))
        pos_frac += k / max(1, len(a))
    n = max(1, len(ref))
    return same_seq / n, pos_frac / n


def _prompts(tokenizer, n: int) -> list[list[int]]:
    base = ("Solve the following programming problem step by step.\n\n"
            "Write a Python function that, given a list of integers, returns "
            "the length of the longest strictly increasing subsequence. "
            "Explain your reasoning, then provide the final implementation in "
            "the last fenced Python code block.\n")
    return [tokenizer.encode(base + f"\nCase #{i}: n = {1000 + 37 * i}.",
                             add_special_tokens=False) for i in range(n)]


def _run(llm, prompts, *, m: int, length: int, mode: str, rnd: str):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from reliquary.miner.vllm_forced_seed import (
        FORCED_SEED_EXTRA_KEY, forced_seed_extra_args)
    _fam, temp, flag = MODES[mode]
    reqs, sps = [], []
    for pi, toks in enumerate(prompts):
        for r in range(m):
            reqs.append(TokensPrompt(prompt_token_ids=toks))
            kw = dict(n=1, max_tokens=length, min_tokens=length,
                      ignore_eos=True, detokenize=False, temperature=temp)
            if flag is not None:
                fs = forced_seed_extra_args(
                    randomness=rnd, prompt_idx=10_000 + pi,
                    checkpoint_hash="bench", rollout_index=r,
                    base_offset=0, start_len=len(toks))
                if flag:
                    fs = {**fs, flag: True}
                kw["extra_args"] = {FORCED_SEED_EXTRA_KEY: fs}
            sps.append(SamplingParams(**kw))
    t0 = time.perf_counter()
    outs = llm.generate(reqs, sampling_params=sps, use_tqdm=False)
    dt = time.perf_counter() - t0
    toks = [list(o.outputs[0].token_ids) for o in outs]
    if {len(t) for t in toks} != {length}:
        raise RuntimeError(f"longueurs inattendues {sorted({len(t) for t in toks})[:5]}")
    return dt, toks


def _ints(name: str, default: str) -> list[int]:
    return [int(x) for x in os.environ.get(name, default).split(",") if x]


def main() -> int:
    from transformers import AutoTokenizer
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.miner.vllm_backend import VLLMBackend

    model = os.environ["BENCH_MODEL"]
    tok_path = os.environ.get("BENCH_TOKENIZER", "Qwen/Qwen3-4B-Base")
    family = os.environ.get("BENCH_FAMILY", "fs")
    default_modes = "nu_greedy,nu_t1" if family == "nu" else "libre_t1,fs,fast"
    modes = [md for md in os.environ.get("BENCH_MODES", default_modes).split(",") if md]
    bad = [md for md in modes if MODES[md][0] != family]
    if bad:
        raise SystemExit(f"modes {bad} hors de la famille {family}")
    sizes = _ints("BENCH_PROMPTS", "10")
    lens = _ints("BENCH_LENS", os.environ.get("BENCH_LEN", "512"))
    reps = int(os.environ.get("BENCH_REPS", "3"))
    soft = float(os.environ.get("BENCH_SOFT_S", "1e9"))
    out = os.environ.get("BENCH_OUT", "/workspace/bench_fs_h100.jsonl")
    m = 16

    t0 = time.time()
    be = VLLMBackend(
        model_path=model, tokenizer_path=tok_path, gpu_id=0,
        gpu_memory_utilization=float(
            os.environ.get("RELIQUARY_VLLM_GPU_FRACTION", "0.45")),
        max_model_len=vllm_max_model_len(), dtype="bfloat16",
        forced_seed=(family == "fs"))
    be._ensure_loaded()
    llm = be._llm
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    print(f"[bench] famille {family} : moteur prêt en {time.time() - t0:.1f}s", flush=True)

    rnd = "5a" * 32
    warm = _prompts(tokenizer, 2)
    for md in modes:
        _run(llm, warm, m=m, length=64, mode=md, rnd=rnd)

    summary = []
    with open(out, "a", encoding="utf-8") as fh:
        for length in lens:
            for n in sizes:
                prompts = _prompts(tokenizer, n)
                ref_tokens = None
                times: dict[str, list[float]] = {md: [] for md in modes}
                for rep in range(reps):
                    t_rep = time.time()
                    order = modes if rep % 2 == 0 else list(reversed(modes))
                    for md in order:
                        dt, toks = _run(llm, prompts, m=m, length=length, mode=md, rnd=rnd)
                        seq_id = pos_id = None
                        if md in ("fs", "fast", "noop") and MODES[md][2] != "noop":
                            if ref_tokens is None and md == "fs":
                                ref_tokens = toks
                            elif ref_tokens is not None:
                                seq_id, pos_id = identity_rates(ref_tokens, toks)
                        ms = 1000 * dt / length
                        times[md].append(ms)
                        row = {"family": family, "seqs": n * m, "len": length,
                               "mode": md, "rep": rep, "s": round(dt, 3),
                               "ms_par_pas": round(ms, 3),
                               "tok_s": round(n * m * length / dt),
                               "seq_identiques": seq_id, "pos_identiques": pos_id,
                               "ts": time.time(), "model": model}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print("[bench]", json.dumps({k: row[k] for k in (
                            "seqs", "len", "mode", "rep", "ms_par_pas", "tok_s",
                            "seq_identiques", "pos_identiques")}), flush=True)
                    if (time.time() - t0) + (time.time() - t_rep) > soft:
                        break
                best = {md: min(v) for md, v in times.items() if v}
                summary.append((n * m, length, best))
    for seqs, length, best in summary:
        print(f"[bench] {seqs} séq. × {length} (meilleur) : " +
              " · ".join(f"{md} {v:.2f}" for md, v in best.items()) + " ms/pas", flush=True)
    print(f"[bench] total {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
