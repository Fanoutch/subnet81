"""Banc du coût du forced-seed sur la carte de prod (21/09, H100).

Question : sur CETTE carte, à quelle distance est-on du décodage vLLM sans
forced-seed, et que gagne le chemin rapide (``RELIQUARY_FS_FAST``) ?

Méthode : UN moteur, construit exactement comme le mineur (``VLLMBackend``,
``forced_seed=True``, même fraction, même ``max_model_len``, mêmes variables
d'environnement que le processus de prod — lancer avec l'environ du mineur).
UN SEUL chargement (~28 s) pour toutes les variantes, afin que tout tienne
dans UN trou 503 (~280 s, dont ~145 s pour relancer le mineur) :
- ``libre`` : même moteur, SANS ``extra_args``, échantillonnage T=1 (borne
  « sans forced-seed » ; le processeur est enregistré mais n'a rien à forcer) ;
- ``fs``    : paramètres de production (greedy + ``extra_args`` forced-seed) ;
- ``fast``  : idem + ``"fast": True`` dans les ``extra_args`` -> le processeur
  prend le chemin rapide pour ce lot (sélection par requête, cf.
  ``ForcedRowsState.rebuild``).
Chaque séquence génère ``BENCH_LEN`` tokens EXACTEMENT (``ignore_eos``).
Contrôle de conformité : empreinte des tokens ``fs`` == ``fast`` à chaque
taille, sinon le chemin rapide est REJETÉ (code retour 3).
Répétitions (ordre alterné) tant que ``BENCH_SOFT_S`` n'est pas atteint.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time


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
    reqs, sps = [], []
    for pi, toks in enumerate(prompts):
        for r in range(m):
            reqs.append(TokensPrompt(prompt_token_ids=toks))
            kw = dict(n=1, max_tokens=length, min_tokens=length,
                      ignore_eos=True, detokenize=False)
            if mode in ("fs", "fast"):
                fs = forced_seed_extra_args(
                    randomness=rnd, prompt_idx=10_000 + pi,
                    checkpoint_hash="bench", rollout_index=r,
                    base_offset=0, start_len=len(toks))
                if mode == "fast":
                    fs = {**fs, "fast": True}
                kw.update(temperature=0.0, extra_args={FORCED_SEED_EXTRA_KEY: fs})
            else:
                kw.update(temperature=1.0)
            sps.append(SamplingParams(**kw))
    t0 = time.perf_counter()
    outs = llm.generate(reqs, sampling_params=sps, use_tqdm=False)
    dt = time.perf_counter() - t0
    got = {len(o.outputs[0].token_ids) for o in outs}
    if got != {length}:
        raise RuntimeError(f"longueurs inattendues {sorted(got)[:5]}")
    h = hashlib.sha256()
    for o in outs:
        h.update(",".join(map(str, o.outputs[0].token_ids)).encode() + b";")
    return dt, h.hexdigest()[:16]


def main() -> int:
    from transformers import AutoTokenizer
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.miner.vllm_backend import VLLMBackend

    model = os.environ["BENCH_MODEL"]
    tok_path = os.environ.get("BENCH_TOKENIZER", "Qwen/Qwen3-4B-Base")
    length = int(os.environ.get("BENCH_LEN", "512"))
    sizes = [int(x) for x in os.environ.get("BENCH_PROMPTS", "10").split(",")]
    soft = float(os.environ.get("BENCH_SOFT_S", "90"))
    out = os.environ.get("BENCH_OUT", "/workspace/bench_fs_h100.jsonl")
    m = 16

    t0 = time.time()
    be = VLLMBackend(
        model_path=model, tokenizer_path=tok_path, gpu_id=0,
        gpu_memory_utilization=float(
            os.environ.get("RELIQUARY_VLLM_GPU_FRACTION", "0.45")),
        max_model_len=vllm_max_model_len(), dtype="bfloat16", forced_seed=True)
    be._ensure_loaded()
    llm = be._llm
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    print(f"[bench] moteur prêt en {time.time() - t0:.1f}s", flush=True)

    rnd = "5a" * 32
    warm = _prompts(tokenizer, 2)
    for md in ("fs", "fast", "libre"):                          # chauffe
        _run(llm, warm, m=m, length=64, mode=md, rnd=rnd)

    rows, digests = [], {}
    orders = [("libre", "fs", "fast"), ("fast", "fs", "libre")]
    rep = 0
    while True:
        t_rep = time.time()
        for n in sizes:
            prompts = _prompts(tokenizer, n)
            for md in orders[rep % 2]:
                dt, dig = _run(llm, prompts, m=m, length=length, mode=md, rnd=rnd)
                if md != "libre":
                    digests.setdefault((n * m, md), set()).add(dig)
                row = {"seqs": n * m, "len": length, "mode": md, "rep": rep,
                       "s": round(dt, 3), "ms_par_pas": round(1000 * dt / length, 3),
                       "tok_s": round(n * m * length / dt),
                       "tokens_sha": dig if md != "libre" else None}
                rows.append(row)
                print("[bench]", json.dumps(row), flush=True)
        rep += 1
        # ne lancer une répétition que si elle finit avant le budget souple
        if rep >= 4 or (time.time() - t0) + (time.time() - t_rep) > soft:
            break

    with open(out, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "ts": time.time(), "model": model}) + "\n")
    ok_all = True
    for n in sizes:
        s = n * m
        best = {md: min(r["ms_par_pas"] for r in rows
                        if r["seqs"] == s and r["mode"] == md)
                for md in ("libre", "fs", "fast")}
        same = (digests.get((s, "fs")) == digests.get((s, "fast"))
                and len(digests.get((s, "fs"), ())) == 1)
        ok_all &= same
        print(f"[bench] {s} séq. ({rep} rép., meilleur) : libre {best['libre']:.2f} · "
              f"fs {best['fs']:.2f} · fast {best['fast']:.2f} ms/pas -> surcoût fs "
              f"{100 * (best['fs'] / best['libre'] - 1):+.1f} %, fast "
              f"{100 * (best['fast'] / best['libre'] - 1):+.1f} % | tokens fs == fast : "
              f"{'OUI' if same else 'NON'}", flush=True)
    print(f"[bench] VERDICT conformité : "
          f"{'IDENTIQUES' if ok_all else 'DIVERGENTS -> REJETER'}", flush=True)
    print(f"[bench] total {time.time() - t0:.1f}s", flush=True)
    return 0 if ok_all else 3


if __name__ == "__main__":
    sys.exit(main())
