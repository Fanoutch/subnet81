"""Banc du coût du forced-seed sur la carte de prod (21/09, H100).

Question : sur CETTE carte, à quelle distance est-on du décodage vLLM sans
forced-seed ? Les chiffres connus (processeur ~15-20 % d'un pas) datent de la
H200 et le vieux ``bench_tokens.py`` ne reproduit plus le chemin actuel
(processeur batché + graphe CUDA du forced-seed).

Méthode : UN moteur, construit exactement comme le mineur (``VLLMBackend``,
``forced_seed=True``, même fraction, même ``max_model_len``, mêmes variables
d'environnement que le processus de prod — lancer avec l'environ du mineur).
Pour chaque taille (160 séquences = notre bake de 10 prompts, 256 = plafond
``MAX_NUM_SEQS``), on génère ``BENCH_LEN`` tokens EXACTEMENT (``ignore_eos``,
pas de token d'arrêt) :
- ``fs``   : paramètres de production (greedy + ``extra_args`` forced-seed) ;
- ``libre``: même moteur, SANS ``extra_args``, échantillonnage T=1 (le chemin
  vLLM ordinaire ; le processeur reste enregistré mais n'a aucune ligne à
  forcer — c'est la borne « sans forced-seed » atteignable sans 2e moteur).
Sortie : ms par pas de décodage et tokens/s, une ligne JSON par mesure.

Budget : ~1 min de chargement + ~40 s de mesures. Conçu pour tourner dans le
trou 503, mineur ARRÊTÉ (carte vide), avant la relance.
"""
from __future__ import annotations

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
    # suffixe distinct par prompt : pas de partage de préfixe complet
    return [tokenizer.encode(base + f"\nCase #{i}: n = {1000 + 37 * i}.",
                             add_special_tokens=False) for i in range(n)]


def _run(llm, prompts, *, m: int, length: int, forced: bool, rnd: str,
         digest: list | None = None) -> float:
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
            if forced:
                kw.update(temperature=0.0, extra_args={
                    FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness=rnd, prompt_idx=10_000 + pi,
                        checkpoint_hash="bench", rollout_index=r,
                        base_offset=0, start_len=len(toks))})
            else:
                kw.update(temperature=1.0)
            sps.append(SamplingParams(**kw))
    t0 = time.perf_counter()
    outs = llm.generate(reqs, sampling_params=sps, use_tqdm=False)
    dt = time.perf_counter() - t0
    got = {len(o.outputs[0].token_ids) for o in outs}
    if got != {length}:
        raise RuntimeError(f"longueurs inattendues {sorted(got)[:5]}")
    if digest is not None:
        import hashlib
        h = hashlib.sha256()
        for o in outs:
            h.update(",".join(map(str, o.outputs[0].token_ids)).encode() + b";")
        digest.append(h.hexdigest()[:16])
    return dt


def main() -> int:
    from transformers import AutoTokenizer
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.miner.vllm_backend import VLLMBackend

    model = os.environ["BENCH_MODEL"]
    tok_path = os.environ.get("BENCH_TOKENIZER", "Qwen/Qwen3-4B-Base")
    length = int(os.environ.get("BENCH_LEN", "768"))
    sizes = [int(x) for x in os.environ.get("BENCH_PROMPTS", "10,16").split(",")]
    reps = int(os.environ.get("BENCH_REPEATS", "1"))
    out = os.environ.get("BENCH_OUT", "/workspace/bench_fs_h100.jsonl")
    modes = [x for x in os.environ.get("BENCH_MODES", "fs,libre").split(",") if x]
    variant = "fs_fast" if os.environ.get("RELIQUARY_FS_FAST") == "1" else "fs"
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
    _run(llm, warm, m=m, length=64, forced=True, rnd=rnd)     # chauffe
    _run(llm, warm, m=m, length=64, forced=False, rnd=rnd)
    rows = []
    for n in sizes:
        prompts = _prompts(tokenizer, n)
        for rep in range(reps):
            for forced in [md == "fs" for md in modes]:
                dig: list = []
                dt = _run(llm, prompts, m=m, length=length, forced=forced,
                          rnd=rnd, digest=dig)
                row = {"seqs": n * m, "len": length,
                       "mode": variant if forced else "libre", "rep": rep,
                       "tokens_sha": dig[0] if forced else None,
                       "s": round(dt, 3), "ms_par_pas": round(1000 * dt / length, 3),
                       "tok_s": round(n * m * length / dt)}
                rows.append(row)
                print("[bench]", json.dumps(row), flush=True)
    with open(out, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "ts": time.time(), "model": model}) + "\n")
    for n in sizes:
        fs = [r["ms_par_pas"] for r in rows if r["seqs"] == n * m and r["mode"] == variant]
        lb = [r["ms_par_pas"] for r in rows if r["seqs"] == n * m and r["mode"] == "libre"]
        if fs and lb:
            f, l = min(fs), min(lb)
            print(f"[bench] {n * m} séq. : fs {f:.2f} ms/pas, libre {l:.2f} ms/pas "
                  f"-> surcoût forced-seed {100 * (f / l - 1):+.1f} %", flush=True)
    print(f"[bench] total {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
