"""Profilage vLLM (torch profiler) d'une génération par mode (22/09).

Pourquoi : isolé, le processeur forced-seed coûte 0,8-1,9 ms par pas ; dans
vLLM il coûte ~4,2 ms. Ce script mesure, DANS le moteur, pour chaque mode
(noop / fs / fast) : temps GPU des noyaux par pas, temps mort GPU par pas,
et les opérations CPU/GPU les plus lourdes — pour voir où partent les ~2,5 ms
manquantes.

``profiler_config`` est injecté au constructeur ``vllm.LLM`` par une
enveloppe (le code de prod n'est pas modifié). Traces dans PROFILE_DIR/<mode>.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_fs_h100 import _prompts, _run  # noqa: E402


def _analyse(trace_path: str, steps: int) -> dict:
    op = gzip.open if trace_path.endswith(".gz") else open
    with op(trace_path, "rt") as fh:
        ev = json.load(fh).get("traceEvents", [])
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    if not kern:
        return {"erreur": "aucun noyau GPU dans la trace"}
    t0 = min(e["ts"] for e in kern); t1 = max(e["ts"] + e["dur"] for e in kern)
    busy = sum(e["dur"] for e in kern)
    by_k = defaultdict(float)
    for e in kern:
        by_k[e["name"][:70]] += e["dur"]
    by_c = defaultdict(float)
    for e in cpu:
        by_c[e["name"][:70]] += e["dur"]
    return {
        "fenetre_ms": (t1 - t0) / 1000, "gpu_occupe_ms": busy / 1000,
        "par_pas_mur_ms": (t1 - t0) / 1000 / steps,
        "par_pas_gpu_ms": busy / 1000 / steps,
        "par_pas_trou_gpu_ms": ((t1 - t0) - busy) / 1000 / steps,
        "top_noyaux": sorted(((k, v / 1000 / steps) for k, v in by_k.items()), key=lambda x: -x[1])[:12],
        "top_cpu": sorted(((k, v / 1000 / steps) for k, v in by_c.items()), key=lambda x: -x[1])[:15],
    }


def main() -> int:
    import vllm
    from transformers import AutoTokenizer
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.miner.vllm_backend import VLLMBackend

    pdir = os.environ.get("PROFILE_DIR", "/workspace/prof_fs")
    os.makedirs(pdir, exist_ok=True)
    orig = vllm.LLM.__init__

    def _init(self, *a, **kw):
        kw.setdefault("profiler_config", {"profiler": "torch", "torch_profiler_dir": pdir,
                                          "torch_profiler_with_stack": False})
        return orig(self, *a, **kw)
    vllm.LLM.__init__ = _init

    model = os.environ["BENCH_MODEL"]
    steps = int(os.environ.get("PROFILE_STEPS", "64"))
    n = int(os.environ.get("PROFILE_PROMPTS", "10"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    be = VLLMBackend(model_path=model, tokenizer_path="Qwen/Qwen3-4B-Base", gpu_id=0,
                     gpu_memory_utilization=float(os.environ.get("RELIQUARY_VLLM_GPU_FRACTION", "0.45")),
                     max_model_len=vllm_max_model_len(), dtype="bfloat16", forced_seed=True)
    be._ensure_loaded()
    llm, prompts, rnd = be._llm, _prompts(tok, n), "5a" * 32
    for mode in os.environ.get("PROFILE_MODES", "noop,fs,fast").split(","):
        _run(llm, prompts, m=16, length=steps, mode=mode, rnd=rnd)          # chauffe
        before = set(glob.glob(os.path.join(pdir, "**", "*.json*"), recursive=True))
        llm.start_profile()
        dt, _ = _run(llm, prompts, m=16, length=steps, mode=mode, rnd=rnd)
        llm.stop_profile()
        time.sleep(3)
        new = sorted(set(glob.glob(os.path.join(pdir, "**", "*.json*"), recursive=True)) - before,
                     key=os.path.getsize)
        print(f"[profil] {mode}: génération {1000 * dt / steps:.2f} ms/pas (mur, avec profileur) "
              f"traces={len(new)}", flush=True)
        if new:
            res = _analyse(new[-1], steps)              # la plus grosse = worker GPU
            print(f"[profil] {mode}: " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
