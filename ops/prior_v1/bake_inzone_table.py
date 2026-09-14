"""Nouvelle table de scores V1 : score = P(en zone) du modèle inzone_v1.json.

Garde ``risk``/``volume`` et l'EMPREINTE de l'ancienne table (l'empreinte ne
dépend que des modèles chargés par le mineur, inchangés) : le mineur la charge
sans modification de code.
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inzone_scorer import InZoneScorer  # noqa: E402

_ENV = _SC = None


def _init(model):
    global _ENV, _SC
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    _ENV = OpenCodeInstructEnvironment()
    _SC = InZoneScorer(model)


def _work(rng):
    lo, hi = rng
    out = np.full(hi - lo, np.nan, dtype="float32")
    for i in range(lo, hi):
        try:
            out[i - lo] = _SC.proba((_ENV.get_problem(i) or {}).get("prompt", ""))
        except Exception:
            pass
    return lo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()
    old = np.load(a.old)
    n = len(old["score"]) if not a.limit else a.limit
    score = np.full(len(old["score"]), np.nan, dtype="float32")
    ranges = [(lo, min(lo + a.chunk, n)) for lo in range(0, n, a.chunk)]
    t0 = time.time(); done = 0
    def _consume(it):
        nonlocal done
        for lo, arr in it:
            score[lo:lo + len(arr)] = arr
            done += len(arr)
            dt = time.time() - t0
            print(f"  {done:,}/{n:,} ({100*done/n:.1f} %) {dt/60:.1f} min, reste ~{dt/done*(n-done)/60:.1f} min", flush=True)

    if a.workers <= 1:
        # le Pool se bloquait à l'initialisation sur la box (14/09)
        _init(a.model)
        _consume(_work(r) for r in ranges)
    else:
        with Pool(a.workers, initializer=_init, initargs=(a.model,)) as pool:
            _consume(pool.imap_unordered(_work, ranges))
    miss = int(np.isnan(score[:n]).sum())
    if a.limit:
        print(f"essai {n} prompts : {time.time()-t0:.1f} s, manquants {miss}, moyenne {np.nanmean(score[:n]):.3f}")
        return
    score = np.nan_to_num(score, nan=0.0)
    np.savez(a.out, score=score, risk=old["risk"], volume=old["volume"], fingerprint=old["fingerprint"])
    print(f"écrit {a.out} | manquants {miss} | P(en zone) moyenne {score.mean():.3f} | empreinte {str(old['fingerprint'])[:12]}")


if __name__ == "__main__":
    main()
