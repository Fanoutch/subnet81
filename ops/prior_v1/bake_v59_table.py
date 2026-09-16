"""Table de scores V1 « v5.9 seul » (16/09) : score = prompt_predictor.score_prompt(v5.9).

Éval 16/09 (9 257 groupes post-46012) : AUC en zone 0,608 contre 0,441 pour inzone_v1.

Garde ``risk``/``volume`` et l'EMPREINTE de l'ancienne table (l'empreinte ne
dépend que des modèles chargés par le mineur, inchangés) : le mineur la charge
sans modification de code.
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ENV = _SC = None


def _init(model):
    global _ENV, _SC
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    _ENV = OpenCodeInstructEnvironment()
    import json
    from reliquary.miner import prompt_predictor as pp
    m = json.load(open(model))
    class _S:
        def proba(self, t):
            return pp.score_prompt(m, t)
    _SC = _S()


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
    ap.add_argument("--fingerprint", default="",
                    help="empreinte attendue (sinon calculée depuis l'env du mineur)")
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
    # Empreinte ATTENDUE par le mineur (14/09 : recopier celle de l'ancienne
    # table la rendait « PÉRIMÉE » — l'ancienne n'était déjà plus chargée).
    fp = str(old["fingerprint"])
    if a.fingerprint:
        fp = a.fingerprint
    else:
        try:
            from reliquary.miner import engine as eng, prompt_scores as ps
            fp = ps.fingerprint(predictor=eng._load_predictor(), risk=eng._RISK_MODEL,
                                volume=eng._VOLUME_MODEL,
                                revision=os.environ.get("RELIQUARY_DATASET_REVISION", ""))
        except Exception as exc:
            print(f"ATTENTION empreinte non calculée ({exc!r}) : celle de --old reprise")
    np.savez(a.out, score=score, risk=old["risk"], volume=old["volume"], fingerprint=np.array(fp))
    print(f"écrit {a.out} | manquants {miss} | score moyen {score.mean():.3f} | empreinte {fp[:12]}")


if __name__ == "__main__":
    main()
