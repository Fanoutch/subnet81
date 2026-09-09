#!/usr/bin/env python3
"""Pré-calcule les scores de TOUS les prompts, hors ligne (24/08).

POURQUOI. Le classement de tranche coûte **2,80 s p50 en tête de chaque
fenêtre** (4,6 p90, 14 s max), et il tourne sur le thread de la boucle asyncio
— pendant ce temps aucun ``GET /state`` ni aucun POST ne part. Rebanché sur la
box : lecture parquet 0,95 s, ``get_problem`` 0,30 s, notation 0,91 s.

Or les trois notations sont des fonctions PURES du texte. Une fois calculées,
le classement se réduit à trancher des flottants et trier (~2 ms).

CE QUE ÇA VAUT. Mesuré sur 174 fenêtres mûres, effets fixes par fenêtre :
1 seconde d'arrivée = **+1,46 place** = 390 tokens. Le contrefactuel donne
−3 s → payées par fenêtre **0,44 → 0,85**. Et le gain n'est pas linéaire :
depuis 16,1 s, **−1,2 s suffit** à gagner un round drand (39 % du bénéfice).

USAGE
    python3 scripts/precompute_prompt_scores.py --out /workspace/prompt_scores.npz

    # puis dans le launcher :
    export RELIQUARY_PROMPT_SCORES=/workspace/prompt_scores.npz

SÛRETÉ. Le fichier porte une empreinte des trois modèles + de la révision du
dataset. Si l'un change (l'entraînement du prior tourne chaque nuit), le
mineur DÉTECTE la péremption et retombe sur la notation en direct — il ne
sert jamais un classement obsolète.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="fichier .npz de sortie")
    ap.add_argument("--limit", type=int, default=0,
                    help="ne traiter que les N premiers prompts (essai)")
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    import numpy as np

    from reliquary.environment.opencodeinstruct import (
        OpenCodeInstructEnvironment,
    )
    from reliquary.miner import prompt_predictor as pp
    from reliquary.miner import prompt_scores as ps
    from reliquary.miner import engine as eng

    predictor = eng._load_predictor()
    risk = eng._RISK_MODEL
    volume = eng._VOLUME_MODEL
    if predictor is None:
        print("ERREUR: aucun prior chargé (RELIQUARY_PROMPT_PREDICTOR)",
              file=sys.stderr)
        return 2
    print(f"prior       : {'oui' if predictor else 'non'}")
    print(f"malus court : {'oui' if risk else 'non'}")
    print(f"volume      : {'oui' if volume else 'non'}")

    env = OpenCodeInstructEnvironment()
    n = len(env) if not args.limit else min(args.limit, len(env))
    print(f"prompts     : {n:,}")

    score = np.zeros(n, dtype="float32")
    risk_a = np.zeros(n, dtype="float32")
    vol_a = np.zeros(n, dtype="float32")

    t0 = time.time()
    erreurs = 0
    for idx in range(n):
        try:
            text = (env.get_problem(idx) or {}).get("prompt", "")
            score[idx] = pp.score_prompt(predictor, text)
            if risk is not None:
                risk_a[idx] = pp.risk_short(risk, text)
            if volume is not None:
                vol_a[idx] = pp.volume_score(volume, text)
        except Exception:
            # Un prompt illisible garde un score de 0 : il tombera en fin de
            # classement, exactement comme le `continue` du chemin en direct.
            erreurs += 1
        if args.progress_every and idx and idx % args.progress_every == 0:
            dt = time.time() - t0
            reste = dt / idx * (n - idx)
            print(f"  {idx:,}/{n:,}  ({100*idx/n:.1f} %)  "
                  f"{dt/60:.1f} min écoulées, ~{reste/60:.1f} min restantes",
                  flush=True)

    fp = ps.fingerprint(
        predictor=predictor, risk=risk, volume=volume,
        revision=os.environ.get("RELIQUARY_DATASET_REVISION", ""),
    )
    ps.save(args.out, score=score, risk=risk_a, volume=vol_a, fingerprint=fp)

    taille = Path(args.out).stat().st_size / 1e6
    print(f"\nécrit    : {args.out}  ({taille:.1f} Mo)")
    print(f"empreinte: {fp}")
    print(f"durée    : {(time.time()-t0)/60:.1f} min | erreurs: {erreurs}")
    print(f"\nÀ ajouter au launcher :\n"
          f"  export RELIQUARY_PROMPT_SCORES={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
