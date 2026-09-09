#!/usr/bin/env python3
"""Extrait un corpus d'entraînement du prior depuis les archives R2.

Pourquoi : le forced-seed v2 rend les tokens identiques pour tous les mineurs
à (randomness, prompt_idx, checkpoint) donnés — chaque groupe payé archivé est
donc EXACTEMENT ce que notre mineur aurait généré. R2 fournit 32 groupes
complets par fenêtre (16 code + 16 math, tout le marché) contre 4-8 pour notre
dump local, et continue de couler même quand notre mineur est éteint : le
corpus ne gèle plus jamais (leçon du 22-24/08).

Deux classes émises, au format samples_v4.jsonl :
  - POSITIFS  : batch[] -> prompt, rewards (16), sigma, k, score, lens ;
  - NÉGATIFS  : rejected[reason=out_of_zone] -> le rejet EST le label
                (sigma=0 => score=0 exactement) ; rewards inconnues (None),
                texte du prompt récupérable par prompt_idx via le miroir
                parquet. On ne fabrique JAMAIS de rewards.

Usage :
  python3 scripts/r2_pull_windows.py --debut A --fin B --out <cache>
  python3 scripts/r2_to_samples.py --cache <cache> [--depuis 36400] \
      --out data/samples_r2_market.jsonl
"""
import argparse
import glob
import gzip
import json
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+",
                    help="répertoires d'archives window-*.json.gz")
    ap.add_argument("--depuis", type=int, default=0,
                    help="première fenêtre incluse (borne d'ère)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(f for c in a.cache for f in glob.glob(c + "/*.json.gz"))
    seen = set()          # (window, prompt_idx) — les archives sampled peuvent
    npos = nneg = 0       # se recouper entre caches
    with open(a.out, "w") as out:
        for f in files:
            d = json.loads(gzip.decompress(open(f, "rb").read()))
            w = d["window_start"]
            if w < a.depuis:
                continue
            ck = None
            for e in d["batch"]:
                if (w, e["prompt_idx"]) in seen:
                    continue
                seen.add((w, e["prompt_idx"]))
                rewards = [r["reward"] for r in e["rollouts"]]
                mean = st.mean(rewards)
                sigma = st.pstdev(rewards)
                ck = e.get("claimed_checkpoint_hash")
                out.write(json.dumps({
                    "prompt": e.get("prompt", ""),
                    "prompt_idx": e["prompt_idx"],
                    "rewards": rewards,
                    "sigma": sigma,
                    "score": sigma * (1.0 - mean),
                    "k": sum(1 for x in rewards if x >= 0.999),
                    "completion_lens": sorted(
                        r["completion_length"] for r in e["rollouts"]),
                    "window_n": w,
                    "env": e["env_name"],
                    "checkpoint_hash": ck,
                    "source": "r2_market",
                    "in_zone": True,
                }, separators=(",", ":")) + "\n")
                npos += 1
            for e in d["rejected"]:
                if e.get("reason") != "out_of_zone":
                    continue
                if (w, e["prompt_idx"]) in seen:
                    continue
                seen.add((w, e["prompt_idx"]))
                out.write(json.dumps({
                    "prompt": "",              # texte via miroir parquet (prompt_idx)
                    "prompt_idx": e["prompt_idx"],
                    "rewards": None,           # inconnues — jamais fabriquées
                    "sigma": 0.0,              # définition même du rejet
                    "score": 0.0,              # sigma*(1-mean) = 0 quel que soit k
                    "k": None,                 # 0 ou 16, indistinguable dans R2
                    "completion_lens": None,
                    "window_n": w,
                    "env": e["env_name"],
                    "checkpoint_hash": ck,
                    "source": "r2_market_oz",
                    "in_zone": False,
                }, separators=(",", ":")) + "\n")
                nneg += 1
    print(f"{npos} positifs + {nneg} négatifs (score=0) -> {a.out}")


if __name__ == "__main__":
    main()
