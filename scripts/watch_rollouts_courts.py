#!/usr/bin/env python3
"""Vigie du retrait du gate anti-rollout-court — signal d'ABANDON en 4-8 min.

CE QU'ELLE SURVEILLE, et pourquoi chaque signal compte :

  1. `logprob_mismatch` — LE signal d'abandon. Avant PR #188 (mergée le 21/08
     à 13h22, live sur le validateur), la vérification échouait par
     construction sous 32 tokens : 67 rejets sur 67 en venaient. Si ce compteur
     bouge après le retrait du gate, le correctif ne marche pas pour nous et il
     faut remettre le gate SANS ATTENDRE.

  2. DETTE DE PREUVE — le vrai danger. `_PROOF_FAILURE_DEBT_STAGES` inclut le
     stage "logprob" : **deux** rejets de ce type dans une fenêtre et tout le
     reste de la fenêtre est refusé, y compris nos bonnes entrées déjà en vol.
     C'est ça qu'on risque, pas la perte de l'entrée courte elle-même.
     ⚠️ `hash_duplicate` (stage "dedup") n'est PAS dans la liste : le prompt
     pris avant nous ne coûte rien. Ne pas le compter comme un risque.

  3. GROUPES COURTS ADMIS — la preuve que ça marche. On attend ~1,5 entrée
     courte par fenêtre. La première acceptée valide le correctif.

Exposition mesurée : 8,2 envois par fenêtre dont ~18 % de courts. Si le
correctif était totalement inopérant, on atteindrait 2 échecs dans 23 % des
fenêtres — d'où l'urgence de la lecture, pas d'une rampe lente.

Usage :  python3 scripts/watch_rollouts_courts.py [--depuis EPOCH]
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time

BOX, PORT = "root@38.255.28.21", "20098"

DISTANT = r'''
import json, collections, sys, time
DEPUIS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

# longueurs par (fenetre, prompt) pour savoir quelles entrees sont COURTES
G = {}
for l in open("/workspace/samples_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    cl = r.get("completion_lens")
    if cl and r.get("window_n") is not None:
        G[(r["window_n"], r.get("prompt_idx"))] = min(cl)

# verdicts DEDUPLIQUES par merkle_root : le dump reecrit la meme ligne a
# chaque interrogation de /verdicts, compter les lignes gonfle le passe.
V = {}
for l in open("/workspace/verdicts_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    mr = r.get("merkle_root")
    if mr: V[mr] = r

DEBT = {"code_grader_crash","grail","termination","force_span","logprob",
        "distribution","boxed_answer","token_authenticity",
        "all_token_authenticity","code_semantic_auth","forced_seed"}

par_fen = collections.defaultdict(lambda: {"court_env":0,"court_ok":0,
                                           "court_lpm":0,"dette":0,"ok":0})
lpm_total = 0
for l in open("/workspace/submits_v4.jsonl", errors="replace"):
    try: r = json.loads(l)
    except Exception: continue
    ts = r.get("ts")
    if not ts or ts < DEPUIS: continue
    w = r.get("window_n")
    v = V.get(r.get("merkle_root"))
    if not w or not v: continue
    d = par_fen[w]
    court = G.get((w, r.get("prompt_idx")), 999) < 32
    raison = v.get("reject_reason") or v.get("reason") or ""
    stage = v.get("reject_stage") or ""
    if court:
        d["court_env"] += 1
        if raison == "accepted": d["court_ok"] += 1
        if raison == "logprob_mismatch": d["court_lpm"] += 1; lpm_total += 1
    elif raison == "logprob_mismatch":
        lpm_total += 1
    if raison == "accepted": d["ok"] += 1
    elif stage in DEBT: d["dette"] += 1

fens = sorted(par_fen)
print(json.dumps({
  "fenetres": len(fens),
  "courts_envoyes": sum(par_fen[w]["court_env"] for w in fens),
  "courts_acceptes": sum(par_fen[w]["court_ok"] for w in fens),
  "courts_logprob": sum(par_fen[w]["court_lpm"] for w in fens),
  "logprob_total": lpm_total,
  "fen_dette_2plus": sum(1 for w in fens if par_fen[w]["dette"] >= 2),
  "acceptees": sum(par_fen[w]["ok"] for w in fens),
  "derniere": fens[-1] if fens else None,
}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depuis", type=float, default=0.0,
                    help="epoch : ne compter que les envois postérieurs")
    ap.add_argument("--verbeux", action="store_true",
                    help="parler même quand le gate est encore actif")
    a = ap.parse_args()
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no",
         "-p", PORT, BOX, f"/workspace/venv/bin/python - {a.depuis}"],
        input=DISTANT, capture_output=True, text=True, timeout=120).stdout
    try:
        d = json.loads(out.strip().splitlines()[-1])
    except Exception:
        print("vigie courts: box injoignable ou dumps illisibles")
        return 1

    ce, ca, cl = d["courts_envoyes"], d["courts_acceptes"], d["courts_logprob"]
    if ce == 0:
        # ÉTAT D'ATTENTE : le gate est encore actif, rien à dire. On reste
        # SILENCIEUX plutôt que de répéter la même ligne toutes les deux
        # minutes — les compteurs de fenêtres avancent sans cesse et
        # feraient croire à un événement neuf à chaque passage.
        if a.verbeux:
            print(f"vigie courts: aucune entrée courte envoyée "
                  f"({d['fenetres']} fen) — gate encore actif")
        return 0
    taux = 100 * ca / ce
    msg = (f"courts {ca}/{ce} admis ({taux:.0f} %) | logprob_mismatch {cl} | "
           f"fen à dette≥2 : {d['fen_dette_2plus']}/{d['fenetres']}")
    if cl > 0:
        print(f"🛑 ABANDON — {msg}")
        print("   Le correctif #188 ne passe pas pour nos groupes. "
              "Remettre RELIQUARY_MIN_ROLLOUT_LEN=32 et redémarrer.")
    elif d["fen_dette_2plus"] > 0:
        print(f"⚠️  DETTE — {msg}")
        print("   Des fenêtres atteignent 2 rejets à dette. Vérifier la cause "
              "(worker_dropped du validateur ? ou nos courts ?).")
    else:
        print(f"✅ {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
