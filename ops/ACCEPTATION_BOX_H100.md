# Acceptation d'une nouvelle box H100 — référence du 22/09/2026

Box de référence : Lambda `192.222.54.118:20300`, rendue le 22/09 pendant la
panne du validateur (depuis 01:57 UTC). Une nouvelle box n'est gardée que si
elle retrouve ces chiffres à **±5 %**. Sinon : la rendre et en louer une autre
(une carte lente est invisible dans les journaux du mineur).

## 1. Matériel (`nvidia-smi`, `lscpu`)

| grandeur | référence |
|---|---|
| carte | NVIDIA H100 80GB **HBM3** (SXM) — 81 559 MiB |
| driver | 580.126.20 |
| horloge SM max / mémoire max | 1 980 MHz / 2 619 MHz |
| limite de puissance | 700 W (une PCIe à 350 W = autre carte, refuser) |
| PCIe | gen 5 |
| CPU | Intel Xeon Platinum 8480+, 26 cœurs visibles |
| RAM | 221 Go |

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.max --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2
```

## 2. Pile logicielle (identique à la référence)

torch 2.11.0+cu130 · vLLM 0.24.0 · transformers 5.9.0 · Python 3.12.
Relevés exacts : `data_backups/box_2026-09-22_panne/reconstruction/pip_freeze_venv.txt`
(mineur) et `pip_freeze_venv_val.txt` (réplique). Reconstruction :
`ops/RECONSTRUCTION_BOX.md` (⚠️ miroir parquet obligatoire), état à recopier :
`data_backups/box_2026-09-22_panne/etat/` (MD5SUMS fourni).

## 3. Banc de décodage (carte vide, AVANT de lancer le mineur)

Environ du mineur (variables `RELIQUARY_*` de `etat/env_live_20260922.txt` +
launcher). Commande (famille `fs`, 160 séquences × 512, 3 répétitions) :

```bash
cd /workspace/reliquary-miner-priv && PYTHONPATH=. BENCH_MODEL=<snapshot du checkpoint> \
  BENCH_FAMILY=fs BENCH_MODES=libre_greedy,noop,fs,fast BENCH_PROMPTS=10 \
  BENCH_LENS=512 BENCH_REPS=3 /workspace/venv/bin/python ops/bench_fs_h100.py
```

| mode (160 × 512, meilleur des 3) | référence ms/pas | tokens/s | seuil de refus |
|---|---|---|---|
| libre_greedy (sans forced-seed) | **7,96** | ~20 100 | > 8,36 |
| noop (processeur inactif) | 7,97 | ~20 000 | > 8,37 |
| fs (ancien chemin) | 12,24 | ~13 100 | — |
| **fast (chemin de prod, FS_FAST=1)** | **8,61** | ~18 600 | **> 9,04** |

## 4. Bout en bout (vrai bake, chemin de prod)

`ops/e2e_bake_fs.py` (10 prompts réels × 16, EOS réel, plafond 8192), tirages
`0…01`, `0…02`, `0…03` (`E2E_SEEDS`) avec `RELIQUARY_FS_FAST=1` :

| tirage | groupes 1 / 5 / 10 (référence rapide) | tokens (max) |
|---|---|---|
| 1 | 5,0 / 6,8 / 14,2 s | 98 065 (2 406) |
| 3 | 5,6 / 6,7 / 9,1 s | 98 976 (1 263) |

(Le tirage 2 diverge d'une passe à l'autre, même sur l'ancien chemin : ne pas
l'utiliser comme référence.)

## 5. Conformité

`ops/gate_fs_fast.py` (GATE_PHASE=gen puis check) : référence rapide
**0,9366 groupe / 0,8417 pire** (actuel 0,9371 / 0,8417) → PASS obligatoire
(planchers 0,80 / 0,70).

## ⚠️ Pièges

- **Mettre le watchdog en pause pendant les bancs** (`tmux kill-session -t
  watchdog81`, relance : voir `restart_miner.sh` ligne `watchdog81`) : tant que
  le mineur est absent il relance toutes les ~3 min et son `pkill -9 -f
  EngineCore` tue le moteur du banc.
- Dans une commande ssh, jamais `pkill -f EngineCore` en clair (la commande se
  tue elle-même) : `pkill -9 -f "[E]ngineCore"`.
- Après acceptation : relancer le mineur, vérifier `RELIQUARY_FS_FAST=1` dans
  `/proc/<pid>/environ`, puis en vol g1/g5/g10 du 1er bake (référence ancien
  chemin 7,6 / 9,9 / 17,7 s ; attendu ~2 s de moins par groupe).
