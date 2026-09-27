# Mineur SFT (tâche corpus) — 27/09/2026

La tâche `corpus-code-v1` (job `code-qwen38-27b-v1`, Qwen3.8-27B) paie 10 % fixes
de l'émission au token vérifié par TOPLOC. Le mineur d'upstream
(`reliquary corpus mine`) génère un prompt à la fois : 100 tok/s sur une H100.
Celui-ci garde la carte pleine et soumet dans l'ordre du curseur.

| fichier | rôle |
|---|---|
| `book.py` | carnet de curseurs : générer c..c+K en parallèle, soumettre strictement dans l'ordre, réaligner après un refus |
| `miner.py` | flux vLLM (admission pilotée par le cache KV, un prompt par pas), fil de soumission ; protocole = code upstream importé tel quel |
| `test_book.py`, `test_miner.py` | 17 tests (upstream requis : à lancer sur la box) |
| `ops/install_corpus.sh` | venv vLLM 0.30 + upstream en `-e` + téléchargement du modèle |
| `ops/check_render.py` | contrat, empreinte du checkpoint, rendu des prompts identique aux soumissions R2 |
| `ops/corpus_bench.py` | banc lot + auto-audit TOPLOC (chargeur et audit du validateur) |
| `ops/run_sft_dry.sh` | essai à blanc (rien n'est soumis) + auto-audit |
| `ops/run_sft_live.sh` | superviseur en réel (relance ; arrêt définitif sur ban / job fini / `SFT_STOP`) |
| `ops/vigie_sft.py` | cron dev box : au 1er échec d'audit confirmé, pose `SFT_STOP` et coupe le mineur |

Pré-requis sur la box : upstream `origin/main` (f901d2a) dans `/workspace/reliquary_upstream`,
paquet `reliquary-code` 0.1.0a2 (github `reliquadotai/reliquary-environments`,
`environments/code/reliquary_code`) + `verifiers@b2e4e815`, contrat = champ `contract`
de l'entrée `corpus-code-v1` du registre R2, wallet = coldkeypub + hotkey seulement.

Mesures (H100 80 Go HBM3) : lot de 16 prompts 560-722 tok/s ; flux continu final
584 tok/s de moyenne sur 30 min (~900 en régime), 0 prompt régénéré, auto-audit
64/64 (pire écart 20 / seuil 60). Première soumission acceptée le 27/09 17:22 UTC.
