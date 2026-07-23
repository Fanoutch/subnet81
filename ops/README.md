# Ops — remise en route sur box GPU fraîche

Scripts pour redéployer le miner de zéro. Les valeurs de perf et la config sont
dans le CLAUDE.md racine et les mémoires. Débit acquis : ~14k tok/s batch 20 (H200).

## Séquence sur une box neuve
1. `bash ops/vllm_bringup.sh` (VENV/HF_HOME sur /workspace ; ~7 min ; smoke test inclus)
2. `bash ops/install_bt.sh` (bittensor 10.5.0 + fix codec scalecodec)
3. copier le wallet : hotkey81 + coldkeypub SEULEMENT (jamais la coldkey)
4. `bash ops/test_meta.py` — vérifie metagraph + hotkey enregistrée (5s)
5. `bash ops/launch_miner.sh` — mine (batch 20, CUDA graphs, code-only)

## Réglages clés (voir CLAUDE.md pour le détail mesuré)
- BAKE_BATCH_SIZE=20, VLLM_CUDA_GRAPHS=1, VLLM_GPU_FRACTION=0.55 (H200)
- restart_miner.sh : kill propre (évite le piège pkill qui tue son propre ssh)
- diag_submit.sh : force une soumission (ZONE_SIGMA_MIN=0.0) pour tester le pipeline
- run_gates.sh : re-valider la conformité forced-seed sur une NOUVELLE carte
