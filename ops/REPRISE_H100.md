# Reprise H100 — restaurer EXACTEMENT le setup du 2026-08-07

But : partir d'une H100 vierge et revenir à l'état de production qui a donné
le **rang 1 PAYÉ** (fenêtre 27936) : streaming par groupe + sprint 12 s +
porte k=2-only 0.32 + prédicteur v1.1. Tout le nécessaire est dans ce dépôt
au commit **`25e1a29`** (code = `f43e68f`, sauvegardes = `25e1a29`).

## 0. Avant tout : vérifier l'upstream

```bash
bash /root/subnet81/scripts/check_reliquary_updates.sh
```
Si le validateur a changé de protocole/profil depuis le 2026-08-07
(`curl http://209.20.157.231:8080/health` → `generation_profile_id` doit être
`qwen35-4b-auction-v3`, `protocol_version 3`), relire les diffs upstream AVANT
de lancer — un port de conformité peut être nécessaire.

⚠️ **Flat auction (PR #171, mergée 2026-08-07, PAS ENCORE LIVE au moment de ce
document)** : quand elle s'activera, tout k∈[2,6] vaudra 1.0 et le tie-break
débit deviendra LE classement → notre porte `AUCTION_MIN_SCORE=0.32` (k=2 only)
deviendra CONTRE-PRODUCTIVE. À l'activation : `RELIQUARY_AUCTION_MIN_SCORE=0`
(ou recalibrer), et le débit/latence primeront sur la sélection. Vérifier
`/health` au moment de la reprise. Cf. mémoire projet flat_auction_2026_08_07.

## 1. Louer la box

H100 80 Go, **egress DIRECT** vers 209.20.157.231:8080 (tester
`curl -s http://209.20.157.231:8080/health | head -c 200` avant d'installer
quoi que ce soit). Une seule carte suffit. ⚠️ Les box Lium s'effacent au
reboot et changent de port SSH — tout l'état persistant vit dans ce dépôt.

## 2. Environnement Python (les pièges connus, dans l'ordre)

```bash
python3 -m venv /workspace/venv && source /workspace/venv/bin/activate
pip install -r <(grep -vE "^(nvidia-|torch)" ops/prod_backup_2026-08-07/pip_freeze.txt)  # base
pip install torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install vllm==0.24.0 transformers==5.9.0 bittensor==10.5.0
pip install nvidia-cuda-nvcc-cu13==13.0.*        # nvcc DANS le venv (piège n°1)
```
Le gel exact des 212 paquets : `ops/prod_backup_2026-08-07/pip_freeze.txt`.
Les pièges vLLM/Qwen3.5 (préprocesseur, text-only, gdn triton, deep_gemm off,
flashinfer sampler off) sont TOUS gérés par le code + le lanceur — ne pas
improviser, utiliser `start_miner.sh` tel quel.

## 3. Code + configuration

```bash
git clone git@github.com:Fanoutch/subnet81.git /root/subnet81-repo
cd /root/subnet81-repo && git checkout 25e1a29
rsync -rc --exclude='__pycache__' ./ /workspace/reliquary-miner-priv/   # (le dépôt EST reliquary-miner-priv)
cp ops/prod_backup_2026-08-07/start_miner.sh      /workspace/
cp ops/prod_backup_2026-08-07/prompt_predictor_v11.json /workspace/
cp ops/prod_backup_2026-08-07/prompt_predictor_k2_cap2600.json /workspace/   # rollback prédicteur
```
`start_miner.sh` porte TOUTE la config : `RELIQUARY_PROMPT_PREDICTOR=v11`,
`SPRINT_MAX_WAIT_S=12`, `SAMPLE_DUMP`, `TRUNC_DIAG`, batch 16 / chunk 64,
et il exec `ops/launch_miner.sh` (porte 0.32 = défaut du code, sprint 4 =
défaut du code, `CUDA_HOME` venv, deep_gemm off — tout y est).

## 4. Portefeuille (depuis la dev box, JAMAIS la coldkey secrète)

```bash
rsync -av ~/.bittensor/wallets/camille81-v2/hotkeys/hotkey81* \
  root@<BOX>:~/.bittensor/wallets/camille81-v2/hotkeys/
rsync -av ~/.bittensor/wallets/camille81-v2/coldkeypub.txt \
  root@<BOX>:~/.bittensor/wallets/camille81-v2/
```
La hotkey81 est ENREGISTRÉE sur le netuid 81 — ne pas en recréer une.

## 5. Lancement + vérification (dans l'ordre, ~10 min)

```bash
tmux new-session -d -s miner "bash /workspace/start_miner.sh 2>&1 | tee -a /workspace/miner.log"
```
Checklist de bonne santé (grep dans /workspace/miner.log) :
1. `prédicteur ACTIF: /workspace/prompt_predictor_v11.json` (~169k mots)
2. `classement de tranche: ~5000 prompts notés ... en 1.3-1.8s` (1×/fenêtre,
   avec `N écartés en cooldown` > 0 de temps en temps)
3. `stream_fire: groupe 1/16 prêt à 5-15s` (streaming OK)
4. `sprint: balayage enclenché à ≤12.0s` (sprint OK)
5. `submitted window=... accepted=True` (⚠️ chercher CE motif — « submit ok »
   n'existe pas dans le log, piège de monitoring documenté)
6. Verdicts : `curl /verdicts/<ss58 hotkey81>` → rangs des k=2 (attendu 9-38,
   top-8 quand le 1er k=2 tombe au lot 1)

## 6. Défauts CONNUS de cette version (audit multi-agent 2026-08-07)

- **Course reload-checkpoint × bake en vol** : ~1 rechargement/20 tue le
  moteur vLLM en laissant le process VIVANT (11 h de production nulle dans
  la nuit du 06→07). Fix non écrit au moment de ce document. **Poser un
  watchdog dès le lancement** : `ops/prod_backup_2026-08-07/watchdog.sh`,
  MAIS corriger d'abord son défaut : au démarrage sur un log ancien, il
  lit un vieux `stream_fire` et peut tuer un mineur en plein chargement —
  ajouter « ignorer si le process a < 15 min d'âge ».
- Premier cycle post-rechargement : 215-408 s au lieu de 40 s (~20 % de la
  production) — chantier « masquer le re-warm ».
- Le classement du prédicteur est ~aléatoire EN TÊTE (1er vrai k=2 médian en
  position 35) : le sprint mise sur un top 4 sans signal. Chantier v2.
- Drainage triangulaire des lots : +15-25 % de débit possibles par
  enfilement continu.

## 7. Données pour réentraîner le prédicteur

Le maître des échantillons étiquetés est sur la DEV BOX :
`/root/subnet81/data/samples_code.jsonl` (+ copie figée du 2026-08-07 :
`ops/prod_backup_2026-08-07/samples_code.jsonl`, 22 030 lignes, champs
`n_truncated`/`completion_lens` depuis la ligne ~9 200). Sur la nouvelle box,
`RELIQUARY_SAMPLE_DUMP` repart d'un fichier neuf ; relancer
`scripts/pull_samples.sh` côté dev box pour cumuler (dédup par contenu).
Repères d'époques (v0/v1/v1.1) : `ops/prod_backup_2026-08-07/predictor_boundary.txt`
(les numéros de ligne se réfèrent au fichier de la BOX du 06-07, pas au cumul).
