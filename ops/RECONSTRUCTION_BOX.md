# Reconstruire le mineur sur une box neuve — procédure vérifiée

Écrite le 2026-08-20 après une reconstruction réelle : la H200 a redémarré,
`/workspace` était **entièrement vide** (venv, modèle, wallet, code, corpus).
Durée constatée bout en bout : ~40 minutes.

⚠️ **Chez ce fournisseur, un reboot efface `/workspace` ET change le port SSH.**
Le port n'est jamais garanti — demander le nouveau, puis le propager partout
(voir étape 8, c'est ce qui a rendu deux surveillances aveugles).

Tout le nécessaire est dans `data/box_backup_<date>/` du dépôt (modèles, corpus,
scripts, gel des dépendances). **Le wallet n'y est pas** et n'y sera jamais : la
hotkey contient une graine secrète. Il vit sur la dev box uniquement.

---

## 0. Inventaire (2 min)

```bash
ssh -o StrictHostKeyChecking=no -p <PORT> root@<IP> \
  'hostname; uptime; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; ls /workspace'
```
`/workspace` vide + GPU à 0 MiB = reconstruction complète.

## 1. Code (2 min)

```bash
cd /root/subnet81/.worktrees/miner-priv-v6     # ⚠️ V1/fill-closed depuis le 10/09
ssh -p <PORT> root@<IP> 'mkdir -p /workspace/reliquary-miner-priv'
rsync -rc --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='data/' --exclude='*.log' \
      -e "ssh -p <PORT>" ./ root@<IP>:/workspace/reliquary-miner-priv/
```

Le point de retour arrière connu-bon est le tag **`prod-2026-08-20`**
(vérifié md5-identique à la production). `git checkout prod-2026-08-20` si la
branche a divergé.

## 2. Installation (~10 min)

`ops/install_v4.sh` — venv, 212 paquets depuis le gel, torch cu130, puis le
modèle.

⚠️ **VÉRIFIER LE MODÈLE AVANT DE LANCER.** Le script d'install archivé de
l'époque v3 visait `Qwen3.5-4B` : 20 minutes de téléchargement pour rien, puis
échec au démarrage. Le modèle exact se lit sur le validateur lui-même :

```bash
curl -s http://209.20.157.231:8080/health | python3 -m json.tool | grep -A3 model
```
Au 20/08 : `Qwen/Qwen3-4B-Base` révision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.

Jalons attendus dans `/workspace/install.log` : `IMPORTS_OK`, `NVCC_OK`,
`MODEL_PREFETCHED`, `INSTALL_DONE`. L'`ERROR` du résolveur pip est normal
(`--no-deps`).

## 3. Wallet (1 min) — **copier UNIQUEMENT le public et la hotkey**

```bash
ssh -p <PORT> root@<IP> 'mkdir -p /root/.bittensor/wallets/camille81-v2/hotkeys'
scp -P <PORT> ~/.bittensor/wallets/camille81-v2/coldkeypub.txt \
    root@<IP>:/root/.bittensor/wallets/camille81-v2/
scp -P <PORT> ~/.bittensor/wallets/camille81-v2/hotkeys/hotkey81{,pub.txt} \
    root@<IP>:/root/.bittensor/wallets/camille81-v2/hotkeys/
```
⛔ **Ne JAMAIS copier `coldkey`** (la clé secrète). Contrôle :
`ssh ... 'find / -name coldkey'` doit ne rien renvoyer.

Vérifier le SS58 : `5DvpFN3QEa9iimQiA5jQaRmx8dbW2uxonM53j51Cw3kBva7q` (uid 167).

## 4. Modèles, corpus et scripts (2 min)

```bash
# Liste V1 VÉRIFIÉE le 12/09 — elle se DÉRIVE de l'environ capturé
# (`data_backups/box_<date>_coupure/env_live_*.txt`, tous les RELIQUARY_*
# qui pointent un /workspace/…). Ne pas se fier à une liste de mémoire.
R=root@<IP>; P=<PORT>; B=data_backups/box_2026-09-10_coupure
rsync -az -e "ssh -p $P" $B/final_/samples_v4.jsonl        $R:/workspace/samples_v4.jsonl
rsync -az -e "ssh -p $P" data/predictor_v5.9_*.json        $R:/workspace/predictor_v59.json
rsync -az -e "ssh -p $P" $B/prompt_scores_unique_v1.npz    $R:/workspace/
rsync -az -e "ssh -p $P" data/risk_zone_v1.json            $R:/workspace/
rsync -az -e "ssh -p $P" data/volume_v2.json               $R:/workspace/
rsync -az -e "ssh -p $P" data/burned_idx.npy               $R:/workspace/
rsync -az -e "ssh -p $P" $B/sz_blacklist.json              $R:/workspace/
rsync -az -e "ssh -p $P" ops/{launch_miner_v4.sh,restart_miner.sh,watchdog.sh} $R:/workspace/
ssh -p $P $R 'echo /workspace/launch_miner_v4.sh > /workspace/.miner_launcher; chmod +x /workspace/*.sh'
```

⚠️ **Le marqueur `.miner_launcher` doit contenir un CHEMIN**, pas une étiquette
comme `v4` : `restart_miner.sh` fait `bash $(cat .miner_launcher)`. Erreur
commise le 20/08, le premier lancement a échoué en silence.

⚠️ **Ne pas oublier `samples_v4.jsonl`** : c'est la mémoire des vedettes, relue
au démarrage. Sans lui le mineur connaît ~90 prompts payables au lieu de
~21 000, et le créneau mémo tourne à vide — une perte de revenu invisible dans
les logs. Oublié le 20/08, rattrapé une heure plus tard.

⚠️ `volume_v1.json` : **ne PAS le copier** tant que le bonus de volume n'a pas
été validé en production. Le launcher l'exporte quand même ; sans le fichier le
bonus reste inerte (log : `modèle de volume illisible — bonus désactivé`), mais
si le fichier apparaît, il s'active tout seul au redémarrage suivant.

## 4bis. MIROIR PARQUET — **BLOQUANT sur toute box neuve** (2 min)

⛔ **La note « launcher auto-protégé, export conditionnel » est FAUSSE**
(vérifiée le 12/09) : `launch_miner_v4.sh:265` exporte
`RELIQUARY_PARQUET_LOCAL_ROOT=${RELIQUARY_PARQUET_LOCAL_ROOT:-/workspace/parquet_mirror}`
**sans condition**. Or dès que `local_root` est posé, `VirtualParquetDataset`
lit un `LocalFileSystem` : répertoire absent ⇒ `no parquet files under …`,
miroir incomplet ⇒ `len()` ≠ `RELIQUARY_PARQUET_EXPECTED_LEN` ⇒ la garde lève.
C'est le « générateur qui plante en boucle » des box neuves.

```bash
ssh -p <PORT> root@<IP> '/workspace/venv/bin/python - <<EOF
import os; os.environ.setdefault("HF_HOME","/workspace/hf")
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="R0mAI/opencodeinstruct-curated",
  revision="d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7", repo_type="dataset",
  allow_patterns=["data/*.parquet"], local_dir="/workspace/parquet_mirror"))
EOF'
```
Disposition attendue : **`/workspace/parquet_mirror/data/*.parquet`**
(`data_dir` vaut `data` par défaut ; le code n'applique **aucun**
`filename_prefix` — ce filtre `train-` ne concerne que l'env math).
Repo et révision : `_CURATED_REPO` / `_CURATED_REVISION` dans
`reliquary/environment/opencodeinstruct.py`, surchargeables par
`RELIQUARY_OCI_REPO` / `RELIQUARY_OCI_REVISION`.

**Contrôle obligatoire avant lancement** — `len()` est le consensus
prompt-range ; s'il diverge, c'est 100 % de `prompt_out_of_range`, en silence :
```bash
ssh -p <PORT> root@<IP> 'cd /workspace/reliquary-miner-priv && \
  RELIQUARY_PARQUET_LOCAL_ROOT=/workspace/parquet_mirror PYTHONPATH=. \
  /workspace/venv/bin/python -c "
from reliquary.environment.virtual_parquet import VirtualParquetDataset as V
d=V(\"R0mAI/opencodeinstruct-curated\",\"d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7\",
    columns=[\"input\",\"structured_cases\"], local_root=\"/workspace/parquet_mirror\")
print(len(d))"'   # DOIT afficher 2481806
```
Repli si le miroir est indisponible : `RELIQUARY_PARQUET_LOCAL_ROOT=""`
(chemin HF distant, fonctionnel mais +0,95 s de réseau par fenêtre et 5,4 %
de fenêtres dégradées par un timeout).

## 5. Réseau (2 min) — deux pièges

```bash
ssh -p <PORT> root@<IP> '
for i in 1 2 3; do curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
  --max-time 10 http://62.238.81.36:8000/health; done   # V1 depuis le 10/09
for u in api.drand.sh api2.drand.sh api3.drand.sh drand.cloudflare.com api.drand.secureweb3.com; do
  printf "%s %s\n" "$u" "$(curl -s -o /dev/null -w %{http_code} --max-time 6 https://$u/public/latest)"
done'
```
- Validateur injoignable → tunnel inverse
  (`ssh -f -N -R 8080:209.20.157.231:8080 -p <PORT> root@<IP>`). Le launcher
  auto-détecte direct/tunnel.
- **Tout miroir drand qui ne répond pas doit être retiré de
  `RELIQUARY_DRAND_URLS`** : un miroir mort gèle la boucle 15-20 s un tirage sur
  cinq, donc des flips ratés. `secureweb3` était mort les 15 et 20/08.

## 6. Gate forced-seed (~5 min) — **OBLIGATOIRE**

`ops/run_gate_v4.sh`. Elle vérifie que la carte reproduit les tokens que le
validateur reconstruira par teacher-forcing. Sans elle, risque de 100 % de
rejets.

⚠️ **Ne pas sourcer `ops/bench_env.sh`** : il force `SMOKE_CKPT` sur un vieux
checkpoint 2B. Le script de gate v4 le dés-définit lui-même.

Planchers : 0,80 groupe / 0,70 pire rollout. Références H200 (identiques sur
deux cartes différentes) : eager **0,9572 / 0,9123**, graphs **0,9674 / 0,9388**.

## 6bis. BANC DE PERFORMANCE — **OBLIGATOIRE, ET AVANT DE S'ENGAGER SUR LA BOX**

🪤 **Vécu : on est déjà tombé sur une H200 nettement plus lente que celle
d'origine.** Deux cartes portant le même nom ne se valent pas — fréquence
mémoire, throttling thermique, voisinage sur l'hôte, quota CPU du conteneur
(mesuré une fois à **24 CPU**, pas 192, ce qui affamait vLLM). Le mineur
DÉMARRE quand même et passe la gate forced-seed : rien dans les journaux ne
dit « cette carte est lente ». On ne s'en aperçoit qu'en perdant du revenu
pendant des heures, en croyant à une régression de code.

⇒ **Mesurer AVANT de payer plus loin et avant de considérer la box comme
acquise.** Si les chiffres sont sous la référence, RENDRE LA BOX et en
reprendre une autre — c'est moins cher qu'une journée de minage dégradé.

```bash
# 1. débit par séquence (le chiffre qui décide) — ~1 h
bash ops/bench_sprint_matrix.sh          # per-seq à 1/2/4/8/16 prompts × FS ON/OFF
# 2. passe forced-seed + CUDA graphs — ~10 min
python3 ops/test_fs_graph_gpu.py         # graphe contre eager, bit-à-bit
```

**Références mesurées sur les BONNES cartes** (rejeter si on est nettement
dessous — un écart de quelques % est du bruit, 20 % ne l'est pas) :

| mesure | référence | source |
|---|---|---|
| débit par séquence @32 séquences | **102 tok/s** (H200) · 70 (H100) | banc étage 1 |
| passe forced-seed, 32 séquences | **1,181 ms** | `test_fs_graph_gpu.py`, 08/09 |
| passe forced-seed, 48 séquences | **1,451 ms** | idem |
| graphe CUDA contre eager | 2 à 5 % seulement | idem — ne PAS en attendre plus |
| gate forced-seed, eager | 0,9572 / 0,9123 | §6 |
| gate forced-seed, graphs | 0,9674 / 0,9388 | §6 |

**Contrôle en vol, une fois le mineur lancé** — le juge le plus rapide :
`grep -oaE "groupe 1/[0-9]+ prêt à [0-9.]+s" /workspace/miner.log`.

⛔ **JUGER SUR LA MÉDIANE D'AU MOINS ~15 BAKES, JAMAIS SUR LES PREMIERS.**
La règle « à 5 s ou plus la carte est lente » que portait ce paragraphe était
**FAUSSE** — corrigée le 12/09 après l'avoir appliquée à tort à une carte saine.
Distribution mesurée sur une carte de RÉFÉRENCE (n=181, même code, même env) :

| | p25 | **p50** | p75 | max | part ≥ 5 s |
|---|---|---|---|---|---|
| carte de référence | 3,2 s | **3,8 s** | 4,6 s | 13,7 s | **16 %** |

**16 % des bakes d'une carte SAINE dépassent déjà 5 s** : un seuil sur un
relevé isolé condamne une bonne carte une fois sur six. Et les 2-3 premiers
bakes suivent un moteur à froid — le 12/09 ils sont sortis à 5,0 et 5,3 s sur
une carte dont la médiane s'est ensuite établie à **3,75 s** (n=14), soit la
référence exacte.

⇒ **Critère** : médiane ≤ 4,6 s (le p75 de référence) = carte comparable ;
au-delà de ~5,5 s de MÉDIANE = refaire le banc §6bis avant d'accuser le code.

⚠️ Vérifier aussi le quota CPU réel du conteneur (`nproc` ment souvent) :
`cat /sys/fs/cgroup/cpu.max`. Sous ~24 CPU, le grading parallèle affame vLLM.

## 7. Lancement

```bash
ssh -p <PORT> root@<IP> 'bash /workspace/restart_miner.sh'
```
Relance aussi `watchdog81` et `monitor`. Contrôles dans `/workspace/miner.log` :
- `prédicteur ACTIF ... predictor_v59.json (499310 mots)`
- `table de scores ACTIVE ... prompt_scores_unique_v1.npz (2481806 prompts)`
- `payable_memo: ~288000 lignes chargées, ~65600 payables connus`
- `malus anti-court ACTIF ... lambda=0.08`
- `bonus de volume ACTIF ... volume_v2.json (…, mu=0)` (μ=0 ⇒ inerte)
- `launch_v4: egress DIRECT vers le validateur`
- heartbeat `quota=N/64` (V1 : le quota est 64, pas 32 — cf. passationr2.md §1)
- zéro `ERROR`, zéro `Traceback`, zéro `seed_mismatch`/`token_tampered`

## 8. Propager le nouveau port (5 min) — sinon les surveillances sont aveugles

```bash
grep -rln '\b<ANCIEN_PORT>\b' scripts/ ops/ | xargs sed -i 's/\b<ANCIEN_PORT>\b/<PORT>/g'
```
Puis **arrêter et relancer** les boucles de surveillance : un script bash modifié
en cours d'exécution ne recharge pas sa boucle. Le 20/08, deux surveillances sont
restées muettes 40 minutes en pointant l'ancien port.

Surveillances à relancer : `scripts/live_monitor.sh`, `scripts/watch_fixes.sh`,
`scripts/watch_competition.sh`, `scripts/harvest_window_timing.py`,
`ops/pull_samples_v4.sh`, `scripts/poll_dashboard.sh`.

## 9. Reprendre le rapatriement des données

`ops/pull_samples_v4.sh` porte désormais une **garde anti-écrasement** : il
refuse tout transfert qui rétrécirait le fichier local et bascule le nouveau flux
dans un fichier daté. Sans elle, `rsync --append-verify` écrase l'historique
local par les fichiers neufs de la box — c'est arrivé le 20/08 21h45,
`submits_v4.jsonl` réduit de plusieurs milliers de lignes à 10.

---

## Ce qui est réellement irremplaçable

| Élément | Où | Reconstructible ? |
|---|---|---|
| Code | GitHub (tag `prod-2026-08-20`) | oui |
| venv + modèle HF | recette `install_v4.sh` + gel | oui, ~10 min |
| Prédicteur, malus | `data/box_backup_*/models/` | non — entraînés sur des données accumulées |
| **Corpus mémo** | `data/box_backup_*/corpus/` | **non** — des jours de minage |
| Wallet | dev box uniquement | non — **et jamais dans git** |
| Données de timing du jour | `data/` de la dev box | non |
