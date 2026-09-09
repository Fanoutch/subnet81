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
cd /root/subnet81/.worktrees/miner-priv-port-v4-dapo
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
B=data/box_backup_<date>
zcat $B/models/predictor_v50.json.gz  | ssh -p <PORT> root@<IP> 'cat > /workspace/predictor_v50.json'
zcat $B/models/risk_short_v1.json.gz  | ssh -p <PORT> root@<IP> 'cat > /workspace/risk_short_v1.json'
zcat $B/corpus/samples_v4.jsonl.gz    | ssh -p <PORT> root@<IP> 'cat > /workspace/samples_v4.jsonl'
scp -P <PORT> ops/{launch_miner_v4.sh,restart_miner.sh} $B/scripts/watchdog.sh root@<IP>:/workspace/
ssh -p <PORT> root@<IP> 'echo /workspace/launch_miner_v4.sh > /workspace/.miner_launcher; chmod +x /workspace/*.sh'
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

## 5. Réseau (2 min) — deux pièges

```bash
ssh -p <PORT> root@<IP> '
for i in 1 2 3; do curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
  --max-time 10 http://209.20.157.231:8080/health; done
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

## 7. Lancement

```bash
ssh -p <PORT> root@<IP> 'bash /workspace/restart_miner.sh'
```
Relance aussi `watchdog81` et `monitor`. Contrôles dans `/workspace/miner.log` :
- `prédicteur ACTIF ... (65460 mots)`
- `malus anti-court ACTIF ... lambda=0.08`
- `payable_memo: ~29900 lignes chargées, ~21000 payables connus`
- `modèle de volume illisible — bonus désactivé` (**attendu**)
- `launch_v4: egress DIRECT vers le validateur`
- zéro `ERROR`, zéro `Traceback`

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
