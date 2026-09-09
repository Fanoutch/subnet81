# Bascule v6 « fill-closed » — procédure jour J (branche `port/v6-fill-closed`)

Référence upstream : PR #224 `integration/reliquary-v1-final` (tip `589fd43`, 07/09/2026). Déploiement annoncé :
profil `qwen3-4b-base-dapo-fill-closed-v6`, `protocol_version=6`, validateur avec `RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED=1`.
Génération byte-identique v5 ; seuls le régime de fenêtre, l'économie et le wire `/state` changent.
Analyse complète : mémoire `project_upstream_watch_2026_09_05` + rapports `scratchpad/v6port/rapport_*.md` (session 08/09).

## 0. Ce que contient la branche (inerte tant que `RELIQUARY_PROTOCOL_VERSION=5`)
- wire : `GrpoBatchState.fill_closed` (sous-modèle tolérant), `checkpoint_n` optionnel, `RejectReason.reveal_not_selected`,
  `generation_profile_id()` v6, `min_length=None` (sampler HF).
- moteur : veto de tir v6 (phase / fraîcheur du `/state` / cutoff − 40 s / budget env / quota 32), garde pré-flip neutralisée,
  `_sealed_window` inerte sur `batch_filled`, quota exact (remboursement des rejets de stade precommit, `rate_limited` = cap),
  pause du bake (gap 503, phase ≠ collecting, budget env, quota, cutoff), heartbeat 60 s, backoff 503, open exact du validateur,
  reload du checkpoint sur révision nouvelle à n égal (seule divergence v5 : cas rollback/nouveau repo, bénéfique).
- launcher : bloc `if RELIQUARY_PROTOCOL_VERSION = 6` (CURFEW 0, gardes 999999, marge cutoff 40 s, backoff 0,25 s,
  `LOCAL_TOKEN_AUTH=1`, `GRADE_TIMEOUT_S=5`, `WATCHDOG_WEDGE_S=2700`), garde `/health` étendue (sha des templates = abort,
  sha du contrat = avertissement) ; watchdog : heartbeat = signe de vie seulement en repos légitime, seuil paramétrable.

## 1. Signal de bascule
- `curl -s http://209.20.157.231:8080/health | python3 -c 'import sys,json;h=json.load(sys.stdin);print(h["image_revision"][:7],h["protocol_version"],h["generation_profile_id"])'`
  → `… 6 qwen3-4b-base-dapo-fill-closed-v6`. Tant que c'est `5 …reasoning-v5`, NE RIEN FAIRE.
- Le mineur v5 en place s'arrêtera seul de produire (rejets `generation_contract_mismatch`) ; la garde du launcher refuse de
  redémarrer en v5 contre un validateur v6 → boucle d'abort du watchdog : c'est attendu, c'est le moment de basculer.

## 2. Bascule (box, ~3 min)
1. Sauvegarde : `cp -r /workspace/reliquary-miner-priv /workspace/bak-avant-v6-$(date -u +%Y%m%d-%H%M)` ; `cp /workspace/launch_miner_v4.sh /workspace/launch_miner_v4.sh.bak-avant-v6`.
2. Code : depuis la dev box, `rsync -rc --exclude='__pycache__' --exclude='*.log' --exclude='data/' /root/subnet81/.worktrees/miner-priv-v6/ root@157.10.162.245:/workspace/reliquary-miner-priv/` (port SSH courant) puis `rsync` du launcher/watchdog/restart vers `/workspace/`.
   Vérif : `rsync -rcn` vide.
3. Version : `export RELIQUARY_PROTOCOL_VERSION=6` puis `bash /workspace/restart_miner.sh` (le restart écrit `/workspace/.protocol_version`,
   relu par le launcher et le watchdog quand l'env n'est pas là). Alternative durable : passer `:-5` → `:-6` dans le launcher de la box.
4. Contrôles dans la minute (`/workspace/miner.log`) :
   - `constantes OK: 6 qwen3-4b-base-dapo-fill-closed-v6` et `[garde] parite OK`.
   - `tr '\0' '\n' < /proc/$(pgrep -f 'reliquary.cli.main mine' | head -1)/environ | grep -E 'PROTOCOL_VERSION|FIRE_CURFEW|PREFLIP|LOCAL_TOKEN_AUTH|GRADE_TIMEOUT'`
     → 6 / 0 / 999999 / 1 / 5.0.
   - premier `/state` 200 : ligne `heartbeat window=… phase=collecting quota=0/32` ; aucun `Traceback`.
   - premiers verdicts : ZÉRO `generation_contract_mismatch`, `wrong_checkpoint`, `seed_mismatch`, `token_tampered`.

## 3. Repli
- `export RELIQUARY_PROTOCOL_VERSION=5` + `bash /workspace/restart_miner.sh` (ou restaurer `launch_miner_v4.sh.bak-avant-v6`) :
  tout le code v6 est gaté, le mineur redevient celui du 07/09. N'a de sens que si le validateur est revenu en v5.
- Repli code complet : `bak-avant-v6-*` + restart.

## 4. À mesurer sur les 10 premières fenêtres (nouvelles grandeurs, ne PAS comparer aux payées/fen v5)
- durée réelle des fenêtres (flip à flip) et durée du trou 503 ; `fill_closed.proven/admitted/remaining` par env dans `/state`.
- nos precommits acceptés par fenêtre (cible 32), motifs `veto_*` dans `fire_diag`, `quota v6 a -> b` dans le log.
- rate de nos têtes : `payload_bytes / (t_precommit − open)` vs le marché (archives R2 : `precommit_arrival_ts`).
- verdicts : `token_tampered` (doit rester ~0 avec la gate locale), `bad_termination`, `out_of_zone` (chaque rejet = 1/32).
- payées : `rewards_by_hotkey` dans les archives R2 (par lot assemblé, pool/env/16 × eos_tokens/Σ), et `fill_closed_burned_*` côté validateur.
- délai ouverture → 1re entrée (attendu ~8 s hors reload ; ~55 s si reload de checkpoint à l'open → chantier « charger pendant le gap 503 »).

## 5. Leviers connus APRÈS stabilisation (un à la fois, 30 fenêtres)
1. Charger le checkpoint successeur pendant le gap 503 (`/checkpoint` upstream, rapport D §1.2) — le plus gros levier de rate.
2. `SPRINT_SIZE=8` (off) / `HEAD_FIFO=0` : le débit total remplit le quota, la latence de tête ne vaut que le rate.
3. `RELIQUARY_VOLUME_MU` > 0 : paiement ∝ eos_tokens, rate ∝ octets (leçon 03/09 : contrôle faible de la longueur réelle).
4. 2e hotkey (hotkey81.2) sur le GPU inoccupé : +32 places par fenêtre.
5. Garde-fou dette de preuve : cesser d'envoyer après le 1er échec de preuve d'une fenêtre (le 2e = `batch_filled` jusqu'à 30 min).
