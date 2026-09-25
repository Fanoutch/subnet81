# Bascule du mineur vers MATH (openmathinstruct) — 24/09

Branche `math/v1` (tirée de `h100/v1` @ `71b8ec5`, = la box au md5 près le
24/09). Le mineur CODE reste intact : son code dans
`/workspace/reliquary-miner-priv`, son lanceur `/workspace/launch_miner_v4.sh`.
Le mineur math vit à côté : `/workspace/reliquary-miner-math` +
`/workspace/launch_miner_math.sh`, journaux dans `/workspace/math/`.
2e H100 (Lium swift-lion-c9, `ssh root@162.243.220.30 -p 20301`), hotkey `hotkey81.2` (uid 215, même coldkey ; `RELIQUARY_HOTKEY` pour changer),
même réplique (`replica81`, générique).

## Pourquoi (mesuré, mémoire `project_prix_par_env_2409`)
Prix d'émission par env : code 0,057 (au plancher, en baisse), math 0,529 ⇒ un
groupe math ≈ 9 groupes code. La lane math ne se remplit pas dans ~60 % des
fenêtres depuis le départ d'un gros fournisseur (46801).

## Ce que la branche change par rapport au mineur code
| # | changement | preuve |
|---|---|---|
| 1 | `openmathinstruct.get_problem` rend le prompt par le template v5/v6 (« Solve the following math problem step by step. ») — BLOQUANT : sans lui 100 % `prompt_mismatch` et disjoncteur sur la COLDKEY | `tests/test_omi_prompt_template.py` ; parité 200/200 prompts réels contre upstream 0ae6f3b, univers 13 972 791 |
| 2 | Miroir local de `BOXED_ANSWER_TAMPERED` (stage à dette) : groupe jeté si un token de la dernière boîte a p < 2e-3 avec argmax ≥ 0,98 (validateur : 1e-3 / 0,99) | `tests/test_local_boxed_screen.py` ; parité de la recherche de boîte 3 008/3 008 rollouts math R2, 0,27 ms/rollout |
| 3 | Veto du cooldown de CONTENU par empreinte au tirage + dédoublonnage dans la fenêtre | `tests/test_content_digest_veto.py` ; 28 % des prompts math sont brûlés (168/600 mesurés) |
| 4 | `ops/launch_miner_math.sh` : env math, prior/table/modèles/mémo/fantômes/index code DÉBRANCHÉS (tirage uniforme dans la tranche), `OMI_SOURCES=""`, cap 4096, 128 envois, journaux séparés, garde qui rend un VRAI prompt math | `bash -n` |
| 5 | `scripts/refresh_burned_digests.py` (dev box) : empreintes brûlées depuis R2 → box | exécuté : 114 974 empreintes |

## Déploiement (au trou 503, go utilisateur)
```bash
BOX=root@162.243.197.137; P=20301
W=/root/subnet81/.worktrees/miner-priv-mathv1
# 1. code math à côté du code (le mineur code n'est PAS touché)
rsync -rc --exclude='__pycache__' --exclude='*.log' --exclude='data/' \
  -e "ssh -p $P" $W/ $BOX:/workspace/reliquary-miner-math/
scp -P $P $W/ops/launch_miner_math.sh $BOX:/workspace/
# 2. empreintes brûlées
python3 $W/scripts/refresh_burned_digests.py --push --box $BOX --port $P
# 3. garde à blanc (constantes + prompt math rendu) SANS lancer le mineur
ssh -p $P $BOX 'cd /workspace/reliquary-miner-math && sed -n "/^\/workspace\/venv\/bin\/python - <<.EOF/,/^EOF/p" /workspace/launch_miner_math.sh | head -n -1 | tail -n +2 > /tmp/garde.py; RELIQUARY_PROTOCOL_VERSION=6 PYTHONPATH=. HF_HOME=/workspace/hf RELIQUARY_OMI_SOURCES= /workspace/venv/bin/python /tmp/garde.py'
# 4. bascule : marqueur + restart au trou 503
ssh -p $P $BOX 'echo /workspace/launch_miner_math.sh > /workspace/.miner_launcher && bash /workspace/restart_miner.sh'
```
Rafraîchir les empreintes toutes les ~30 min (cron dev box, étape 2).

## Repli (1 minute)
```bash
ssh -p $P $BOX 'echo /workspace/launch_miner_v4.sh > /workspace/.miner_launcher && bash /workspace/restart_miner.sh'
```

## À surveiller dans les premières fenêtres
- `[garde] prompt math rendu OK` au démarrage ; ZÉRO `prompt_mismatch`,
  `seed_mismatch`, `token_tampered`, `boxed_answer_tampered` dans les verdicts.
- `pre_bake[local_boxed_answer]` : fréquence (coût du filtre).
- drops locaux de terminaison (cap 4096) : s'ils pèsent, remonter
  `RELIQUARY_MATH_MAX_NEW_TOKENS` vers 8192.
- `empreintes brûlées (contenu) rechargées: ~115 000` au démarrage.
- R2 : nos groupes math payés/fenêtre, part du total mineur (plancher 2 %).

## Ensuite (une chose à la fois)
1. Mémo math : rallumer `RELIQUARY_MATH_MEMO_SLOT=1` / `..._HEAD_SLOTS` quand
   `samples_math.jsonl` contient des ex-payables.
2. Prior math (le rendement en zone par prompt est l'inconnue n°1).
3. Miroir parquet OMI local (latence des lectures HF).
