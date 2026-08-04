# Plan de reprise — dès qu'un GPU est disponible

**Contexte** : miner 4B/v3 conforme au wire (tous gates automatiques verts),
mais **21 soumissions rejetées `bad_termination`** le 2026-08-03. Cause racine
trouvée et corrigée le 2026-08-04 **sans GPU** ; il reste à valider sur machine.
État détaillé : mémoire `project_bad_termination_root_cause`.

Box précédente (H100 `93.120.231.186:40297`) **morte**. Code prêt sur la dev box
`/root/subnet81/reliquary-miner-priv/` + GitHub. Rien n'est déployé.

---

## Étape 0 — Remise en route (~10 min)

```bash
# 1. rsync du code vers la box
rsync -rc --exclude='__pycache__' --exclude='*.log' --exclude='data/' \
  /root/subnet81/reliquary-miner-priv/ root@<IP>:/workspace/reliquary-miner-priv/

# 2. bring-up (si box fraîche) — voir ops/README.md
bash ops/vllm_bringup.sh && bash ops/install_bt.sh
# 3. wallet : hotkey81 + coldkeypub SEULEMENT (jamais la coldkey secrète)
# 4. vérifier la hotkey enregistrée
bash ops/test_meta.py
```

⚠️ Si l'egress vers le validateur est bloqué (constaté sur les box H100),
monter le tunnel inverse DEPUIS la dev box :
`ssh -f -N -R 8080:209.20.157.231:8080 -p <port> root@<IP>` puis lancer avec
`RELIQUARY_VALIDATOR_URL=http://127.0.0.1:8080`.

---

## Étape 1 — Simulateur hors-ligne ✅ **FAIT le 2026-08-04 (sans GPU)**

`scripts/simulate_validator.py` (= `ops/`) — construit, exécuté, **vert**.
Intégré comme **gate G9** dans `scripts/check_4b_port.sh`.

Il fait juger la sortie RÉELLE de notre correctif par le CODE du validateur
upstream (`origin/main`, extrait à la volée) — pas une ré-implémentation qui
pourrait dériver. Boucle fermée : les entrées cassées passent dans notre vrai
`truncate_at_first_eos` et c'est son résultat qui est jugé.

Résultat (jugé par le validateur lui-même) :
| Scénario | Verdict |
|---|---|
| EOS retiré par vLLM (0 EOS) — les 13 premiers rejets | 🔴 `bad_termination` |
| EOS au milieu (cas `ignore_eos`) | 🔴 `bad_termination` |
| Deux EOS | 🔴 `bad_termination` |
| **Après notre fix** (troncature) | 🟢 **accepté** |
| Après fix + groupe payable k=3/8 | 🟢 **accepté**, utilité 0.3026 |
| Groupe sain mais unanime (σ=0) | 🟡 `out_of_zone` (attendu) |

⚠️ Ne couvre PAS `SEED_MISMATCH` / `GRAIL_FAIL` (chemin preuve GPU) → étape 4.

---

## ⚠️ Ce qui N'EST PAS encore validé (à faire tourner sur GPU)

Le fix de terminaison est **prouvé correct** (étape 1) mais **pas complet** :

| Acquis | État |
|---|---|
| Le mineur ne peut plus soumettre un rollout invalide | ✅ prouvé |
| `bad_termination` de notre fait | ✅ impossible |
| **Le mineur soumet-il encore quelque chose ?** | 🟡 **fortement amélioré, à confirmer** |
| Quelle config d'arrêt choisir | ❓ à mesurer (étape 2) |

**Le risque précis** : la troncature ne *rend pas* un rollout valide, elle
*empêche* d'envoyer ceux qui ne le sont pas. Si vLLM retire le token d'arrêt à
l'arrêt naturel, alors 100% des groupes seraient jetés localement → **zéro
`bad_termination`, mais zéro soumission**. Silence au lieu d'erreur, ce qui est
plus insidieux (rien ne clignote).

**Réponse CONSTRUCTIVE ajoutée le 2026-08-04** : jusque-là le fix était
purement défensif (« ne pas envoyer de mauvais ») sans garantir qu'on envoie du
bon. Complété : quand vLLM dit s'être arrêté (`finish_reason='stop'`) mais ne
nomme pas le token (`stop_reason=None`), c'est l'EOS du MODÈLE qui a stoppé et
a été retiré → on le **reconstruit** depuis le `generation_config` du
checkpoint chargé (`_primary_eos_id()`, jamais une constante en dur). Légitime :
c'est le token que le forced-seed a réellement tiré à cette position, le
teacher-forcing du validateur retombera dessus. Sans indice sans équivoque
(plusieurs EOS déclarés), on n'invente RIEN — mieux vaut jeter le rollout.

Les trois mécanismes couvrent donc les trois cas :
| vLLM rend | Mécanisme |
|---|---|
| stop-token nommé (`stop_reason`) | ré-append |
| arrêt sans nom (`finish_reason='stop'`) | **reconstruction depuis generation_config** |
| EOS présent (au milieu ou à la fin) | troncature au 1er EOS |
| aucun EOS, cap atteint (`length`) | groupe jeté (garde §5) — correct |

**Filet posé le 2026-08-04** : `DropTracker` (engine.py) émet un
`logger.error("PANNE SILENCIEUSE probable : N/N groupes jetés pour
'termination' et AUCUNE soumission")` dès qu'une cause TECHNIQUE domine les
rejets sur 40 groupes. Les rejets `out_of_zone` (régime normal ~99%) ne
déclenchent jamais l'alarme. Une soumission réussie réarme le compteur.
→ **Au lancement, surveiller cette ligne : elle apparaît dans les ~2 minutes
si on est dans le mauvais cas.**

---

## Étape 2 — Le gaspillage GPU (`ignore_eos`) — MESURER puis décider

**La question** : le mineur continue-t-il à générer APRÈS que le modèle a émis
son token de fin ? Le fix de troncature corrige la *conformité* (on coupe avant
d'envoyer) mais **PAS le gaspillage** : le GPU produit toujours ces tokens, on
les jette ensuite. Sous le tie-break v3, débit = revenu, donc ça coûte.

**Origine** : `ignore_eos=True` a été ajouté le 2026-08-03 pour récupérer
`stop_reason` (l'EOS moteur donne `stop_reason=None`, impossible à restaurer).
Depuis que `truncate_at_first_eos` existe, il est **probablement inutile**.

⚠️ **Ne pas le retirer à l'aveugle** : sans lui on retombe peut-être sur le bug
d'origine (vLLM retire le token d'arrêt → 13 premiers rejets).

**Mesure** (2 min, sur le GPU libre, script déjà écrit) :
```bash
cd /workspace/reliquary-miner-priv && HF_HOME=/workspace/hf PYTHONPATH=. \
  RELIQUARY_VLLM_FORCED_SEED=1 VLLM_USE_DEEP_GEMM=0 \
  VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_FLASHINFER_SAMPLER=0 \
  CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13 \
  /workspace/venv/bin/python ops/probe_eos_behavior.py
```

Il compare les configs sur de vrais prompts code et rend : EOS présent ? en
dernière position ? au milieu ? **tokens gaspillés après le 1er EOS** ?

**⚠️ 2e question que le probe doit trancher — QUEL EOS arrête ?**
Le checkpoint 4B déclare **DEUX** EOS : `248044 <|endoftext|>`
(generation_config) et `248046 <|im_end|>` (tokenizer). Vérifié : l'ensemble
résolu est **identique** côté mineur et côté validateur (même fonction
`resolve_eos_token_ids`, même checkpoint) — donc pas de désalignement là-dessus.
MAIS la reconstruction de l'EOS (`RELIQUARY_RECONSTRUCT_EOS`, **défaut OFF**)
appose `248044` : si l'arrêt venait en réalité de `248046`, on soumettrait un
token JAMAIS généré → le teacher-forcing trouverait autre chose
(`SEED_MISMATCH` / soupçon de falsification), **pire qu'un groupe jeté**.
→ Lire `stop_reason` / `finish_reason` dans la sortie du probe pour identifier
le token, puis activer `RELIQUARY_RECONSTRUCT_EOS=1` **seulement** s'il vaut
bien `248044`.

**3 configurations testées** (la 3e est la candidate idéale) :
1. `ignore_eos=ON + stop_ids` — l'actuelle : correcte mais gaspille
2. `ignore_eos=OFF + stop_ids`
3. **`ignore_eos=OFF, SANS stop_ids`** — arrêt purement naturel, **zéro
   gaspillage par construction**

**Règle de décision — deux critères, les deux obligatoires** :
- **(a) correction** : « passe le validateur » ≈ « contient un EOS ». Si le
  token est retiré, la troncature n'a rien à couper → retour aux
  `bad_termination` des fenêtres 27585/27586. *(C'est le piège : l'arrêt
  naturel SEUL ne suffit pas s'il ne conserve pas le token.)*
- **(b) économie** : « tokens gaspillés après le 1er EOS » ≈ 0.

→ Si la config 3 satisfait (a), **elle gagne** : plus aucun token généré après
la fin. Sinon garder `ignore_eos=ON` + restauration + troncature, et accepter
le gaspillage mesuré.

**Enjeu réel du gaspillage** (au-delà du débit) : un rollout terminé mais
compté « a atteint le cap » fait tomber **les 7 autres du groupe** avec lui
(garde `MAX_TRUNCATED_CODE=0`). Un faux boucleur ne coûte donc pas seulement sa
propre génération — il détruit un groupe entier qui aurait pu être valide.

---

## Étape 3 — Fenêtres hors zone : le validateur comme banc de test rapide

**Pourquoi ça marche (env code, vérifié dans le code upstream)** :
1. `server.py::_preflight` **saute** le contrôle de zone car
   `opencodeinstruct.validator_authoritative_reward = True` → la TERMINAISON
   est testée en premier.
2. `admission.py` : terminaison (l.612) vient **avant** `OUT_OF_ZONE` (l.871).

Donc un groupe hors zone renvoie quand même les erreurs de **génération**.

```bash
RELIQUARY_ZONE_SIGMA_MIN=0.0 bash /workspace/start_miner.sh   # 2-3 fenêtres MAX
```

**Lecture des résultats** :
- `out_of_zone` = **SUCCÈS** — le validateur a franchi la terminaison sans rien
  redire, il ne reproche plus que la zone (normal, on envoie du non payable).
- `bad_termination` encore = le fix est incomplet → NE PAS INSISTER, couper.

**⚠️ Sécurité — vérifié dans `validator/quarantine.py`** :
`out_of_zone` n'est **pas** un rejet « haut risque » → aucun effet.
Mais **`bad_termination` en est un** : 32 rejets haut-risque dans une fenêtre
la mettent en quarantaine d'entraînement (`TRAINING_QUARANTINE_REJECT_SPIKE_MIN`).
Ça ne sanctionne pas notre hotkey (la quarantaine ne décide pas des émissions,
elle protège le modèle du validateur) mais **ça pénalise le subnet**. Donc :
ne jamais lancer cette expérience si l'étape 1 n'est pas verte, et la limiter
à 2-3 fenêtres.

**Après l'expérience** : remettre le filtre (retirer `RELIQUARY_ZONE_SIGMA_MIN`).

---

## Étape 4 — Gate seed-consistency 4B (le dernier maillon jamais testé)

`SEED_MISMATCH` et `GRAIL_FAIL` vivent dans `batcher.py`, sur le chemin
**preuve** — le validateur ne les calcule que pour les candidats qui **gagnent**
l'enchère. Un groupe hors zone ne les déclenchera donc **jamais** : l'étape 3
ne peut pas les valider.

**À ne pas attendre passivement** : la parité forced-seed a été validée à 0.988
sur le **2B**, jamais sur le **4B** — et le validateur tourne
`flash_attention_2` là où nous sommes en `sdpa`.

```bash
GATE_EAGER=0 python ops/validate_vllm_forced_seed_group.py   # planchers 0.80 / 0.75
bash ops/run_gates.sh
```

---

## Étape 5 — Production + suite

1. Config de production : filtre σ à 0.43, `BAKE_BATCH_SIZE=40`,
   `MAX_NEW_TOKENS=2600`, `MAX_TRUNCATED_CODE=0`, CUDA graphs ON.
2. Surveiller le premier verdict **non-`bad_termination`** → clôt le gate G8.
3. Prédicteur de difficulté : **probe DÉDIÉ** (`scripts/difficulty_probe.py`),
   pas de collecte pendant le minage — préférence utilisateur, cf.
   `feedback_separer_probe_et_mineur`. Le dump `RELIQUARY_SAMPLE_DUMP` existe
   dans le code mais reste **ÉTEINT**.

---

## Fixes déjà faits (dev box, non déployés)

| Fix | Fichier | Tests |
|---|---|---|
| Parse `/state` v3 (3 champs) | `protocol/submission.py` | `test_state_v3_parse.py` |
| Pipeline chunké double-bufferé | `miner/engine.py` | `test_bake_chunk_pipeline.py` |
| Restauration du stop-token | `miner/vllm_backend.py` | `test_forced_phase1_includes_eos.py` |
| **Troncature au 1er EOS (vLLM)** | `miner/engine.py` | `test_termination_parity_validator.py` |
| **Prédicat validateur en garde locale** | `miner/engine.py` (×2 sites) | idem |
| Dump échantillons (ÉTEINT) | `miner/engine.py` | `test_sample_dump.py` |

Suite complète : **314 passés / 22 échoués** = les 22 pré-existants (`vllm`
absent, pas de driver CUDA, tests pinnés v2). Zéro régression.
Gates wire : `bash scripts/check_4b_port.sh --tests` → tous verts.
