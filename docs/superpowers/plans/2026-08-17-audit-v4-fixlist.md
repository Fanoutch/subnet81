# Audit multi-agents conformité v4 — 2026-08-17

> Source : workflow 33 agents (7 dimensions × contre-vérification adversariale × synthèse),
> upstream feat/qwen3-base-dapo-v4-profile @ 8c38992 vs branche feat/port-v4-dapo.
> 25 findings confirmés → 18 items dédupliqués. L'item 1 (bloquant) est FIXÉ (commit 736d740).

# Liste de travail finale — port v4 (RELIQUARY_PROTOCOL_VERSION=4)

Dédoublonnage : 6 fusions (profile_id ×2, gate σ ×2, AUCTION_MIN_SCORE ×2, label in_zone ×2, P_STOP ×2, WindowTally ×2). 13 items code + 2 items ops. Règle transverse : **tout fix est gaté `PROTOCOL_VERSION >= 4`, défaut v3 byte-identique.**

## 🟥 Bloquant

**1. GENERATION_PROFILE_ID non gaté → 100% GENERATION_CONTRACT_MISMATCH**
- `reliquary/constants.py:482-488` : défaut `"qwen3-4b-base-dapo-v4" if protocol_version(env) >= 4 else "qwen35-4b-auction-v3"` (utiliser `protocol_version(env)` du module, pas le snapshot, pour que l'appel avec env dict explicite suive cet env). `GENERATION_PROFILE_ID` (l.491) suit.
- Ops : `ops/start_miner_v4_DRAFT.sh` bloc sanity (~l.34) — assert `c.GENERATION_PROFILE_ID == "qwen3-4b-base-dapo-v4"`.
- Test : `tests/test_v4_constants.py` — `test_v4_values` assert v4 ; `test_default_is_v3_byte_identical` assert `"qwen35-4b-auction-v3"`.

## 🟧 Important

**2. Gate σ pré-soumission épinglé 0.43 (prérequis des items 3, 6, 12)**
- `reliquary/miner/engine.py:1093` (`_skip_for_out_of_zone`) : `threshold = _VALIDATOR_STEADY_SIGMA_MIN` → import paresseux `from reliquary.miner.zone import active_thresholds` + `threshold = active_thresholds()[0]` (0.43 en v3, 0.24 en v4). Bloc override `RELIQUARY_ZONE_SIGMA_MIN` l.1094-1102 inchangé ; warning l.1100-1102 formate `threshold`. Docstring l.1063-1077 à jour.
- Test : `tests/unit/test_pre_bake_out_of_zone.py` — cas v4 (monkeypatch PROTOCOL_VERSION=4) : k=1 et k=15 sur M=16 passent ; cas v3 inchangés verts.

**3. AUCTION_MIN_SCORE défaut 0.32 > max théorique v4**
- `reliquary/miner/engine.py:1034-1036` : ajouter `PROTOCOL_VERSION` à l'import constants (bloc l.30-41), puis `AUCTION_MIN_SCORE = float(_os.environ.get("RELIQUARY_AUCTION_MIN_SCORE", "0.0" if PROTOCOL_VERSION >= 4 else "0.32"))`. Commentaire l.1024-1033 : noter calibrage M=8/v3.
- Ops : `export RELIQUARY_AUCTION_MIN_SCORE=0` dans `ops/start_miner_v4_DRAFT.sh` (ceinture).
- Test : `tests/test_auction_score_gate.py:76` reste vert (v3) + nouveau cas v4 : défaut 0.0, un groupe k=15/16 passe.

**4. K_MIN=3/K_MAX=5 en dur — bande v4 = k∈[1,15]**
- `reliquary/miner/engine.py:751-752` : `K_MIN = int(_os.environ.get("RELIQUARY_K_MIN", "1" if PROTOCOL_VERSION >= 4 else "3"))` ; idem K_MAX `"15"/"5"`. **Pas** de dérivation depuis `active_thresholds()` (changerait v3 [3,5]→[2,6]).
- Ops : au lancement v4, retirer `RELIQUARY_K_MIN=2`/`K_MAX=6` de `launch_miner.sh:10-11` (sinon ils écrasent les défauts).
- Test : nouveau cas dans `tests/test_v4_constants.py` ou test _try_select : sous PROTOCOL_VERSION=4, `k_order == range(1,16)` ; sous v3, `range(3,6)`.

**5. P_STOP_LOCAL_MIN=0.01 littéral vs MIN_EOS_PROBABILITY v4=0.001**
- `reliquary/miner/engine.py:971` : ajouter `MIN_EOS_PROBABILITY` à l'import constants l.30, puis `P_STOP_LOCAL_MIN = MIN_EOS_PROBABILITY` (constante déjà gatée dans constants.py:434 ; 0.01 en v3). Optionnel (miroir `terminal_pick_ok` verifier.py:319 upstream) : sous FORCED_SEED_ENFORCE, `bt_ok = in_eos` seul aux sites 4110-4114 et 4560-4564.
- Test : assert sous v4 `engine.P_STOP_LOCAL_MIN == 0.001`, sous v3 `== 0.01` (import module sous env monkeypatché).

**6. Garde de terminaison tout-ou-rien vs budget truncated v4 (1 math / 3 code)**
- Port constants : `reliquary/constants.py` — `MAX_TRUNCATED_PER_SUBMISSION=1`, `MAX_TRUNCATED_PER_SUBMISSION_BY_ENV={"opencodeinstruct": 3}`, `BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION=1`, `max_truncated_for_environment(env)` (parité upstream constants.py:272-301).
- `reliquary/miner/engine.py:3790-3799` (`_pre_bake_entry`) : par génération — `validator_termination_ok` → ok ; sinon si `PROTOCOL_VERSION >= 4` ET zéro EOS dans la complétion ET `len(tokens) >= MAX_NEW_TOKENS_PROTOCOL_CAP` → truncated ; sinon bad. Drop si bad ≠ ∅ OU truncated > budget(env). Commentaire l.3784-3789 à jour. Optionnel : aligner `max_truncated_allowed()`/`too_many_truncated` (l.931-959).
- Test : nouveau `tests/test_v4_truncated_budget.py` — v4 : 1 rollout au cap sans EOS accepté (math), 4e tronqué code = drop, EOS mid-stream = bad ; v3 : tout-ou-rien inchangé.

**7. Révision modèle v4 (906bfd4b…) épinglée nulle part**
- `reliquary/constants.py:408` : `DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Base" if PROTOCOL_VERSION >= 4 else "Qwen/Qwen3.5-2B"` + `DEFAULT_BASE_MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539" if PROTOCOL_VERSION >= 4 else None`.
- `reliquary/cli/main.py` : dans les deux branches de fallback (~l.123-128 « no published checkpoint » et l.134 sortie de retries), si `DEFAULT_BASE_MODEL_REVISION` et `checkpoint == DEFAULT_BASE_MODEL` : `initial_path = snapshot_download(repo_id=…, revision=…, allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS)` — pinne HF, tokenizer ET vLLM (un kwargs `revision` ne couvrirait pas VLLMBackend:209). v3 : `None` → zéro changement.
- Test : `tests/test_v4_constants.py` — DEFAULT_BASE_MODEL/REVISION sous v4 et v3.

**8. Label `in_zone` du SAMPLE_DUMP figé à 0.43 (empoisonne le prédicteur v5)** *(après item 2 : même pattern)*
- `reliquary/miner/engine.py:876` (`dump_group_sample`, dans le try après l.870) : `_sigma_min = active_thresholds()[0]` ; `"in_zone": bool(sigma >= _sigma_min)` + ajouter champ `"sigma_min": _sigma_min` (rend les datasets v3/v4 séparables).
- Test : cas dans le test du dump — sous v4 un σ=0.30 est `in_zone=True` et `sigma_min==0.24` ; sous v3 `in_zone=False`.

**9. difficulty_probe rend les prompts selon le PROTOCOL_VERSION du process (défaut 3)**
- `scripts/difficulty_probe.py` : ① flag `--expect-protocol <int>` au sous-parseur generate (~l.848-860) + sys.exit si `constants.PROTOCOL_VERSION != args.expect_protocol` en tête de chaque stage_generate (~l.302/395/662) ; ② banner `[generate] proto=v{…} raw_prompts={constants.RAW_COMPLETION_PROMPTS}` ; ③ documenter la commande v4 (`RELIQUARY_PROTOCOL_VERSION=4 … --expect-protocol 4 --temperature 1.0` — le défaut T=0.6 l.855 n'est PAS corrigé par l'env var).
- Test : `tests/test_difficulty_probe_bft.py` — nouveau cas : mismatch → SystemExit ; sans flag → comportement inchangé.

**10. RANKING_TIME_BUDGET_S=25 s calibré fenêtre 300 s (v4 = 150 s)**
- `reliquary/miner/engine.py:243-244` : `RANKING_TIME_BUDGET_S = float(_os.environ.get("RELIQUARY_RANKING_BUDGET_S", "12" if PROTOCOL_VERSION >= 4 else "25"))`.
- Ops : `export RELIQUARY_RANKING_BUDGET_S="${RELIQUARY_RANKING_BUDGET_S:-12}"` dans le draft v4.
- Test : assert défaut 12 sous v4 / 25 sous v3.

**11. Prédicteurs v4.x = priors du monde v3 (M=8, T=0.6, thinking, manifest complet) — OPS, zéro code**
- Script de lancement v4 : NE PAS exporter `RELIQUARY_PROMPT_PREDICTOR` / `_2` (supprimer les lignes 31-32 héritées de start_miner_h200.sh) → fallback uniforme (engine.py:497-498). `RELIQUARY_EXPLORE_SLOTS` ne suffit pas (engine.py:537 garde ≥2 slots vedettes). Garder `RELIQUARY_SAMPLE_DUMP` pour collecter des labels v4 → prédicteur v5. Optionnel : WARNING dans `_load_predictor` (engine.py:491) si PV≥4 sans marqueur de provenance v4.
- Verrou : checklist dans le draft v4 (commentaire) + l'assert sanity n'exporte pas ces vars.

## 🟨 Mineur

**12. WindowTally figé M=8/0.43/rank8** *(après item 2)*
- `reliquary/miner/engine.py:292` `len(v) != M_ROLLOUTS` ; l.299-300 `/ float(M_ROLLOUTS)` ; l.301 `sd >= active_thresholds()[0]` ; `_build` l.392-404 : `at(B_BATCH)` / clé `f"rank{B_BATCH}"` (B_BATCH à l'import l.30-42 ; v3 : 8 → libellés inchangés).
- Test : `tests/test_window_ranking.py:130` + cas v4 (vecteurs de 16, seuil 0.24, clé "rank16").

**13. max_model_len=16384 en dur (2× le cap v4)**
- `reliquary/cli/main.py:214` : `max_model_len=int(os.environ.get("RELIQUARY_VLLM_MAX_MODEL_LEN", str(MAX_NEW_TOKENS_PROTOCOL_CAP + 1024 if PROTOCOL_VERSION >= 4 else 16384)))` → 9216 en v4. Ne PAS toucher `vllm_backend.py:1026`.
- Test : unit sur la valeur dérivée sous v4/v3 (ou assert dans le sanity ops).

**14. FS_GRAPH : paliers ≤32 lignes → eager quasi permanent sous v4**
- `reliquary/miner/vllm_forced_seed.py:165` : `BUCKETS = (1,2,4,8,16,24,32,48,64)` (64 couvre SIZE=4×16 ; 128 seulement si mesuré). Chemin déjà gaté par `RELIQUARY_FS_GRAPH=1` → v3 inerte. Obligatoire avant activation : re-mesurer la mémoire des pools (≈78 Mo à 64 + pool CUDA graph — cf. OOM 08-15).
- Test : repasser `ops/test_fs_graph_gpu.py` en adaptant l'assert l.93-94 (n=60 devient capturé ; tester eager au-delà du dernier palier).

**15. Hot-swap self-gate : plancher 0.80 toutes-positions non recalibré full-support**
- `reliquary/miner/engine.py:2862-2905` : signature `floor: float | None = None` ; si None → `FORCED_SEED_ROLLOUT_FLOOR if PROTOCOL_VERSION >= 4 else 0.80`. Sous v4 : ne compter que les positions stochastiques (`probs.max() < FORCED_SEED_STOCHASTIC_MAXPROB`), abstention=PASS si `n_stoch < FORCED_SEED_ROLLOUT_MIN_STOCH` (20). v3 : boucle byte-identique.
- Ops : avant tout `RELIQUARY_HOT_SWAP=1` en v4, re-mesurer le taux honnête vLLM↔HF sous T=1.0 (et déboguer le gel du 08-15 d'abord).
- Test : unit v4 — positions déterministes exclues du taux ; abstention sous 20 stoch ; v3 inchangé.

**16. Commentaire MIN_LOCAL_Q10 destructeur en v4 (marge 0.05 vs seuil 0.0002)**
- `reliquary/miner/engine.py:960-963` : remplacer par le commentaire version-aware (v3 q10 0.025/médiane 0.30, v4 q10 0.0002/médiane 0.05 ; marges sûres v4 : 0.0005/0.08 ; « NEVER set the v3 values under PV=4 ») pointant vers `constants.SAMPLING_LOW_Q10_MAX`/`SAMPLING_MEDIAN_LOW_MAX`. Défaut 0=off inchangé.
- Test : aucun (commentaire) — verrou = grep dans le sanity ops que ces env vars ne sont pas exportées en v4 avec des valeurs v3.

**17. Tunings fenêtre 150 s (sprint/bake) — OPS, zéro code**
- `ops/start_miner_v4_DRAFT.sh` en-tête : « NE PAS reprendre SPRINT_SIZE=2/SPRINT_MAX_WAIT_S=90 de start_miner_h200.sh (90 s = 60% de 150 s). Re-bancher jour J : `RELIQUARY_SPRINT_SIZE` (candidat 1-2 ; à M=16, SIZE=4 = 64 seqs en vol), `RELIQUARY_SPRINT_MAX_WAIT_S` (≲10-15 s), `BAKE_BATCH_SIZE` ». Défauts engine.py:1356/3160 intouchés. Ignorer les constantes v4 validator-side (MAX_RANKED_PROOF_ATTEMPTS, CHECKPOINT_PUBLISH_INTERVAL=16, OVERLONG_*, FORENSIC_SAMPLE, DO_SAMPLE_PROTO) — vérifié consommateur par consommateur.

**18. selector.py n=8/k∈[2,6] — code mort, à corriger SEULEMENT si re-câblé**
- `reliquary/miner/selector.py:15,79` : dérivation paresseuse `_protocol_band()` depuis M_ROLLOUTS + active_thresholds (donne exactement k∈[2,6] à n=8/0.43 = byte-identique v3 ; n=16, k∈[1,15] en v4), bande stockée dans `Selector.__init__`. Ne pas toucher les défauts de signature (figés par `tests/test_score.py`).
- Test : si re-câblé, cas v4 dans test_score.py via monkeypatch.

---
**Ordre d'exécution suggéré** : 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 10 → 12 (edits engine/constants, un seul passage de `tests/test_v4_constants.py` + suite v3 complète en fin pour prouver la byte-identité) ; 9, 13-16 ensuite ; 11 et 17 = checklist ops du draft v4 ; 18 = dette documentée.