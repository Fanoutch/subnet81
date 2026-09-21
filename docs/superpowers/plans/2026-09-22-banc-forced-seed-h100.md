# Banc forced-seed sur H100 louée — plan (22/09/2026)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objectif :** trouver où partent les ~3,8 ms par pas que coûte le forced-seed
sur H100 (12,15 contre 8,36 ms à 160 séquences) et les réduire, avec les
mêmes tokens que le validateur.

**Architecture :** une H100 louée (identique à la prod, sans wallet) reçoit
la même pile logicielle que la box de prod. On y fait (1) un banc de
décomposition qui isole chaque source de coût, (2) un profilage GPU/CPU d'un
pas de décodage, puis (3) on code et mesure les correctifs dictés par le
profil, chacun validé par un contrôle de conformité avant toute mise en prod.

**Pile :** torch 2.11.0+cu130 · vLLM 0.24.0 · transformers 5.9.0 · Python 3.12 ·
driver 580.126.20 · Qwen3-4B-Base (checkpoint ReliquaryForge/qwen3-4b-base-dapo-v4).

**Point de départ (mesuré le 21/09 sur la box de prod, carte vide, 3 rép.) :**

| 160 séquences × 512 tokens | ms/pas | tok/s |
|---|---|---|
| sans forced-seed (T=1, processeur enregistré mais inactif) | 8,36 | ~19 100 |
| forced-seed actuel | 12,15 | ~13 100 |
| forced-seed chemin rapide (`fast`, commit 01f601d/5872931) | 11,98 | ~13 300 |

Le chemin rapide (u par préfixe SHA-256, masque creux, plus de copie) n'a
gagné que 0,17 ms : le coût N'EST PAS dans les passes de masque ni dans la
boucle de hachage. Trois pistes restent ouvertes, à trancher par la mesure :
1. **Mécanisme vLLM** : un processeur personnalisé impose
   `logitsprocs_need_output_token_ids=True` (gpu_model_runner.py:685) et écarte
   le model runner V2 (config/vllm.py:2070). Si les tokens doivent revenir au
   CPU avant le pas suivant, le recouvrement CPU/GPU est cassé.
2. **Notre calcul GPU** : softmax + somme + division + `cumsum` sur
   [160, 151 936] en fp32 à chaque pas (un `cumsum` sur 152 k colonnes peut
   être lent) — `FS_GRAPH` ne couvre que n ≤ 32 lignes, donc rien à 160.
3. **Glouton contre échantillonnage** : le mode « libre » échantillonnait à
   T=1 ; le forcé passe en glouton. Il faut une référence gloutonne sans
   processeur pour comparer à armes égales.

## Contraintes globales

- ⛔ **Aucun wallet ni coldkey sur la box louée.** Banc hors ligne uniquement.
- ⛔ Rien ne part en prod sans : taux de tokens identiques ≥ celui du chemin
  actuel contre lui-même (bruit de fond), gate `ops/validate_vllm_forced_seed_group.py`
  PASS (planchers 0,80 groupe / 0,75 rollout), puis go explicite de
  l'utilisateur pour le restart.
- ⚠️ Les empreintes globales de tokens NE SONT PAS un critère : le chemin
  actuel lui-même change d'empreinte d'une répétition à l'autre (bascules
  numériques près d'une frontière de CDF, 21/09). Critère = taux de tokens
  identiques par position et part de séquences identiques.
- Même carte que la prod : **H100 80GB HBM3 (SXM)**. Si le banc de
  référence (Tâche 1) s'écarte de plus de 5 % des 8,36 / 12,15 ms, rendre la
  box et en louer une autre (règle « banc de perf avant de garder une box »).
- Protocole V1 : T=1, top_k=0, top_p=1, 16 rollouts, max_new_tokens 8192.
- Code : branche `h100/v1` du dépôt, worktree `.worktrees/miner-priv-h100`.
  Un commit par tâche, poussé.
- Budget visé : 4 à 6 h de location.

## Fichiers

| fichier | rôle |
|---|---|
| `ops/bench_fs_h100.py` (modifier) | banc de décomposition : modes, tailles, longueurs, taux d'identité |
| `reliquary/miner/vllm_forced_seed.py` (modifier) | mode `noop` par requête (banc uniquement) ; puis correctifs |
| `ops/profile_fs_step.py` (créer) | profilage vLLM (torch profiler) de N pas, par mode |
| `ops/setup_bench_box.sh` (créer) | installation de la box louée (pile identique, modèle, code) |
| `tests/test_fs_fast_0921.py` (modifier) | tests du mode `noop` et des correctifs |
| `docs/superpowers/plans/2026-09-22-banc-forced-seed-h100.md` | ce plan + résultats consignés au fil de l'eau |

---

## Phase A — ce soir / demain matin, SANS GPU (préparation)

### Tâche A1 : mode `noop` par requête (processeur présent, rempli, mais inactif)

Sert à mesurer le coût du **mécanisme** (suivi des requêtes, `update_state`,
tokens rapatriés) sans notre calcul. Même principe que `fast` : champ posé
dans les `extra_args` par le banc seulement ; le mineur ne le pose jamais.
Le drapeau d'environnement `RELIQUARY_FS_NOOP` existe déjà mais ne permet pas
de comparer dans le même moteur.

**Fichiers :** modifier `reliquary/miner/vllm_forced_seed.py` (`ForcedRowsState`),
tester dans `tests/test_fs_fast_0921.py`.

**Interfaces :** produit `ForcedRowsState._noop_now: bool` (vrai si toutes les
requêtes du lot portent `"noop": True`).

- [ ] **Étape 1 : écrire le test qui échoue**

```python
def test_noop_par_requete_laisse_les_logits_intacts(monkeypatch):
    monkeypatch.delenv("RELIQUARY_FS_FAST", raising=False)
    req = {i: ({**_fs(5, i), "noop": True}, [0] * i) for i in range(4)}
    st = ForcedRowsState(); st.rebuild(req, device="cpu")
    assert st._noop_now is True
    logits = torch.randn(4, 900)
    assert torch.equal(st.apply(logits.clone()), logits)


def test_noop_mixte_ne_s_active_pas(monkeypatch):
    req = {0: ({**_fs(5, 0), "noop": True}, []), 1: (_fs(5, 1), [])}
    st = ForcedRowsState(); st.rebuild(req, device="cpu")
    assert st._noop_now is False
```

- [ ] **Étape 2 : lancer, vérifier l'échec**

Run : `cd .worktrees/miner-priv-h100 && python3 -m pytest -q tests/test_fs_fast_0921.py -k noop`
Attendu : FAIL, `AttributeError: 'ForcedRowsState' object has no attribute '_noop_now'`

- [ ] **Étape 3 : implémenter**

Dans `ForcedRowsState.__init__` ajouter `self._noop_now = False`. Dans
`rebuild`, juste après le calcul de `self._fast_now` :

```python
        self._noop_now = bool(n) and all(fs.get("noop") for fs, _out in self._slots)
```

Dans `apply`, juste après `if _FS_NOOP: return logits` :

```python
        if self._noop_now:
            return logits
```

- [ ] **Étape 4 : tests verts + suite forced-seed inchangée**

Run : `python3 -m pytest -q tests/test_fs_fast_0921.py` → tout PASS.
Run : `ls tests | grep -iE "forced|fs_|batched|device_resident|vllm" | sed 's#^#tests/#' | xargs python3 -m pytest -q -p no:cacheprovider`
Attendu : exactement les 10 échecs préexistants connus (v2_port ×2,
vllm_forced_phase1_multi ×5, …), rien de plus.

- [ ] **Étape 5 : commit**

```bash
git add reliquary/miner/vllm_forced_seed.py tests/test_fs_fast_0921.py
git commit -m "forced-seed : mode noop par requête pour le banc (inerte en prod)"
```

### Tâche A2 : banc de décomposition complet

Étend `ops/bench_fs_h100.py` : plus de modes, plusieurs tailles et longueurs,
taux de tokens identiques à la place des empreintes.

**Modes** (un moteur par « famille », les modes d'une famille alternent) :

| famille (moteur) | modes | ce que ça isole |
|---|---|---|
| `nu` : `VLLMBackend(forced_seed=False)` | `nu_greedy` (T=0), `nu_t1` (T=1) | vLLM sans aucun processeur |
| `fs` : `VLLMBackend(forced_seed=True)` | `libre_greedy`, `libre_t1`, `noop`, `fs`, `fast` | coût d'enregistrement, du mécanisme rempli, de notre calcul |

Différences à lire :
- `libre_greedy − nu_greedy` = coût d'un processeur **enregistré** (vide) ;
- `noop − libre_greedy` = coût du **mécanisme rempli** (requêtes suivies, `update_state`) ;
- `fs − noop` = coût de **notre calcul** ;
- `fast − fs` = gain du chemin rapide.

**Grille :** séquences ∈ {32, 64, 160, 256} (2, 4, 10, 16 prompts × 16),
longueur ∈ {512, 2048}. Longueur 2048 : la part fixe par pas se dilue quand
l'attention grossit — vérifie si le surcoût relatif baisse en fin de génération.

**Taux d'identité :** pour chaque (taille, longueur), garder les tokens du
1er passage `fs` comme référence ; pour chaque autre passage `fs`/`fast`,
calculer `seq_identiques` (part des séquences identiques) et `pos_identiques`
(part des positions identiques avant la 1re divergence, moyennée).

- [ ] **Étape 1 : écrire le test qui échoue** (fonction pure, CPU)

Créer `tests/test_bench_fs_identite.py` :

```python
from ops.bench_fs_h100 import identity_rates


def test_taux_identite():
    ref = [[1, 2, 3, 4], [5, 6, 7, 8]]
    other = [[1, 2, 3, 4], [5, 6, 9, 9]]
    seq, pos = identity_rates(ref, other)
    assert seq == 0.5
    assert pos == (1.0 + 0.5) / 2          # 2e séquence : diverge à la position 2


def test_taux_identite_parfait():
    ref = [[1, 2], [3, 4]]
    assert identity_rates(ref, [r[:] for r in ref]) == (1.0, 1.0)
```

- [ ] **Étape 2 : lancer, vérifier l'échec**

Run : `python3 -m pytest -q tests/test_bench_fs_identite.py`
Attendu : FAIL, `ImportError: cannot import name 'identity_rates'` (ajouter
`ops/__init__.py` vide si l'import de `ops` échoue).

- [ ] **Étape 3 : implémenter `identity_rates` et la grille**

Dans `ops/bench_fs_h100.py` :

```python
def identity_rates(ref: list[list[int]], other: list[list[int]]) -> tuple[float, float]:
    """(part des séquences identiques, part moyenne des positions identiques
    avant la 1re divergence). Critère de conformité entre deux passages : le
    chemin actuel contre lui-même donne le bruit de fond à ne pas dépasser."""
    same_seq, pos_frac = 0, 0.0
    for a, b in zip(ref, other):
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        same_seq += int(a == b)
        pos_frac += k / max(1, len(a))
    n = max(1, len(ref))
    return same_seq / n, pos_frac / n
```

Modifier `_run` pour renvoyer aussi la liste des tokens
(`[list(o.outputs[0].token_ids) for o in outs]`), et accepter les modes
`nu_greedy`, `nu_t1`, `libre_greedy`, `libre_t1`, `noop`, `fs`, `fast`
(`noop` = `extra_args` forced-seed + `"noop": True` ; `*_greedy` = T=0 sans
`extra_args` ; `*_t1` = T=1 sans `extra_args`). Ajouter les variables
`BENCH_FAMILY` (`nu` ou `fs`, choisit `forced_seed=` du `VLLMBackend`),
`BENCH_PROMPTS` (liste), `BENCH_LENS` (liste), `BENCH_MODES` (liste),
`BENCH_REPS` (défaut 3, plus de budget souple : la box louée n'a pas de trou
à respecter). Écrire une ligne JSON par passage avec `seqs, len, mode, rep,
ms_par_pas, tok_s, seq_identiques, pos_identiques` dans `BENCH_OUT`.

- [ ] **Étape 4 : tests verts, compilation**

Run : `python3 -m pytest -q tests/test_bench_fs_identite.py && python3 -m py_compile ops/bench_fs_h100.py`

- [ ] **Étape 5 : commit**

```bash
git add ops/bench_fs_h100.py ops/__init__.py tests/test_bench_fs_identite.py
git commit -m "banc forced-seed : décomposition complète + taux de tokens identiques"
```

### Tâche A3 : script de profilage d'un pas

**Fichiers :** créer `ops/profile_fs_step.py`.

Utilise le profileur intégré de vLLM : variable `VLLM_TORCH_PROFILER_DIR`
posée avant la construction du moteur, puis `llm.start_profile()` /
`llm.stop_profile()` autour d'une génération courte (160 séquences × 64
tokens, après chauffe). Un fichier de trace par mode (`libre_greedy`,
`noop`, `fs`, `fast`), lisible dans Perfetto / chrome://tracing. En sortie
texte : les 25 noyaux GPU les plus coûteux par pas, le temps CPU dans
`VLLMForcedSeedBatchedLogitsProcessor.apply` / `update_state`, et les appels
de synchronisation (`cudaStreamSynchronize`, `cudaMemcpy*` D2H) par pas.

- [ ] **Étape 1 : écrire le script**

```python
"""Profilage d'un pas de décodage par mode (vLLM torch profiler).
Usage : VLLM_TORCH_PROFILER_DIR=/workspace/prof BENCH_MODEL=... \
        PROFILE_MODES=libre_greedy,noop,fs,fast python ops/profile_fs_step.py"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from bench_fs_h100 import _prompts, _run   # mêmes requêtes que le banc


def main() -> int:
    from transformers import AutoTokenizer
    from reliquary.cli.main import vllm_max_model_len
    from reliquary.miner.vllm_backend import VLLMBackend
    model = os.environ["BENCH_MODEL"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    be = VLLMBackend(model_path=model, tokenizer_path="Qwen/Qwen3-4B-Base",
                     gpu_id=0, gpu_memory_utilization=0.45,
                     max_model_len=vllm_max_model_len(), dtype="bfloat16",
                     forced_seed=True)
    be._ensure_loaded()
    llm, prompts, rnd = be._llm, _prompts(tok, 10), "5a" * 32
    for mode in os.environ.get("PROFILE_MODES", "libre_greedy,noop,fs,fast").split(","):
        _run(llm, prompts, m=16, length=64, mode=mode, rnd=rnd)       # chauffe
        llm.start_profile()
        _run(llm, prompts, m=16, length=64, mode=mode, rnd=rnd)
        llm.stop_profile()
        print(f"[profil] {mode} : trace écrite dans {os.environ['VLLM_TORCH_PROFILER_DIR']}",
              flush=True)
        time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Étape 2 : compilation** — `python3 -m py_compile ops/profile_fs_step.py`
- [ ] **Étape 3 : commit** — `git add ops/profile_fs_step.py && git commit -m "profilage forced-seed par mode"`

### Tâche A4 : script d'installation de la box louée

**Fichiers :** créer `ops/setup_bench_box.sh`. Source des versions :
`data_backups/box_2026-09-18_coupure/reconstruction/pip_freeze_venv.txt`
(venv de prod : torch 2.11/cu130, vLLM 0.24.0) et `gpu.txt` (driver).
Recette de prod : `ops/RECONSTRUCTION_BOX.md` — on n'en prend QUE les étapes
venv + modèle + code (pas de wallet, pas de réplique, pas de miroir parquet :
le banc ne lit aucun dataset).

- [ ] **Étape 1 : écrire le script** (idempotent, s'arrête à la 1re erreur)

```bash
#!/bin/bash
# Installation d'une H100 louée pour le banc forced-seed. AUCUN wallet.
set -euo pipefail
W=/workspace; mkdir -p $W && cd $W
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3.12 -m venv $W/venv
$W/venv/bin/pip install -q --upgrade pip
$W/venv/bin/pip install -q -r $W/pip_freeze_venv.txt   # copié depuis la dev box
export HF_HOME=$W/hf
$W/venv/bin/python -c "from huggingface_hub import snapshot_download as s; \
print(s('ReliquaryForge/qwen3-4b-base-dapo-v4')); print(s('Qwen/Qwen3-4B-Base'))"
git clone -b h100/v1 https://github.com/Fanoutch/subnet81.git $W/subnet81
ln -sfn $W/subnet81/.worktrees/miner-priv-h100 $W/reliquary-miner-priv || true
echo "OK — copier l'environ de prod (etat/env_live_*.txt) puis lancer le banc de référence"
```

⚠️ Vérifier au moment du clone le chemin réel du code mineur dans le dépôt
(le worktree n'existe pas dans un clone : utiliser `git worktree add` ou
cloner la branche dans `$W/reliquary-miner-priv`). Le banc doit tourner avec
l'environ de prod : `data_backups/box_2026-09-21_soir/env_live_20260921_soir.txt`
(variables `RELIQUARY_*`) + les variables vLLM du launcher
(`VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`VLLM_DEEP_GEMM_WARMUP=skip`, `RELIQUARY_FS_GRAPH=1`,
`RELIQUARY_VLLM_CUDA_GRAPHS=1`, `RELIQUARY_VLLM_MAX_NUM_SEQS=256`,
`RELIQUARY_VLLM_GPU_FRACTION=0.45`, `RELIQUARY_PROTOCOL_VERSION=6`,
`RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reliquary-v1`).

- [ ] **Étape 2 : `bash -n ops/setup_bench_box.sh`**
- [ ] **Étape 3 : commit** — `git add ops/setup_bench_box.sh && git commit -m "installation box de banc (sans wallet)"`

---

## Phase B — sur la H100 louée

### Tâche B1 : installation + banc de référence (porte d'entrée)

- [ ] Louer une **H100 80GB HBM3 SXM**. Copier `pip_freeze_venv.txt` et
  `env_live_20260921_soir.txt`, lancer `ops/setup_bench_box.sh`.
- [ ] Banc de référence, famille `fs`, 160 séquences × 512, modes
  `libre_t1,fs,fast`, 3 rép. **Porte :** `libre_t1` à 8,36 ms ± 5 % et `fs`
  à 12,15 ms ± 5 %. Sinon rendre la box.
- [ ] Consigner les chiffres dans la section « Résultats » de ce plan.

### Tâche B2 : décomposition (le cœur de la journée)

- [ ] Famille `nu` : modes `nu_greedy,nu_t1`, séquences 32/64/160/256, longueurs 512/2048.
- [ ] Famille `fs` : modes `libre_greedy,libre_t1,noop,fs,fast`, même grille.
- [ ] Tableau des écarts par taille : enregistrement (`libre_greedy − nu_greedy`),
  mécanisme (`noop − libre_greedy`), notre calcul (`fs − noop`), gain rapide
  (`fast − fs`). Taux d'identité `fs` contre `fs` (bruit de fond) et
  `fast` contre `fs`.
- [ ] **Décision** : la plus grosse des trois parts désigne la phase C à
  mener en premier. Consigner.

### Tâche B3 : profilage

- [ ] `ops/profile_fs_step.py` sur `libre_greedy,noop,fs,fast` (160 séq.).
- [ ] Extraire : noyaux GPU dominants par pas, temps CPU du processeur, synchronisations par pas, trous GPU entre deux pas.
- [ ] Consigner la chronologie d'un pas `fs` contre `libre_greedy`.

---

## Phase C — correctifs, dans l'ordre dicté par B2/B3

Chaque correctif : test CPU qui échoue → implémentation → tests verts →
banc sur la box louée (ms/pas + taux d'identité ≥ bruit de fond) → gate
`ops/validate_vllm_forced_seed_group.py` PASS → commit → consigner. Aucun ne
part en prod sans go.

### C1 (si le MÉCANISME domine) : ne plus dépendre des tokens rapatriés

Hypothèse : `logitsprocs_need_output_token_ids=True` force l'attente des
tokens côté CPU à chaque pas. Piste : dériver la position `t` d'un compteur
de pas tenu par le processeur (incrémenté à chaque `apply` pour chaque
requête active, remis à la bonne valeur à chaque `update_state` d'ajout)
au lieu de `len(output_tok_ids)`, puis vérifier si vLLM accepte alors
`logitsprocs_need_output_token_ids=False` (lecture de
`vllm/v1/worker/gpu_model_runner.py` autour des lignes 655-690 et de
l'ordonnancement asynchrone, lignes 506 et 698). ⚠️ Risque : un décalage
d'un pas = tokens faux = rejet massif. Tests obligatoires sur la
préemption / le prefill par morceaux ; gate de parité avant tout.

### C2 (si NOTRE CALCUL domine) : graphe CUDA à 160/256 lignes

`_FsGraphPass.BUCKETS` s'arrête à 32 : étendre à 64/160/256 via
`RELIQUARY_FS_GRAPH_BUCKETS` pour le chemin rapide (capturer
`fast_pick_t1` + l'écriture creuse). Mesurer la VRAM ajoutée par palier
(historique : 0,3-0,5 Go/palier, OOM des preuves le 15/08 — la box de prod a
~22 Go libres). Parité : mêmes noyaux dans le même ordre → identité attendue
au niveau du bruit de fond.

### C3 (si le `cumsum` domine le profil) : noyau fusionné

Un noyau Triton par ligne : max, somme des exponentielles, puis recherche du
premier indice où la CDF dépasse `u` par blocs (somme par bloc → bloc cible →
balayage dans le bloc). Deux passes sur les logits au lieu de ~7. ⚠️ L'ordre
de sommation change au niveau de l'ULP : les bascules possibles sont du même
ordre que le bruit vLLM contre HF déjà toléré — le critère est le taux
d'identité (≥ bruit de fond) ET la gate de parité.

### C4 (si la boucle CPU ressort encore) : `u` pré-calculés par bloc

Calculer les `u` des 64 positions suivantes d'une séquence d'un coup (ou
dans un fil séparé) pour sortir le hachage du chemin critique du pas.

---

## Phase D — mise en production (après go)

- [ ] Correctif retenu derrière une variable (inerte par défaut), commit poussé.
- [ ] Copie sur la box de prod, restart au trou 503 (go utilisateur).
- [ ] Vigie : `seed_mismatch` / `token_tampered` = ZÉRO toléré ; groupe 1
  prêt (réf ~7,4 s) et groupe 10 (réf 15-18 s) ; tirs avant 18 s.
- [ ] Repli : variable à 0 + restart.

## Résultats (à remplir pendant la journée)

| tâche | mesure | valeur |
|---|---|---|
| B1 | libre_t1 / fs / fast (160×512) | |
| B2 | enregistrement / mécanisme / calcul / gain rapide | |
| B2 | identité fs↔fs · fast↔fs | |
| B3 | noyaux dominants · synchros par pas | |
| C* | correctif retenu, gain ms/pas, identité, gate | |
