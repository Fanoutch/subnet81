# Ré-alignement miner v7 / BFT / thinking (cot-2b) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (ou subagent-driven-development). Steps en checkbox (`- [ ]`).
> **NOTE:** `reliquary-miner-priv` n'est PAS git → « commit » = **checkpoint de revue**. Tests : `cd /root/subnet81/reliquary-miner-priv && PYTHONPATH=. python3 -m pytest <chemin> -v`.

**Goal :** aligner le mineur sur le protocole **v7** du subnet (cot-2b) — modèle **Qwen3.5-2B thinking**, **Budget-Forced Termination (BFT)**, sampler **T=0.6/top_p=0.95/top_k=20** — pour qu'il ne soit plus hard-rejeté et génère comme le validateur. La **même logique BFT** sert à re-générer des données prédicteur valides (via la probe HF).

**Architecture :** on PORTE verbatim les pièces déterministes du validateur (constantes, helpers `</think>`/FORCE, `_bft_assemble_rollouts`, schema). La génération BFT est **HF** (`model.generate` 2-phases) — indépendante de vLLM (gelé, à régler via bump de version en session séparée). On valide sur le **vrai Qwen3.5-2B via la probe HF** (vLLM ne charge pas encore le modèle).

**Tech Stack :** Python 3.12, transformers 5.x, torch, pydantic v2, pytest. Génération = HF. Pas de vLLM touché.

## Global Constraints

- **Source de vérité = validateur d9471f2** (`/root/subnet81/reliquary`). Copier les blocs cités **verbatim** (parité GRAIL/`validate_force_span` byte-exact).
- **NE PAS toucher `reliquary/miner/vllm_backend.py`** (gelé — vLLM ne charge pas Qwen3.5, réglé plus tard par version).
- **Modèle courant** = `Qwen/Qwen3.5-2B` (base) ; checkpoint publié `ReliquaryForge/qwen3.5-2b-reliquary` @ `6a8c5637b52b85a63973f74be25361169e222aec` — **même archi hybride** → **HF charge** (`load_text_generation_model` → `AutoModelForImageTextToText`), vLLM non.
- **BFT ne s'applique QU'À `openmathinstruct`** (`bft_applicable = BFT_ENABLED and (env is None or env=="openmathinstruct")`). Le code (opencode) garde la génération mono-phase.
- Validation runtime = **via la probe HF sur le 2B** (Tasks 6). Le câblage moteur miner (Task 7) est GPU-gaté (vLLM absent).

---

### Task 1 : Constantes v7

**Files:** Modify `reliquary/constants.py` · Test `tests/test_v7_constants.py`

- [ ] **Step 1 : test**
```python
# tests/test_v7_constants.py
import reliquary.constants as c
def test_v7_values():
    assert c.GRAIL_PROOF_VERSION == "v7"
    assert c.MAX_NEW_TOKENS_PROTOCOL_CAP == 32768
    assert c.BFT_ENABLED is True
    assert c.BFT_THINKING_BUDGET == 2048 and c.BFT_ANSWER_BUDGET == 512
    assert c.BFT_FORCE_TEMPLATE == "</think>\n\nFinal Answer: \\boxed{"
    assert c.T_PROTO == 0.6 and c.TOP_P_PROTO == 0.95 and c.TOP_K_PROTO == 20
    assert c.DEFAULT_BASE_MODEL == "Qwen/Qwen3.5-2B"
    assert c.TOKEN_AUTH_THRESHOLD == 1e-8
```
- [ ] **Step 2 : run → FAIL.**
- [ ] **Step 3 : éditer `reliquary/constants.py`** (valeurs verbatim du validateur constants.py) :
  - `GRAIL_PROOF_VERSION = "v7"` (était v6)
  - `MAX_NEW_TOKENS_PROTOCOL_CAP = 32768` (était 8192)
  - ajouter : `BFT_ENABLED = True`, `BFT_THINKING_BUDGET = 2048`, `BFT_ANSWER_BUDGET = 512`, `BFT_FORCE_TEMPLATE = "</think>\n\nFinal Answer: \\boxed{"`
  - `T_PROTO = 0.6`, `TOP_P_PROTO = 0.95`, `TOP_K_PROTO = 20` (étaient 0.9/1.0/0)
  - `DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"` (était 4B)
  - `TOKEN_AUTH_THRESHOLD = 1e-8` (était 1e-10)
  - (optionnel, cohérence shaping) `SHAPE_PENALTY = 0.5`, `SHAPE_LEN_FRAC = 0.5`
- [ ] **Step 4 : run → PASS.** ⚠️ note : `SIGMA_MIN` reste **0.33** dans constants (bootstrap) — inchangé, `ZONE_THRESHOLD_STEADY`=0.43 gère la zone (cf. opencode). Ne pas y toucher.
- [ ] **Step 5 : checkpoint.**

---

### Task 2 : enable_thinking = True

**Files:** Modify `reliquary/protocol/tokens.py` · Test `tests/test_enable_thinking.py`

- [ ] **Step 1 : test** — vérifie que le kwarg passé au chat_template est True.
```python
# tests/test_enable_thinking.py
from reliquary.protocol.tokens import encode_prompt

class _Tok:
    chat_template = "…{% if enable_thinking %}…"
    def apply_chat_template(self, messages, **kw):
        _Tok.seen = kw
        return [1, 2, 3]
    def encode(self, *a, **k): return [9]

def test_thinking_enabled():
    encode_prompt(_Tok(), "hi")
    assert _Tok.seen.get("enable_thinking") is True
```
- [ ] **Step 2 : run → FAIL** (miner met False).
- [ ] **Step 3 :** dans `reliquary/protocol/tokens.py`, remplacer `kwargs["enable_thinking"] = False` par `True` et mettre le commentaire du validateur (tokens.py:62-64) : « Enable thinking: the raised token cap gives the CoT room to close </think> and emit \boxed{} before truncating. »
- [ ] **Step 4 : run → PASS.** Step 5 checkpoint.

---

### Task 3 : Helpers BFT (modeling)

**Files:** Modify `reliquary/shared/modeling.py` · Test `tests/test_bft_helpers.py`

**Produces:** `think_close_token_ids(tokenizer)`, `force_close_token_ids(tokenizer)`, `has_think_close(tokens, think_close_ids)`. (`first_eos_index` existe déjà côté miner.)

- [ ] **Step 1 : test** (avec un tokenizer factice).
```python
# tests/test_bft_helpers.py
from reliquary.shared.modeling import (
    think_close_token_ids, force_close_token_ids, has_think_close)

class _Tok:
    def convert_tokens_to_ids(self, t): return 999 if t == "</think>" else -1
    def encode(self, s, add_special_tokens=False): return [40, 41]  # tail ids

def test_think_close_atomic():
    assert think_close_token_ids(_Tok()) == [999]

def test_force_close_is_close_plus_tail():
    assert force_close_token_ids(_Tok()) == [999, 40, 41]

def test_has_think_close():
    assert has_think_close([1, 999, 2], {999}) is True
    assert has_think_close([1, 2], {999}) is False
```
- [ ] **Step 2 : run → FAIL** (helpers absents côté miner — vérifier `grep -c think_close_token_ids reliquary/shared/modeling.py` = 0).
- [ ] **Step 3 :** **coller VERBATIM** dans `reliquary/shared/modeling.py` les fonctions `think_close_token_ids`, `force_close_token_ids`, `has_think_close` depuis le validateur `reliquary/shared/modeling.py:154-177` (elles importent `BFT_FORCE_TEMPLATE` de constants — présent via Task 1). Confirmer que `first_eos_index` existe déjà (sinon copier `:147-152`).
- [ ] **Step 4 : run → PASS.** Step 5 checkpoint.

---

### Task 4 : Schema v7 (RolloutMetadata + proof_version)

**Files:** Modify `reliquary/protocol/submission.py` · Test `tests/test_v7_schema.py`

- [ ] **Step 1 : test**
```python
# tests/test_v7_schema.py
from reliquary.protocol.submission import RolloutMetadata, CommitModel

def test_rollout_metadata_bft_fields():
    m = RolloutMetadata(prompt_length=1, completion_length=2, success=True,
                        total_reward=0.0, advantage=0.0, token_logprobs=[0.0],
                        forced=True, force_span=[10, 13])
    assert m.forced is True and m.force_span == [10, 13] and m.truncated is False

def test_commit_requires_v7():
    import pydantic, pytest
    # a v6 proof_version must now be rejected
    with pytest.raises(pydantic.ValidationError):
        CommitModel.model_validate({"tokens":[0]*40,"commitments":[],
            "proof_version":"v6","model":{"name":"x","layer_index":-1},
            "signature":"ab","beacon":{"randomness":"r"},
            "rollout":{"prompt_length":1,"completion_length":1,"success":True,
                       "total_reward":0.0,"advantage":0.0,"token_logprobs":[0.0]}})
```
- [ ] **Step 2 : run → FAIL.**
- [ ] **Step 3 :** dans `reliquary/protocol/submission.py` :
  - `RolloutMetadata` : ajouter (après `token_logprobs`) `forced: bool = False`, `force_span: list[int] | None = None`, `truncated: bool = False` (verbatim validateur submission.py:274-279).
  - `CommitModel.proof_version` : `Literal["v6"]` → `Literal["v7"]`.
- [ ] **Step 4 : run → PASS.** Step 5 checkpoint (+ `PYTHONPATH=. python3 -c "import reliquary.protocol.submission"`).

---

### Task 5 : Cœur BFT (`_bft_assemble_rollouts` + `_rollout_metadata`)

**Files:** Create `reliquary/miner/bft.py` · Test `tests/test_bft_assemble.py`

**Produces:** `bft_assemble_rollouts(*, model, phase1_tensor, prompt_tokens, think_close_ids, force_ids, eos_ids, answer_budget, gen_kwargs=None) -> list[dict]` et `rollout_metadata(generation, token_logprobs) -> dict`.

- [ ] **Step 1 : test** (fake model dont `.generate` renvoie un tenseur-like ; couvre les 3 cas : EOS-en-phase-1, </think>-sans-EOS→phase2 sans force, ni-EOS-ni-</think>→force).
```python
# tests/test_bft_assemble.py
import torch
from reliquary.miner.bft import bft_assemble_rollouts

PLEN = 3; EOS = {2}; CLOSE = {7}; FORCE = [7, 8]

class _Model:
    device = "cpu"
    def generate(self, rows, attention_mask=None, max_new_tokens=0, **kw):
        # append a boxed answer then EOS to every primed row
        return torch.tensor([r.tolist() + [50, 2] for r in rows])

def _p1():
    # row0: EOS in phase1 (…,2); row1: </think>(7) no EOS; row2: neither
    return torch.tensor([[1,1,1, 9, 2, 0],
                         [1,1,1, 7, 9, 9],
                         [1,1,1, 9, 9, 9]])

def test_three_bft_cases():
    out = bft_assemble_rollouts(model=_Model(), phase1_tensor=_p1(),
        prompt_tokens=[1,1,1], think_close_ids=CLOSE, force_ids=FORCE,
        eos_ids=EOS, answer_budget=4, gen_kwargs={})
    assert out[0]["forced"] is False and out[0]["tokens"][-1] == 2   # trimmed at EOS
    assert out[1]["forced"] is False and "force_span" not in out[1]  # closed, no force
    assert out[2]["forced"] is True and out[2]["force_span"][1] - out[2]["force_span"][0] == 2
```
- [ ] **Step 2 : run → FAIL** (module absent).
- [ ] **Step 3 :** créer `reliquary/miner/bft.py` en **collant VERBATIM** `_bft_assemble_rollouts` (validateur engine.py:211-283) et `_rollout_metadata` (`:286-302`), renommés public `bft_assemble_rollouts` / `rollout_metadata`. Imports : `torch`, et depuis `reliquary.shared.modeling` `first_eos_index, has_think_close`.
- [ ] **Step 4 : run → PASS** (3 cas). Step 5 checkpoint.

---

### Task 6 : Génération BFT dans la probe HF (validable sur le vrai 2B)

**Files:** Modify `scripts/difficulty_probe.py` (`stage_generate_code_hf` + une génération math BFT) · Test `tests/test_probe_bft_wire.py`

**But :** brancher le sampler v7 (T=0.6/top_p=0.95/top_k=20) et, pour l'env math, la génération 2-phases BFT via `bft_assemble_rollouts`. Le **code (opencode) reste mono-phase** (bft_applicable=False). Ceci **valide le BFT sur le vrai Qwen3.5-2B** (GPU) ET produit des labels prédicteur corrects.

- [ ] **Step 1 :** ajouter au `stage_generate_code_hf` (et à un `stage_generate_hf` math si on veut labelliser math) : sampler `temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=TOP_K_PROTO` (importés de constants) ; pour math, phase-1 cap `min(max_tokens, BFT_THINKING_BUDGET)` puis `bft_assemble_rollouts(...)`. Pour code : inchangé (mono-phase), juste le sampler v7.
- [ ] **Step 2 : test CPU** (fake model + fake tokenizer) que la branche math appelle `bft_assemble_rollouts` et la branche code non. (Réutiliser le `_Model` de Task 5.)
- [ ] **Step 3 : run → PASS.**
- [ ] **Step 4 : validation GPU (la vraie preuve)** : sur la box, re-lancer la génération code avec `--model ReliquaryForge/qwen3.5-2b-reliquary` (rev `6a8c5637…`) → confirmer que les rollouts sortent, le sampler v7 est utilisé, les labels continus se produisent. (Le 2B est plus petit → plus rapide que le 4B.) Ces labels sont maintenant **valides** pour le prédicteur.
- [ ] **Step 5 : checkpoint.**

---

### Task 7 : Câblage moteur miner (GPU-gaté — vLLM absent)

**Files:** Modify `reliquary/miner/engine.py` (génération + commit GRAIL emit forced/force_span) · **NE PAS toucher vllm_backend.py**

**But :** le miner émet `proof_version v7` + `forced`/`force_span` dans le commit, et sa génération applique BFT+thinking+sampler v7. **Runtime non validable tant que vLLM ne charge pas le 2B** → codé + non-régression CPU, validé GPU quand vLLM+2B sera réglé (session séparée) OU via un fallback HF si décidé.

- [ ] **Step 1 :** dans `_finalize_pool_entry`/`_build_grail_commit` du miner, ajouter au dict `rollout` du commit les champs `forced` + `force_span` depuis l'entrée bakée (défaut `False`/`None` pour les entrées sans BFT). Utiliser `rollout_metadata` (Task 5) comme source.
- [ ] **Step 2 :** `GRAIL_PROOF_VERSION` (Task 1) propage déjà `proof_version="v7"` dans le commit (`_build_grail_commit` lit la constante). Vérifier.
- [ ] **Step 3 :** router la génération math du miner vers la 2-phases BFT (via `bft_assemble_rollouts`) quand `bft_applicable`. ⚠️ **dépend du backend de génération** : le miner génère via vLLM (gelé, ne charge pas 2B). → coder la logique BFT en s'appuyant sur un `model.generate` HF-compatible ; **marquer la validation runtime comme GPU-gated** (Task 7 non testable tant que le modèle ne charge pas côté miner).
- [ ] **Step 4 :** `PYTHONPATH=. python3 -c "import reliquary.miner.engine"` + suite CPU filtrée → **0 nouveau FAIL** (les entrées non-BFT gardent `forced=False`, comportement inchangé).
- [ ] **Step 5 : checkpoint.** Validation E2E = quand vLLM charge le 2B (ou via fallback HF), sur GPU.

---

## Self-Review

**Couverture des changements cot-2b (audits) :**
- proof_version v6→v7 (hard-reject) → Task 1 + Task 4 ✅
- BFT 2-phases + force_span + metadata → Task 5 (cœur) + Task 6 (probe/valid) + Task 7 (miner) ✅
- enable_thinking True → Task 2 ✅
- sampler T=0.6/top_p=0.95/top_k=20 → Task 1 (constantes) + Task 6/7 (usage) ✅
- cap 32768 + constantes BFT → Task 1 ✅
- base model 2B → Task 1 ✅
- helpers modeling → Task 3 ✅
- token-auth 1e-8 → Task 1 ✅
- RolloutMetadata forced/force_span/truncated → Task 4 ✅
- overlong shaping (PASS, rien à faire — training-side, math validator-authoritative) → non-tâche, documenté.

**Placeholders :** blocs « COLLER VERBATIM » = fichiers/lignes exacts du validateur d9471f2 (action définie). Tests = code complet.

**Cohérence types :** `bft_assemble_rollouts(...)`, `rollout_metadata(...)`, `think_close_token_ids/force_close_token_ids/has_think_close`, `RolloutMetadata.forced/force_span/truncated`, `proof_version Literal["v7"]` — identiques test↔impl↔appelants.

**vLLM :** intact (gelé). **SIGMA_MIN** constants 0.33 intact (zone gérée par ZONE_THRESHOLD_STEADY). **BFT math-only** (code reste mono-phase).

**Hors-scope :** fix vLLM↔Qwen3.5 (bump version, session dédiée) ; re-génération complète des données prédicteur (après Task 6, avec le bon modèle) ; ré-appliquer nos optims (σ-zone, multi-env) au-dessus si le moteur de génération change.
