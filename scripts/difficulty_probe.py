#!/usr/bin/env python3
"""Difficulty probe (#10) — does prompt style/wording predict in-zone?

CONTROLLED experiment on a slice of OpenMathInstruct-2 *we* choose, independent
of the validator's per-window slices. Goal: verify that cheap features (and/or
TF-IDF of the wording) predict whether the subnet model lands a prompt in the
σ-zone (2..6 correct out of 8 = reward variance high enough).

Design (locked):
  * Universe = first N shards of OMI train (= the miner's universe, ~1.75M for 4).
    The per-window slice is a RANDOM 5000-window over this FLAT index space, so a
    slice is always a proportional mix of the 4 sources — hence we train on the
    PROPORTIONAL distribution (matches what the miner ranks), with a gsm8k floor.
  * ONE source-aware model (source is a feature, not a partition).
  * X = hand features | TF-IDF | TF-IDF ⊕ structural features (configurable).
  * y = in_zone (2<=k<=6), from running the subnet model (GPU) on each prompt.
  * TRAIN on the proportional 'train' split; TEST on a SEPARATE random/proportional
    'test' split the model never saw. Report train+test AUC (overfit gap),
    learning curve (AUC vs #labels), per-source AUC (diagnostic).

Three stages, runnable independently:
  sample   (CPU) : proportional pick over N shards + gsm8k floor + disjoint test
                   split + feature extraction               -> sample.jsonl (with "split")
  generate (GPU) : M rollouts/prompt with the current checkpoint, grade, label
                   in-zone (labels BOTH splits)             -> labeled.jsonl
  analyze  (CPU) : train on 'train', eval on 'test'; AUC + learning curve + verdict

Only `generate` needs the H100. sample/analyze are pure-CPU.

Usage:
  python scripts/difficulty_probe.py sample   --train-n 10000 --test-n 2000 --gsm8k-floor 300
  python scripts/difficulty_probe.py generate --in sample.jsonl --out labeled.jsonl \
         --model ReliquaryForge/qwen3.5-4b-reliquary --m 8 --max-model-len 8192
  python scripts/difficulty_probe.py analyze  --in labeled.jsonl --features tfidf+hand
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

OMI_REPO = "nvidia/OpenMathInstruct-2"
SHARD_TEMPLATE = "data/train-{i:05d}-of-00032.parquet"
DEFAULT_SHARDS = 4  # must match RELIQUARY_OMI_SHARDS / validator's len(env)
SOURCES = ("augmented_math", "math", "augmented_gsm8k", "gsm8k")
_ANSWER_FORMAT_INSTRUCTION = "\n\nPut your final answer within \\boxed{}."


# ──────────────────────────────────────────────────────────────────────────
# Features — pure, cheap, from text only. CPU.
# ──────────────────────────────────────────────────────────────────────────
def extract_features(problem: str, expected_answer: str, source: str) -> dict:
    q = problem or ""
    a = str(expected_answer or "")
    n_char = len(q)
    ql = q.lower()
    return {
        # one-hot of problem_source (structural; complements TF-IDF wording).
        "src_augmented_math": 1 if source == "augmented_math" else 0,
        "src_math": 1 if source == "math" else 0,
        "src_augmented_gsm8k": 1 if source == "augmented_gsm8k" else 0,
        "src_gsm8k": 1 if source == "gsm8k" else 0,
        "n_char": n_char,
        "n_word": len(q.split()),
        "n_sentence": max(1, len(re.findall(r"[.!?]+", q))),
        "latex_backslash": q.count("\\"),
        "latex_density": q.count("\\") / n_char if n_char else 0.0,
        "dollar": q.count("$"),
        "digit_ratio": sum(c.isdigit() for c in q) / n_char if n_char else 0.0,
        "ans_is_int": 1 if re.fullmatch(r"-?\d+", a.strip()) else 0,
        "ans_is_frac": 1 if "/" in a or "\\frac" in a else 0,
        "ans_is_decimal": 1 if re.fullmatch(r"-?\d+\.\d+", a.strip()) else 0,
        "ans_len": len(a),
        "kw_find": 1 if "find" in ql else 0,
        "kw_prove": 1 if "prove" in ql else 0,
        "kw_evaluate": 1 if ("evaluate" in ql or "compute" in ql) else 0,
        "kw_howmany": 1 if "how many" in ql else 0,
        "kw_let": 1 if re.search(r"\blet\b", ql) else 0,
    }


# Full hand-feature set (--features hand).
HAND_FEATURES = (
    "src_augmented_math", "src_math", "src_augmented_gsm8k", "src_gsm8k",
    "n_char", "n_word", "n_sentence", "latex_backslash", "latex_density",
    "dollar", "digit_ratio", "ans_is_int", "ans_is_frac", "ans_is_decimal",
    "ans_len", "kw_find", "kw_prove", "kw_evaluate", "kw_howmany", "kw_let",
)
# Structural subset to pair with TF-IDF (the bits a bag-of-words can't see:
# source label, notation density, length, answer TYPE from the ground-truth).
STRUCTURAL_FEATURES = (
    "src_augmented_math", "src_math", "src_augmented_gsm8k", "src_gsm8k",
    "latex_density", "n_word", "ans_is_int", "ans_is_frac", "ans_is_decimal",
)


# ──────────────────────────────────────────────────────────────────────────
# Code (opencode) features — pure, cheap, from prompt text only. CPU.
# ──────────────────────────────────────────────────────────────────────────
def extract_features_code(prompt: str) -> dict:
    q = prompt or ""
    ql = q.lower()
    n_char = len(q)
    m = re.search(r"takes\s+(\d+)\s+argument", ql)
    n_args = int(m.group(1)) if m else 0
    return {
        "n_char": n_char,
        "n_word": len(q.split()),
        "n_line": (q.count("\n") + 1) if q else 0,
        "digit_ratio": sum(c.isdigit() for c in q) / n_char if n_char else 0.0,
        "n_args": n_args,
        "has_example": 1 if ("example" in ql or "input:" in ql
                             or "output:" in ql or ">>>" in q) else 0,
        "has_fence": 1 if "```" in q else 0,
        "kw_function": 1 if "function" in ql else 0,
        "kw_return": 1 if "return" in ql else 0,
        "kw_list": 1 if ("list" in ql or "array" in ql) else 0,
        "kw_string": 1 if "string" in ql else 0,
        "kw_integer": 1 if ("integer" in ql or "number" in ql) else 0,
        "kw_loop": 1 if ("iterate" in ql or "each" in ql or "loop" in ql) else 0,
        "kw_sort": 1 if "sort" in ql else 0,
        "kw_implement": 1 if ("implement" in ql or "write a" in ql) else 0,
    }


CODE_HAND_FEATURES = (
    "n_char", "n_word", "n_line", "digit_ratio", "n_args", "has_example",
    "has_fence", "kw_function", "kw_return", "kw_list", "kw_string",
    "kw_integer", "kw_loop", "kw_sort", "kw_implement",
)
CODE_STRUCTURAL_FEATURES = (
    "n_word", "n_line", "digit_ratio", "n_args", "has_example",
    "kw_function", "kw_list", "kw_string",
)

# Steady zone threshold (matches the validator's SIGMA_MIN=0.43). The code
# reward is CONTINUOUS (passed/total), so in-zone = std(rewards) >= this, NOT a
# binary k-band. (For binary math, 2<=k<=6 out of 8 is exactly std>=0.43.)
CODE_SIGMA_MIN = 0.43


def _code_in_zone(rewards, sigma_min: float = CODE_SIGMA_MIN):
    """(sigma, in_zone) over CONTINUOUS rewards. in_zone=1 iff std >= sigma_min."""
    n = len(rewards)
    if n == 0:
        return 0.0, 0
    mean = sum(rewards) / n
    sigma = (sum((x - mean) ** 2 for x in rewards) / n) ** 0.5
    return sigma, (1 if sigma >= sigma_min else 0)


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — sample (CPU). Proportional train (+ gsm8k floor) + disjoint random test.
# ──────────────────────────────────────────────────────────────────────────
def stage_sample(train_n, test_n, gsm8k_floor, shards, out_path, seed) -> None:
    import random as _random

    from datasets import load_dataset

    data_files = [SHARD_TEMPLATE.format(i=i) for i in range(shards)]
    ds = load_dataset(OMI_REPO, data_files=data_files, split="train")
    n = len(ds)
    print(f"[sample] universe = {n:,} rows over {shards} shards")
    srccol = list(ds["problem_source"])  # materialise once (~1.75M strings)

    rng = _random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)

    # TRAIN: proportional (random draw over shuffled universe).
    train_idx = order[:train_n]
    rest = order[train_n:]
    # gsm8k FLOOR: top up rare source so it's at least learnable (keeps the rest
    # proportional — only adds gsm8k, never removes the majority).
    gsm_have = sum(1 for i in train_idx if srccol[i] == "gsm8k")
    if gsm_have < gsm8k_floor:
        need = gsm8k_floor - gsm_have
        extra = [i for i in rest if srccol[i] == "gsm8k"][:need]
        train_idx = train_idx + extra
        extra_set = set(extra)
        rest = [i for i in rest if i not in extra_set]
        print(f"[sample] gsm8k floor: had {gsm_have}, added {len(extra)} → {gsm_have+len(extra)}")
    # TEST: separate, disjoint, PURELY random/proportional (no floor — honest eval).
    test_idx = rest[:test_n]

    def src_breakdown(idxs):
        c = defaultdict(int)
        for i in idxs:
            c[srccol[i]] += 1
        return {s: c.get(s, 0) for s in SOURCES}

    print(f"[sample] train n={len(train_idx)} sources={src_breakdown(train_idx)}")
    print(f"[sample] test  n={len(test_idx)} sources={src_breakdown(test_idx)}")

    with open(out_path, "w") as f:
        for split, idxs in (("train", train_idx), ("test", test_idx)):
            for idx in idxs:
                row = ds[idx]
                feats = extract_features(
                    row["problem"], row.get("expected_answer", ""),
                    row["problem_source"],
                )
                f.write(json.dumps({
                    "split": split,
                    "dataset_index": idx,
                    "prompt": row["problem"] + _ANSWER_FORMAT_INSTRUCTION,
                    "ground_truth": str(row.get("expected_answer", "")),
                    "problem_source": row["problem_source"],
                    "features": feats,
                }) + "\n")
    print(f"[sample] wrote {len(train_idx)+len(test_idx)} rows → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — generate + grade (GPU). Labels BOTH splits. Reuses the miner backend.
# ──────────────────────────────────────────────────────────────────────────
def count_truncated(rollouts, eos_ids) -> int:
    """Rollouts with NO EOS = hit the generation cap → dropped by the miner
    (terminating_rollouts / too_many_truncated). Diagnostic: high counts at the
    prod cap mean the prompt's solution overflows the generation cap."""
    eos = set(int(e) for e in (eos_ids or ()))
    if not eos:
        return 0
    return sum(
        1 for toks in rollouts if not any(int(t) in eos for t in toks)
    )


def bft_generate_math(backend, tokenizer, prompt_token_ids, m, temperature,
                      eos_ids):
    """m rollouts per prompt following the v7 BFT math flow, free sampling.

    Mirrors ``reliquary.miner.bft.bft_assemble_rollouts`` structurally (EOS →
    keep; ``</think>`` without EOS → natural phase-2; neither → inject the
    FORCE template then phase-2) with the protocol sampler and the protocol
    budgets (phase-1 = EXACTLY BFT_THINKING_BUDGET, phase-2 = BFT_ANSWER_BUDGET).
    Free sampling (no forced seed) is fine for difficulty LABELS — the forced
    stream is a draw from the same warped distribution — but the sampler and
    the BFT structure must match, or the labels are off-distribution.

    Returns, per prompt, ``m`` completion token-id lists (prompt stripped) —
    the same shape ``grade()`` consumes.
    """
    from reliquary.constants import (
        BFT_ANSWER_BUDGET, BFT_THINKING_BUDGET, TOP_K_PROTO, TOP_P_PROTO,
    )
    from reliquary.shared.modeling import (
        first_eos_index, force_close_token_ids, has_think_close,
        think_close_token_ids,
    )

    close_set = set(think_close_token_ids(tokenizer))
    force_ids = [int(t) for t in force_close_token_ids(tokenizer)]
    stop_ids = sorted(int(t) for t in eos_ids)

    phase1 = backend.generate_multi(
        prompt_token_ids, n=m, temperature=temperature,
        top_p=TOP_P_PROTO, top_k=TOP_K_PROTO,
        max_tokens=BFT_THINKING_BUDGET, stop_token_ids=stop_ids,
    )

    results = [[None] * m for _ in prompt_token_ids]
    p2_inputs: list[list[int]] = []   # primed prompt+gen(+force) sequences
    p2_slots: list[tuple[int, int, list[int]]] = []  # (prompt_i, rollout_j, gen-part)
    for i, (ptoks, rollouts) in enumerate(zip(prompt_token_ids, phase1)):
        for j, gen in enumerate(rollouts):
            gen = list(gen)
            fe = first_eos_index(gen, eos_ids)
            if fe is not None:
                results[i][j] = gen[: fe + 1]
            elif has_think_close(gen, close_set):
                p2_inputs.append(list(ptoks) + gen)
                p2_slots.append((i, j, gen))
            else:
                p2_inputs.append(list(ptoks) + gen + force_ids)
                p2_slots.append((i, j, gen + force_ids))

    if p2_inputs:
        phase2 = backend.generate_multi(
            p2_inputs, n=1, temperature=temperature,
            top_p=TOP_P_PROTO, top_k=TOP_K_PROTO,
            max_tokens=BFT_ANSWER_BUDGET, stop_token_ids=stop_ids,
        )
        for (i, j, gen_part), tails in zip(p2_slots, phase2):
            tail = list(tails[0])
            fe = first_eos_index(tail, eos_ids)
            results[i][j] = gen_part + (tail[: fe + 1] if fe is not None else tail)
    return results


def stage_generate(in_path, out_path, model, m, temperature, max_tokens,
                   max_model_len, batch) -> None:
    del max_tokens  # math budgets are protocol-pinned (BFT 2048→force→512)
    import torch  # noqa: F401  (GPU env presence check)

    from reliquary.constants import (
        BFT_ANSWER_BUDGET, BFT_THINKING_BUDGET, TOP_K_PROTO, TOP_P_PROTO,
    )
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    from reliquary.miner.vllm_backend import VLLMBackend
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        first_eos_index, load_tokenizer, resolve_eos_token_ids,
    )

    records = [json.loads(l) for l in open(in_path)]
    print(f"[generate] {len(records)} prompts, m={m} rollouts each "
          f"(v7 BFT: sampler T={temperature}/top_p={TOP_P_PROTO}/top_k={TOP_K_PROTO}, "
          f"think {BFT_THINKING_BUDGET} → force → answer {BFT_ANSWER_BUDGET})")

    tokenizer = load_tokenizer(model)
    # forced_seed=True selects the Triton/FLA GDN prefill backend (the only one
    # that loads the qwen3.5-2b hybrid — flashinfer's GDN JIT crashes on ptxas).
    # It does NOT force the sampling: generate_multi still samples freely from
    # the SamplingParams below (top_p/top_k protocole). See vllm_backend._build_llm.
    backend = VLLMBackend(model_path=model, tokenizer_path=model,
                          max_model_len=max_model_len, forced_seed=True)
    try:
        eos_ids = resolve_eos_token_ids(None, tokenizer)
    except Exception:
        eos_ids = [tokenizer.eos_token_id]
    env = OpenMathInstructEnvironment()

    def grade(problem, token_ids):
        cut = first_eos_index(token_ids, eos_ids)
        ids = token_ids[:cut] if cut is not None else token_ids
        return env.compute_reward(problem, tokenizer.decode(ids, skip_special_tokens=True))

    out = open(out_path, "w")
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        prompt_ids = [encode_prompt(tokenizer, r["prompt"]) for r in chunk]
        gen = bft_generate_math(
            backend, tokenizer, prompt_ids, m=m, temperature=temperature,
            eos_ids=set(eos_ids),
        )
        for r, rollouts in zip(chunk, gen):
            problem = {"prompt": r["prompt"], "ground_truth": r["ground_truth"],
                       "id": r["dataset_index"]}
            k = int(sum(round(grade(problem, toks)) for toks in rollouts))
            r["k"], r["m"], r["p_correct"] = k, m, k / m
            r["in_zone"] = 1 if 2 <= k <= (m - 2) else 0
            out.write(json.dumps(r) + "\n")
        print(f"[generate] {min(start + batch, len(records))}/{len(records)}")
    out.close()
    print(f"[generate] wrote labels → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Code (opencode) stages — mirror math, but curated env + continuous σ label.
# ──────────────────────────────────────────────────────────────────────────
def stage_sample_code(train_n, test_n, out_path, seed) -> None:
    """Sample curated opencode prompts (with contract) + code features. CPU-ish
    (loads the curated dataset over the network). No problem_source / no floor."""
    import random as _random

    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    env = OpenCodeInstructEnvironment()
    n = len(env)
    print(f"[sample-code] universe = {n:,} curated prompts")
    rng = _random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    train_idx = order[:train_n]
    test_idx = order[train_n:train_n + test_n]
    print(f"[sample-code] train n={len(train_idx)}  test n={len(test_idx)}")

    # Campagne v4.3 (2026-08-15) : PAS de fetch par prompt ici — 700 ms/ligne
    # réseau x 21k = 4 h de CPU avant le premier token GPU. Le texte est
    # re-dérivé par stage_generate_code au fil des batches (coût masqué par la
    # génération) et écrit dans le label à ce moment-là.
    with open(out_path, "w") as f:
        for split, idxs in (("train", train_idx), ("test", test_idx)):
            for idx in idxs:
                f.write(json.dumps({
                    "split": split,
                    "dataset_index": idx,
                    "env": "code",
                }) + "\n")
    print(f"[sample-code] wrote {len(train_idx)+len(test_idx)} rows → {out_path}")


def stage_generate_code(in_path, out_path, model, m, temperature, max_tokens,
                        max_model_len, batch) -> None:
    """GPU: M rollouts/prompt, grade CONTINUOUS (passed/total), label in_zone by
    std(rewards) >= 0.43. Re-derives each prompt+cases via env.get_problem(idx)
    (the structured_cases live in-memory, lost across the sample→generate process)."""
    import torch  # noqa: F401  (GPU env presence check)

    from reliquary.constants import TOP_K_PROTO, TOP_P_PROTO
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    from reliquary.miner.vllm_backend import VLLMBackend
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        first_eos_index, load_tokenizer, resolve_eos_token_ids,
    )

    records = [json.loads(l) for l in open(in_path)]
    print(f"[generate-code] {len(records)} prompts, m={m} rollouts each")

    tokenizer = load_tokenizer(model)
    # forced_seed=True selects the Triton/FLA GDN prefill backend (the only one
    # that loads the qwen3.5-2b hybrid — flashinfer's GDN JIT crashes on ptxas).
    # It does NOT force the sampling: generate_multi still samples freely from
    # the SamplingParams below (top_p/top_k protocole). See vllm_backend._build_llm.
    backend = VLLMBackend(model_path=model, tokenizer_path=model,
                          max_model_len=max_model_len, forced_seed=True)
    try:
        eos_ids = resolve_eos_token_ids(None, tokenizer)
    except Exception:
        eos_ids = [tokenizer.eos_token_id]
    env = OpenCodeInstructEnvironment()

    def grade(problem, token_ids):
        cut = first_eos_index(token_ids, eos_ids)
        ids = token_ids[:cut] if cut is not None else token_ids
        return env.compute_reward(problem, tokenizer.decode(ids, skip_special_tokens=True))

    # REPRISE par INDEX (2026-08-15 soir) : dédup par dataset_index (robuste à
    # l ordre dynamique du mode actif), sortie en append.
    import os as _os
    done_idx = set()
    if _os.path.exists(out_path):
        for _l in open(out_path):
            try: done_idx.add(json.loads(_l)["dataset_index"])
            except Exception: pass
        print(f"[generate-code] reprise: {len(done_idx)} labels existants (dédup par index)")
    records = [r for r in records if r["dataset_index"] not in done_idx]
    done = 0
    out = open(out_path, "a")

    # MODE ACTIF (RELIQUARY_PROBE_ACTIVE=1) : apprentissage actif au fil de
    # l eau — un TAMPON de candidats préchargés est re-noté par le modèle v5
    # (poids RELIQUARY_PROBE_MODEL) ; cycles alternés : impair = top-batch du
    # tampon (bras "guided"), pair = tirage uniforme (bras "uniform"). Le
    # pré-scoring hors-ligne du monde entier (3-4 h de fetch) devient inutile :
    # on ne note que ce qui est déjà téléchargé pour générer.
    _active = _os.environ.get("RELIQUARY_PROBE_ACTIVE", "0") == "1"
    _score_model = None

    def _load_score_model():
        # feat_logit = 11 features ; word_logit = 11 features + vocabulaire
        # (300 mots, présence binaire). Rechargé à chaque cycle guided → un
        # nouveau JSON déposé s applique sans restart.
        m = json.load(open(_os.environ.get(
            "RELIQUARY_PROBE_MODEL", "/workspace/predictor_v5_beta.json")))
        if m.get("type") == "word_logit":
            m["_vidx"] = {w: i for i, w in enumerate(m["vocab"])}
        return m

    if _active:
        try:
            _score_model = _load_score_model()
            print(f"[generate-code] MODE ACTIF: tampon re-noté "
                  f"({_score_model.get('type', 'feat_logit')}), cycles guided/uniform alternés")
        except Exception as e:
            print(f"[generate-code] mode actif indisponible ({e}) — uniforme pur")
            _active = False

    def _v5_feats(q):
        ql = q.lower(); n = len(q)
        return [1.0, n/1000, len(q.split())/100, (q.count("\n")+1)/10,
                sum(c.isdigit() for c in q)/max(1, n)*10,
                1.0 if ("example" in ql or ">>>" in q or "input:" in ql) else 0.0,
                1.0 if "```" in q else 0.0, 1.0 if "class " in q else 0.0,
                1.0 if "string" in ql else 0.0,
                1.0 if ("list" in ql or "array" in ql) else 0.0,
                (n/1000)**2]

    import re as _re

    def _v5_score(q):
        w_all = _score_model["weights"]
        base = sum(w_*x for w_, x in zip(w_all, _v5_feats(q)))
        vidx = _score_model.get("_vidx")
        if vidx:
            for wd in set(_re.findall(r"[a-z]{3,}", q.lower())):
                i = vidx.get(wd)
                if i is not None:
                    base += w_all[11 + i]
        return base
    # PRÉCHARGEMENT (2026-08-15) : get_problem est réseau (~0,7 s/prompt) —
    # en série dans la boucle ça coûtait ~34 s de GPU idle par batch. On
    # précharge le batch SUIVANT pendant la génération du courant (recouvrement
    # complet), avec un pool de threads pour le premier.
    from concurrent.futures import ThreadPoolExecutor
    # TIMEOUT réseau (2026-08-16) : sans lui, une socket fermée côté serveur
    # (CLOSE_WAIT) gèle un thread de fetch pour toujours → tampon jamais
    # rempli → probe bloquée GPU à 0 % (incident H200 ~22:00 UTC).
    import socket as _socket
    _socket.setdefaulttimeout(60)
    _fetch_pool = ThreadPoolExecutor(max_workers=12)
    _fetch_inner = ThreadPoolExecutor(max_workers=24)

    def _fetch_one(r):
        for _try in range(2):
            fut = _fetch_inner.submit(env.get_problem, r["dataset_index"])
            try:
                return fut.result(timeout=75)
            except Exception:
                continue
        print(f"[generate-code] fetch abandonné idx={r['dataset_index']}", flush=True)
        return None

    def _fetch(chunk):
        return list(_fetch_pool.map(_fetch_one, chunk))

    # RECOUVREMENT (2026-08-15 nuit) : la correction (CPU, 30-90 s) du batch N
    # tourne en fond PENDANT la génération GPU du batch N+1 → plus de creux à
    # 0 % entre les batches. File bornée à 1 (on attend la correction N-1 avant
    # de soumettre la N) ; les erreurs remontent au .result().
    _grade_pool = ThreadPoolExecutor(max_workers=1)
    _grade_state = {"fut": None}

    def _run_batch(chunk, problems):
        prompt_ids = [encode_prompt(tokenizer, p["prompt"]) for p in problems]
        # Protocol sampler (v7): match what the miner actually draws under
        # forced-seed (warp = T_PROTO/top_p=0.95/top_k=20). Free sampling
        # (top_p=1.0, top_k=-1) samples a fuller distribution → biased
        # difficulty estimate that would not transfer to production.
        gen = backend.generate_multi(
            prompt_ids, n=m, temperature=temperature,
            top_p=TOP_P_PROTO, top_k=TOP_K_PROTO,
            max_tokens=max_tokens, stop_token_ids=eos_ids,
        )
        if _grade_state["fut"] is not None:
            _grade_state["fut"].result()
        _grade_state["fut"] = _grade_pool.submit(_grade_and_write, chunk, problems, gen)

    def _grade_and_write(chunk, problems, gen):
        # Grading PARALLÈLE (2026-08-15) : le sandbox subprocess libère le GIL
        # → ThreadPool comme grade_group_parallel. Appelé depuis le thread de
        # fond (seul écrivain de `out`).
        with ThreadPoolExecutor(max_workers=32) as _ex:
            all_rewards = list(_ex.map(
                lambda pr: [grade(pr[0], toks) for toks in pr[1]],
                [(problem, rollouts) for _, problem, rollouts
                 in zip(chunk, problems, gen)]))
        for (r, problem, rollouts), rewards in zip(
                zip(chunk, problems, gen), all_rewards):
            sigma, in_zone = _code_in_zone(rewards)
            r["n_truncated"] = count_truncated(rollouts, eos_ids)
            # completion_lens : longueur jusqu'au 1er EOS inclus (tronqué =
            # longueur brute) — le champ que la cible bucket de v4.3 consomme
            # (Σ, max), même convention que le dump du mineur.
            lens = []
            for toks in rollouts:
                cut = first_eos_index(toks, eos_ids)
                lens.append((cut + 1) if cut is not None else len(toks))
            r["completion_lens"] = lens
            r["prompt_idx"] = r["dataset_index"]  # alias format mineur
            r["prompt"] = problem["prompt"]  # texte pour l'entraînement (fetch amorti ici)
            r["rewards"], r["m"], r["sigma"], r["in_zone"] = rewards, m, sigma, in_zone
            out.write(json.dumps(r) + "\n")
        out.flush()

    def _drain_grades():
        if _grade_state["fut"] is not None:
            _grade_state["fut"].result()
            _grade_state["fut"] = None

    if not _active:
        # Chemin historique : ordre du plan, préchargement du batch suivant.
        next_future = None
        for start in range(0, len(records), batch):
            chunk = records[start:start + batch]
            problems = next_future.result() if next_future is not None else _fetch(chunk)
            nxt = records[start + batch:start + 2 * batch]
            next_future = _fetch_pool.submit(_fetch, nxt) if nxt else None
            keep = [(r, pb) for r, pb in zip(chunk, problems) if pb is not None]
            if not keep:
                continue
            chunk, problems = [r for r, _ in keep], [pb for _, pb in keep]
            _run_batch(chunk, problems)
            print(f"[generate-code] {min(start + batch, len(records))}/{len(records)}", flush=True)
        _drain_grades()
    else:
        # Boucle À TAMPON (mode actif) : on garde ~4 batches de candidats déjà
        # téléchargés ; cycle pair = top-batch v5 du tampon (bras "guided"),
        # cycle impair = tirage uniforme (bras "uniform"). Le refill (réseau)
        # recouvre la génération (GPU) — ~6 s de fetch pour ~7 min de GPU.
        import random as _random
        _rng = _random.Random(815)
        state = {"ptr": 0, "pending": None}
        buf = []
        TARGET = 4 * batch

        def _submit_refill():
            if state["pending"] is None and state["ptr"] < len(records):
                c = records[state["ptr"]:state["ptr"] + batch]
                state["ptr"] += len(c)
                state["pending"] = _fetch_pool.submit(
                    lambda cc=c: list(zip(cc, _fetch(cc))))

        def _drain_refill(block):
            if state["pending"] is not None and (block or state["pending"].done()):
                for rec, pb in state["pending"].result():
                    if pb is not None:
                        buf.append((rec, pb))
                state["pending"] = None

        while len(buf) < TARGET and (state["ptr"] < len(records)
                                     or state["pending"] is not None):
            _submit_refill()
            _drain_refill(block=True)
        _submit_refill()

        cycle, labeled = 0, 0
        while buf:
            n_take = min(batch, len(buf))
            if cycle % 2 == 0:
                try:  # recharge à chaud : déposer un nouveau JSON suffit
                    _score_model = _load_score_model()
                except Exception:
                    pass  # fichier en cours d écriture → on garde le modèle courant
                order = sorted(range(len(buf)),
                               key=lambda i: _v5_score(buf[i][1]["prompt"]),
                               reverse=True)
                take, arm = set(order[:n_take]), "guided"
            else:
                take, arm = set(_rng.sample(range(len(buf)), n_take)), "uniform"
            sel = [buf[i] for i in range(len(buf)) if i in take]
            buf = [buf[i] for i in range(len(buf)) if i not in take]
            chunk = [rec for rec, _pb in sel]
            problems = [pb for _rec, pb in sel]
            for rec in chunk:
                rec["arm"] = arm
            _run_batch(chunk, problems)
            labeled += len(chunk)
            cycle += 1
            _drain_refill(block=False)
            _submit_refill()
            if not buf and (state["pending"] is not None
                            or state["ptr"] < len(records)):
                _drain_refill(block=True)
                _submit_refill()
            print(f"[generate-code] {labeled}/{len(records)} (cycle {cycle}, "
                  f"bras {arm}, tampon {len(buf)})", flush=True)
        _drain_grades()
    out.close()
    print(f"[generate-code] wrote labels → {out_path}")


def stage_generate_code_hf(in_path, out_path, model, m, temperature, max_tokens,
                           max_model_len, batch) -> None:
    """GPU via HF transformers (NO vLLM) — loads the EXACT model with the
    validator's loader (`load_text_generation_model`), which handles Qwen3.5
    (conditional image-text-to-text, text-only) that vLLM 0.10.2 can't load.
    Slower than vLLM but exact. Same continuous-σ labelling as the vLLM path.
    Re-derives prompt+cases via env.get_problem(idx)."""
    del max_model_len  # HF has no fixed engine context; kept for CLI symmetry
    import torch

    # torch 2.11+cu130 cuDNN SDPA has no execution plan for qwen3_5's hybrid
    # attention → fall back to the mem-efficient / math SDPA backend.
    torch.backends.cuda.enable_cudnn_sdp(False)

    from reliquary.constants import TOP_K_PROTO, TOP_P_PROTO
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        first_eos_index, load_text_generation_model, load_tokenizer,
        resolve_eos_token_ids,
    )

    records = [json.loads(l) for l in open(in_path)]
    print(f"[generate-code-hf] {len(records)} prompts, m={m} rollouts each (HF backend, "
          f"v7 sampler T={temperature}/top_p={TOP_P_PROTO}/top_k={TOP_K_PROTO})")

    tokenizer = load_tokenizer(model)
    tokenizer.padding_side = "left"  # left-pad for batched generation
    hf_model = load_text_generation_model(
        model, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    eos_ids = resolve_eos_token_ids(hf_model, tokenizer) or {tokenizer.eos_token_id}
    eos_list = sorted(eos_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_list[0]
    env = OpenCodeInstructEnvironment()

    def grade(problem, comp_ids):
        cut = first_eos_index(comp_ids, eos_ids)
        ids = comp_ids[:cut] if cut is not None else comp_ids
        return env.compute_reward(problem, tokenizer.decode(ids, skip_special_tokens=True))

    out = open(out_path, "w")
    done = 0
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        problems = [env.get_problem(r["dataset_index"]) for r in chunk]
        ptoks = [encode_prompt(tokenizer, p["prompt"]) for p in problems]
        maxlen = max(len(t) for t in ptoks)
        # Left-pad the batch to maxlen (generation reads the last position).
        input_ids = torch.full((len(ptoks), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ptoks), maxlen), dtype=torch.long)
        for i, t in enumerate(ptoks):
            input_ids[i, maxlen - len(t):] = torch.tensor(t, dtype=torch.long)
            attn[i, maxlen - len(t):] = 1
        input_ids = input_ids.to("cuda")
        attn = attn.to("cuda")
        with torch.no_grad():
            gen = hf_model.generate(
                input_ids, attention_mask=attn, do_sample=True,
                temperature=temperature, top_p=TOP_P_PROTO, top_k=TOP_K_PROTO,
                max_new_tokens=max_tokens,
                num_return_sequences=m, eos_token_id=eos_list, pad_token_id=pad_id,
            )
        # gen: [len(chunk)*m, maxlen+new]. Prompt i's rollouts = rows [i*m:(i+1)*m];
        # completion = everything after the (left-padded) prompt block (maxlen).
        for i, (r, problem) in enumerate(zip(chunk, problems)):
            seqs = gen[i * m:(i + 1) * m]
            comp_ids_list = [seq[maxlen:].tolist() for seq in seqs]
            rewards = [grade(problem, comp_ids) for comp_ids in comp_ids_list]
            sigma, in_zone = _code_in_zone(rewards)
            r["n_truncated"] = count_truncated(comp_ids_list, eos_ids)
            r["rewards"], r["m"], r["sigma"], r["in_zone"] = rewards, m, sigma, in_zone
            out.write(json.dumps(r) + "\n")
        done += len(chunk)
        print(f"[generate-code-hf] {done}/{len(records)} prompts")
    out.close()
    print(f"[generate-code-hf] wrote labels → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Stage 3 — analyze (CPU). Train on 'train' split, eval on 'test' split.
# ──────────────────────────────────────────────────────────────────────────
def _build_X(train, test, mode, tfidf_max_features, min_df,
             hand_feats=HAND_FEATURES, struct_feats=STRUCTURAL_FEATURES):
    """Return (Xtr, Xte) for the chosen feature mode. sklearn-based."""
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    def hand_matrix(rows, names, scaler=None, fit=False):
        X = np.array([[float(r["features"][f]) for f in names] for r in rows], float)
        if scaler is not None:
            X = scaler.fit_transform(X) if fit else scaler.transform(X)
        return X

    if mode == "hand":
        sc = StandardScaler()
        return hand_matrix(train, hand_feats, sc, True), hand_matrix(test, hand_feats, sc, False)

    # TF-IDF fitted on TRAIN prompts only (no leakage).
    vec = TfidfVectorizer(max_features=tfidf_max_features, min_df=min_df,
                          ngram_range=(1, 2), sublinear_tf=True)
    Xtr_t = vec.fit_transform(r["prompt"] for r in train)
    Xte_t = vec.transform(r["prompt"] for r in test)
    print(f"[analyze] TF-IDF vocab = {len(vec.vocabulary_)} terms")
    if mode == "tfidf":
        return Xtr_t, Xte_t
    # tfidf+hand: append the structural subset (scaled), sparse-stacked.
    sc = StandardScaler()
    Htr = csr_matrix(hand_matrix(train, struct_feats, sc, True))
    Hte = csr_matrix(hand_matrix(test, struct_feats, sc, False))
    return hstack([Xtr_t, Htr]).tocsr(), hstack([Xte_t, Hte]).tocsr()


def stage_analyze(in_path, features_mode, tfidf_max_features, min_df) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rows = [json.loads(l) for l in open(in_path)]
    rows = [r for r in rows if "in_zone" in r]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    if not train or not test:  # backward-compat: random 80/20 if no split tags
        import random as _r
        _r.Random(0).shuffle(rows)
        cut = max(1, int(len(rows) * 0.2))
        test, train = rows[:cut], rows[cut:]
        print("[analyze] no split tags → random 80/20")

    # Env-aware: code rows carry code features (no problem_source); math rows
    # carry the OMI source one-hots. Pick the matching hand/structural sets.
    is_code = bool(rows) and rows[0].get("env") == "code"
    hand_feats = CODE_HAND_FEATURES if is_code else HAND_FEATURES
    struct_feats = CODE_STRUCTURAL_FEATURES if is_code else STRUCTURAL_FEATURES

    ytr = np.array([r["in_zone"] for r in train])
    yte = np.array([r["in_zone"] for r in test])
    print(f"\n=== Difficulty probe — env={'code' if is_code else 'math'} features={features_mode} ===")
    print(f"train={len(train)} (in-zone {100*ytr.mean():.1f}%)  "
          f"test={len(test)} (in-zone {100*yte.mean():.1f}%)")

    Xtr, Xte = _build_X(train, test, features_mode, tfidf_max_features, min_df,
                        hand_feats, struct_feats)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    auc_tr = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
    p_te = clf.predict_proba(Xte)[:, 1]
    auc_te = roc_auc_score(yte, p_te)
    print(f"\nAUC train = {auc_tr:.3f}   AUC test (held-out) = {auc_te:.3f}"
          f"   gap = {auc_tr-auc_te:+.3f}  ({'overfit' if auc_tr-auc_te>0.1 else 'ok'})")

    # per-source AUC on test (diagnostic — MATH only; code has no problem_source).
    if not is_code:
        print("\nper-source test AUC (diagnostic):")
        for s in SOURCES:
            m = np.array([r["problem_source"] == s for r in test])
            if m.sum() >= 20 and 0 < yte[m].sum() < m.sum():
                print(f"  {s:18s} n={int(m.sum()):>5}  AUC={roc_auc_score(yte[m], p_te[m]):.3f}")
        else:
            print(f"  {s:18s} n={int(m.sum()):>5}  (trop peu / une seule classe)")

    # learning curve — test AUC vs fraction of train (vectorizer fixed on full train).
    print("\nlearning curve (test AUC vs #train labels):")
    rng = np.random.RandomState(0)
    order = rng.permutation(len(train))
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(2, int(len(train) * frac))
        sub = order[:k]
        if len(set(ytr[sub].tolist())) < 2:
            continue
        c = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        c.fit(Xtr[sub], ytr[sub])
        a = roc_auc_score(yte, c.predict_proba(Xte)[:, 1])
        print(f"  {int(frac*100):>3}% ({k:>5} labels)  test AUC={a:.3f}")

    print("\n=== VERDICT ===")
    if auc_te >= 0.60:
        print(f"  test AUC {auc_te:.3f} ≥ 0.60 → SIGNAL : le prédicteur marche, brancher (#9).")
    elif auc_te >= 0.55:
        print(f"  test AUC {auc_te:.3f} faible → escalader (tfidf→embeddings) ou plus de labels.")
    else:
        print(f"  test AUC {auc_te:.3f} ≈ 0.5 → pas de signal exploitable → oversample/throughput.")
    print("  (regarde la learning curve : si elle monte encore → plus de labels aiderait.)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--env", choices=("math", "code"), default="math")
    s.add_argument("--train-n", type=int, default=10000)
    s.add_argument("--test-n", type=int, default=2000)
    s.add_argument("--gsm8k-floor", type=int, default=300)
    s.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    s.add_argument("--out", default="sample.jsonl")
    s.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("generate")
    g.add_argument("--env", choices=("math", "code"), default="math")
    g.add_argument("--backend", choices=("vllm", "hf"), default="vllm",
                   help="hf = generate via transformers (loads Qwen3.5, no vLLM)")
    g.add_argument("--in", dest="in_path", required=True)
    g.add_argument("--out", default="labeled.jsonl")
    g.add_argument("--model", required=True, help="HF repo/local path of the CURRENT published checkpoint")
    g.add_argument("--m", type=int, default=8)
    g.add_argument("--temperature", type=float, default=0.6)  # T_PROTO (v7)
    g.add_argument("--max-tokens", type=int, default=2600,
                   help="code envs only (PROD cap = 2600, ops/launch_miner.sh); "
                        "math ignores it (protocol BFT budgets)")
    g.add_argument("--max-model-len", type=int, default=8192)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--expect-protocol", type=int, default=None,
                   help="garde-fou (audit v4 item 9) : sys.exit si "
                        "constants.PROTOCOL_VERSION du process ne matche pas. "
                        "Les prompts/labels dépendent du protocole "
                        "(RAW_COMPLETION_PROMPTS, zone) — un probe v4 lancé "
                        "sans RELIQUARY_PROTOCOL_VERSION=4 produit des "
                        "données v3 SILENCIEUSEMENT. Commande v4 type : "
                        "RELIQUARY_PROTOCOL_VERSION=4 … --expect-protocol 4 "
                        "--temperature 1.0 (le défaut T=0.6 n'est PAS corrigé "
                        "par l'env var).")

    a = sub.add_parser("analyze")
    a.add_argument("--in", dest="in_path", required=True)
    a.add_argument("--features", choices=("hand", "tfidf", "tfidf+hand"), default="tfidf+hand")
    a.add_argument("--tfidf-max-features", type=int, default=5000)
    a.add_argument("--min-df", type=int, default=3)

    args = ap.parse_args()
    if args.stage == "sample":
        if args.env == "code":
            stage_sample_code(args.train_n, args.test_n, args.out, args.seed)
        else:
            stage_sample(args.train_n, args.test_n, args.gsm8k_floor, args.shards,
                         args.out, args.seed)
    elif args.stage == "generate":
        from reliquary import constants as _c
        if (args.expect_protocol is not None
                and _c.PROTOCOL_VERSION != args.expect_protocol):
            sys.exit(
                f"[generate] ABORT: constants.PROTOCOL_VERSION="
                f"{_c.PROTOCOL_VERSION} != --expect-protocol "
                f"{args.expect_protocol} — lancer avec "
                f"RELIQUARY_PROTOCOL_VERSION={args.expect_protocol}"
            )
        print(f"[generate] proto=v{_c.PROTOCOL_VERSION} "
              f"raw_prompts={getattr(_c, 'RAW_COMPLETION_PROMPTS', False)}")
        if args.env == "code" and args.backend == "hf":
            stage_generate_code_hf(args.in_path, args.out, args.model, args.m,
                                   args.temperature, args.max_tokens,
                                   args.max_model_len, args.batch)
        elif args.env == "code":
            stage_generate_code(args.in_path, args.out, args.model, args.m,
                                args.temperature, args.max_tokens, args.max_model_len,
                                args.batch)
        else:
            stage_generate(args.in_path, args.out, args.model, args.m, args.temperature,
                           args.max_tokens, args.max_model_len, args.batch)
    elif args.stage == "analyze":
        stage_analyze(args.in_path, args.features, args.tfidf_max_features, args.min_df)


if __name__ == "__main__":
    main()
