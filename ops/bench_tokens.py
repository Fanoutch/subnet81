"""Banc d'essai OFFLINE du débit de génération — sans validateur, sans subnet.

Mesure le débit tokens/s du CHEMIN DE PRODUCTION : phase-1 forced-seed sur vLLM
(processeur VLLMForcedSeedLogitsProcessor enregistré, greedy, extra_args par
séquence), exactement comme `generate_forced_phase1_multi` dans le mineur.

Pourquoi offline : le débit de génération est une propriété LOCALE du GPU. Le
mesurer via le mineur le pollue avec les flips de fenêtre, les avances de
checkpoint et les tailles de tranche variables — ce sont ces variables parasites
qui ont produit de fausses conclusions le 2026-07-21.

CONTRÔLE : `ignore_eos=True` force chaque séquence à générer exactement
`--max-tokens`, ce qui rend les configurations directement comparables (sans
cela, l'arrêt naturel variable domine la mesure). C'est un choix de banc, pas la
réalité de production — en prod beaucoup de séquences s'arrêtent tôt.

Une invocation = UNE configuration (gpu_memory_utilization ne se change pas à
chaud). Le driver `bench_sweep.sh` boucle sur les configurations.

  BENCH_PROMPTS=8 BENCH_GPU_FRAC=0.55 BENCH_EAGER=1 python bench_tokens.py
"""
from __future__ import annotations

import json
import os
import time

CKPT = os.environ.get("SMOKE_CKPT", "ReliquaryForge/qwen3.5-2b-reliquary-v3")
REV = os.environ.get("SMOKE_REV", "5db7a1f55218c68fd3dff1d927b02c7508684c9e")

N_PROMPTS = int(os.environ.get("BENCH_PROMPTS", "8"))
M_ROLLOUTS = int(os.environ.get("BENCH_ROLLOUTS", "8"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "512"))
GPU_FRAC = float(os.environ.get("BENCH_GPU_FRAC", "0.55"))
EAGER = os.environ.get("BENCH_EAGER", "1") == "1"
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))

# MODE PRODUCTION (BENCH_PROD=1) : reproduit exactement la mesure lue dans les
# logs du mineur sur H200 (7521 tok/s), pour pouvoir COMPARER. Trois differences
# avec le mode controle :
#   - arret naturel des sequences (pas d'ignore_eos)
#   - budget de tokens de production (phase-1 BFT = 2048, pas 512)
#   - c'est vLLM lui-meme qui rapporte le debit (use_tqdm), comme dans les logs
# Le mode controle reste le bon pour COMPARER DES CONFIGS entre elles (longueur
# fixe = pas de variance d'arret) ; le mode prod sert a comparer DES CARTES.
PROD = os.environ.get("BENCH_PROD", "0") == "1"

# Deux parametres d'ORDONNANCEMENT (aucun impact numerique : ils ne changent ni
# les tokens produits ni les preuves, seulement combien de sequences vLLM
# accepte de traiter en meme temps).
#   max_model_len : 16384 par defaut alors que nos sequences font au plus
#     ~2340 tokens (293 de prompt max + 2048 generes). vLLM dimensionne sa
#     reserve KV sur cette valeur, donc la baisser doit liberer de la place.
#   max_num_seqs : plafond d'ordonnancement ; peut brider la concurrence
#     independamment de la memoire disponible.
MAX_MODEL_LEN = int(os.environ.get("BENCH_MAX_MODEL_LEN", "16384"))
MAX_NUM_SEQS = os.environ.get("BENCH_MAX_NUM_SEQS", "")

# Prefix caching : nos M rollouts partagent le MEME prefixe de prompt, donc le
# prefill est aujourd'hui recalcule M fois pour rien (8 x ~213 tokens en code).
# Reutilisation exacte de KV deja calcules => pas de changement numerique
# attendu, contrairement au fp8 ou a la quantification.
PREFIX_CACHE = os.environ.get("BENCH_PREFIX_CACHE", "0") == "1"

# BENCH_NO_FORCED=1 : genere SANS le processeur forced-seed (echantillonnage
# glouton nu). Les tokens ne sont alors PAS valides pour le subnet — c'est
# uniquement une mesure de COUT. L'ecart avec/sans donne le plafond du gain
# qu'on pourrait esperer en optimisant le processeur (pre-calcul des u_at,
# vectorisation du warp).
NO_FORCED = os.environ.get("BENCH_NO_FORCED", "0") == "1"
if PROD:
    MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "2048"))

# BENCH_ENV=opencodeinstruct|openmathinstruct tire de VRAIS prompts du dataset.
# Indispensable : mesuré le 2026-07-22, un prompt code fait ~224 tokens contre
# ~68 en maths (3,3x). Le prefill et le remplissage du cache KV en dépendent
# directement, donc un débit mesuré sur des prompts maths ne vaut PAS pour le
# code — et l'optimum de batch peut différer.
BENCH_ENV = os.environ.get("BENCH_ENV", "")

# Repli : prompts maths en dur (si aucun env demandé).
PROMPTS = [
    "Compute 17 * 23 step by step. Answer:",
    "What is the sum of the first 10 prime numbers? Explain briefly. Answer:",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? "
    "Show the reasoning in detail before answering. Answer:",
    "Simplify (x+1)^2 - (x-1)^2. Answer:",
    "If 3x + 7 = 22, solve for x and verify the result. Answer:",
    "A rectangle has perimeter 34 and area 60. Find its dimensions, showing "
    "each algebraic step and checking both constraints at the end. Answer:",
    "Convert 7/8 to a decimal and to a percentage. Answer:",
    "How many distinct arrangements are there of the letters in MISSISSIPPI? "
    "Justify the multinomial coefficient you use. Answer:",
]


def main() -> None:
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.miner.vllm_forced_seed import (
        FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        build_forced_seed_logitsproc_class,
    )

    local = snapshot_download(CKPT, revision=REV, allow_patterns=None)
    tok = load_tokenizer(local)

    t0 = time.perf_counter()
    llm = LLM(
        model=local,
        tokenizer=CKPT,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_FRAC,
        enforce_eager=EAGER,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"gdn_prefill_backend": "triton"},
        **({} if NO_FORCED
           else {"logits_processors": [build_forced_seed_logitsproc_class()]}),
        enable_prefix_caching=PREFIX_CACHE,
        **({"max_num_seqs": int(MAX_NUM_SEQS)} if MAX_NUM_SEQS else {}),
    )
    load_s = time.perf_counter() - t0

    # une séquence par (prompt, rollout), chacune avec ses propres extra_args
    if BENCH_ENV:
        from reliquary.environment import load_environment
        from reliquary.protocol.tokens import encode_prompt
        env = load_environment(BENCH_ENV)
        n = len(env)
        prompts_tokens = [
            encode_prompt(tok, env.get_problem((i * 9973) % n)["prompt"])
            for i in range(N_PROMPTS)
        ]
        L = [len(t) for t in prompts_tokens]
        print(f"[bench] env={BENCH_ENV} longueurs prompt: min={min(L)} "
              f"median={sorted(L)[len(L)//2]} max={max(L)}", flush=True)
    else:
        prompts_tokens = [
            tok(PROMPTS[i % len(PROMPTS)], return_tensors=None)["input_ids"]
            for i in range(N_PROMPTS)
        ]
    reqs, sps = [], []
    for pi, ptoks in enumerate(prompts_tokens):
        for r in range(M_ROLLOUTS):
            reqs.append(TokensPrompt(prompt_token_ids=ptoks))
            sps.append(SamplingParams(
                n=1, temperature=0.0, max_tokens=MAX_TOKENS,
                ignore_eos=not PROD,      # longueur fixe => configs comparables
                **({} if NO_FORCED else {"extra_args": {
                    FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness="ab" * 32, prompt_idx=1000 + pi,
                        checkpoint_hash="bench", rollout_index=r,
                        base_offset=0, start_len=len(ptoks))}}),
            ))

    n_seq = len(reqs)
    runs = []
    for it in range(REPEATS):
        t = time.perf_counter()
        outs = llm.generate(reqs, sampling_params=sps, use_tqdm=PROD)
        dur = time.perf_counter() - t
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        runs.append((dur, gen, gen / dur))
        print(f"[bench] run {it+1}/{REPEATS}: {dur:6.2f}s  {gen:7d} tok  "
              f"{gen/dur:8.1f} tok/s", flush=True)

    best = max(r[2] for r in runs)
    med = sorted(r[2] for r in runs)[len(runs) // 2]
    spread = (max(r[2] for r in runs) - min(r[2] for r in runs)) / med * 100
    print(json.dumps({
        "prompts": N_PROMPTS, "rollouts": M_ROLLOUTS, "sequences": n_seq,
        "max_tokens": MAX_TOKENS, "mode": "prod" if PROD else "controle", "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS or "defaut", "prefix_cache": PREFIX_CACHE, "forced_seed": not NO_FORCED, "gpu_frac": GPU_FRAC, "eager": EAGER, "env": BENCH_ENV or "math-hardcoded",
        "load_s": round(load_s, 1),
        "toks_per_s_median": round(med, 1),
        "toks_per_s_best": round(best, 1),
        "spread_pct": round(spread, 1),
        "wall_s_median": round(sorted(r[0] for r in runs)[len(runs) // 2], 2),
    }), flush=True)


if __name__ == "__main__":
    main()
