#!/usr/bin/env python3
"""Étude offline H1/H2 (+boxing/longueurs/H5) sur box GPU — etudev4.md §A.

Sans validateur : N prompts OMI réels (manifest v4 train-*), générés DEUX fois
(randomness A/B = deux « fenêtres ») × M=16 rollouts forced-seed, gradés au
contrat v4 (MATH_ANSWER_FORMAT=boxed). Sort un JSONL + l'analyse inline :
  - H2 : distribution de k, part payable k∈[1,15], TAUX DE BOXING SPONTANÉ
    (métrique n°1 du runbook — connue AVANT le jour J) ;
  - H1 : répétabilité k_A↔k_B (corrélation, |Δk| médian, P(payable→payable)) ;
  - longueurs naturelles (médiane/p90/cap-hits) — thermostat du cap 8192 ;
  - H5 : si ETUDE_PREDICTOR pointe un json v4.3, AUC(score → k>=1) sur ces labels.
Usage (box) : RELIQUARY_PROTOCOL_VERSION=4 python ops/etude_h1_h2_offline.py
Env : ETUDE_ENV (opencodeinstruct — NOTRE env de prod ; openmathinstruct pour
trancher « ouvrir le math ou pas »), ETUDE_N (64), ETUDE_OUT, ETUDE_MAX_TOKENS
(cap protocole), ETUDE_PREDICTOR (chemin json, optionnel — v4.3 est entraîné
sur CODE, l'AUC n'a de sens que sur ETUDE_ENV=opencodeinstruct).
Le grader code local = subprocess pur (sandbox builtins verbatim validateur,
pas de runsc) → tourne sur n'importe quelle box.
"""
import json
import os
import time

ENV_NAME = os.environ.get("ETUDE_ENV", "opencodeinstruct")
N = int(os.environ.get("ETUDE_N", "64"))
OUT = os.environ.get("ETUDE_OUT", f"/workspace/etude_h1h2_{ENV_NAME}.jsonl")
PRED_PATH = os.environ.get("ETUDE_PREDICTOR", "")
GPU_FRAC = float(os.environ.get("BENCH_GPU_FRAC", "0.76"))
RANDS = {"A": "deadbeef" * 8, "B": "cafebabe" * 8}


def main() -> None:
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from reliquary import constants as c
    from reliquary.environment import load_environment
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.miner.vllm_forced_seed import (
        FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        build_forced_seed_logitsproc_class,
    )

    assert c.PROTOCOL_VERSION >= 4, "lancer avec RELIQUARY_PROTOCOL_VERSION=4"
    M = c.M_ROLLOUTS
    cap = int(os.environ.get("ETUDE_MAX_TOKENS", str(c.MAX_NEW_TOKENS_PROTOCOL_CAP)))

    local = snapshot_download(c.DEFAULT_BASE_MODEL,
                              revision=c.DEFAULT_BASE_MODEL_REVISION)
    tok = load_tokenizer(local)
    env = load_environment(ENV_NAME)
    n_env = len(env)
    print(f"[etude] env={ENV_NAME} len={n_env}  (H8)", flush=True)

    idxs = [(i * 104729) % n_env for i in range(N)]  # dispersé, déterministe
    problems = {i: env.get_problem(i) for i in idxs}
    ptoks = {i: encode_prompt(tok, problems[i]["prompt"]) for i in idxs}

    llm = LLM(
        model=local, tokenizer=local, trust_remote_code=True, dtype="bfloat16",
        max_model_len=cap + 1024, gpu_memory_utilization=GPU_FRAC,
        enforce_eager=False, disable_log_stats=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"gdn_prefill_backend": "triton"},
        logits_processors=[build_forced_seed_logitsproc_class()],
        max_num_seqs=256,
    )

    rows = []
    with open(OUT, "w", encoding="utf-8") as fh:
        for tag, randomness in RANDS.items():
            reqs, sps, meta = [], [], []
            for i in idxs:
                for r in range(M):
                    reqs.append(TokensPrompt(prompt_token_ids=ptoks[i]))
                    sps.append(SamplingParams(
                        n=1, temperature=0.0, max_tokens=cap, ignore_eos=False,
                        extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                            randomness=randomness, prompt_idx=i,
                            checkpoint_hash="etude-v4", rollout_index=r,
                            base_offset=0, start_len=len(ptoks[i]))},
                    ))
                    meta.append((i, r))
            t0 = time.perf_counter()
            outs = llm.generate(reqs, sampling_params=sps, use_tqdm=False)
            dt = time.perf_counter() - t0
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            print(f"[etude] fenetre {tag}: {len(reqs)} seqs, {ntok} tok, "
                  f"{dt:.0f}s ({ntok/dt:.0f} tok/s)", flush=True)

            by_prompt = {}
            for (i, r), o in zip(meta, outs):
                text = tok.decode(o.outputs[0].token_ids)
                by_prompt.setdefault(i, []).append(
                    (float(env.compute_reward(problems[i], text)),
                     len(o.outputs[0].token_ids),
                     ("\\boxed" in text or "\\fbox" in text)
                     if ENV_NAME == "openmathinstruct"
                     else ("```" in text or "def " in text)))
            for i, lst in by_prompt.items():
                rewards = [x[0] for x in lst]
                lens = sorted(x[1] for x in lst)
                mean = sum(rewards) / len(rewards)
                sigma = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
                row = {
                    "window": tag, "prompt_idx": i,
                    "k": sum(1 for x in rewards if x >= 0.5),
                    "sigma": round(sigma, 4),
                    # zone v4 = σ≥0.24 (générique binaire ET continu code)
                    "payable": bool(sigma >= 0.24),
                    "score": round(sigma * (1.0 - mean), 4),
                    "rewards": rewards, "boxed": sum(1 for x in lst if x[2]),
                    "len_med": lens[len(lens) // 2], "len_max": lens[-1],
                    "cap_hits": sum(1 for L in lens if L >= cap),
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")

    # ── analyse inline ──────────────────────────────────────────────────────
    A = {r["prompt_idx"]: r for r in rows if r["window"] == "A"}
    B = {r["prompt_idx"]: r for r in rows if r["window"] == "B"}
    all_rows = rows
    ks = [r["k"] for r in all_rows]
    boxed_frac = sum(r["boxed"] for r in all_rows) / (len(all_rows) * M)
    payable = [r for r in all_rows if r["payable"]]
    print("\n===== H2 — env=%s, distribution de k (2 fenetres x %d prompts) ====="
          % (ENV_NAME, N))
    from collections import Counter
    print("  k:", dict(sorted(Counter(ks).items())))
    scores = sorted(r["score"] for r in all_rows)
    print(f"  part payable (σ≥0.24) : {len(payable)}/{len(all_rows)} "
          f"= {len(payable)/len(all_rows):.1%}")
    print(f"  score d'enchère : médiane {scores[len(scores)//2]:.3f}, "
          f"p90 {scores[int(len(scores)*0.9)]:.3f}, max {scores[-1]:.3f}")
    print(f"  format OK ({'\\boxed' if ENV_NAME=='openmathinstruct' else 'code'}) "
          f"par rollout : {boxed_frac:.1%}")
    lens_med = sorted(r["len_med"] for r in all_rows)
    print(f"  longueur mediane des medianes : {lens_med[len(lens_med)//2]} tok ; "
          f"cap-hits {sum(r['cap_hits'] for r in all_rows)}/{len(all_rows)*M}")
    print("\n===== H1 — repetabilite k (A vs B, meme prompt, randomness differente) =====")
    common = sorted(set(A) & set(B))
    if common:
        da = [A[i]["k"] for i in common]
        db = [B[i]["k"] for i in common]
        dk = sorted(abs(a - b) for a, b in zip(da, db))
        pay_a = [i for i in common if A[i]["payable"]]
        p2p = (sum(1 for i in pay_a if B[i]["payable"]) / len(pay_a)
               if pay_a else float("nan"))
        n = len(common)
        ma, mb = sum(da) / n, sum(db) / n
        cov = sum((a - ma) * (b - mb) for a, b in zip(da, db)) / n
        va = sum((a - ma) ** 2 for a in da) / n
        vb = sum((b - mb) ** 2 for b in db) / n
        corr = cov / ((va * vb) ** 0.5) if va > 0 and vb > 0 else float("nan")
        print(f"  n={n}  corr(k_A,k_B)={corr:.3f}  |dk| median={dk[n//2]}  "
              f"P(payable->payable)={p2p:.1%}  (baseline v3 binaire: 51%)")
    if PRED_PATH and os.path.exists(PRED_PATH):
        print("\n===== H5 — transfert predicteur v4.3 sur labels v4 =====")
        from reliquary.miner import prompt_predictor as pp
        model = pp.load_model(PRED_PATH) if hasattr(pp, "load_model") else json.load(open(PRED_PATH))
        def _auc(labelled):
            labelled.sort()
            pos = sum(l for _, l in labelled)
            neg = len(labelled) - pos
            if not pos or not neg:
                return None, pos, neg
            rank_sum = sum(i for i, (_, l) in enumerate(labelled, 1) if l)
            return (rank_sum - pos * (pos + 1) / 2) / (pos * neg), pos, neg

        scored = []
        for r in all_rows:
            try:
                s = pp.score_prompt(model, problems[r["prompt_idx"]]["prompt"])
                scored.append((s, r))
            except Exception as e:
                print("  score_prompt KO:", e); break
        if scored:
            p75 = sorted(x["score"] for _, x in scored)[int(len(scored) * 0.75)]
            for name, lab in (
                ("payable σ≥0.24", lambda r: 1 if r["payable"] else 0),
                (f"vedette score≥p75({p75:.3f})", lambda r: 1 if r["score"] >= p75 else 0),
            ):
                auc, pos, neg = _auc([(s, lab(r)) for s, r in scored])
                if auc is None:
                    print(f"  AUC({name}) : labels dégénérés (pos={pos}, neg={neg})")
                else:
                    print(f"  AUC(score v4.3 -> {name}) = {auc:.3f}  (0.5 = hasard)")
    print(f"\n[etude] JSONL: {OUT} ({len(all_rows)} groupes)")
    print("ETUDE_DONE")


if __name__ == "__main__":
    main()
