#!/usr/bin/env python3
"""Probe d'amorçage du prior v5 — étiquette le dataset au ck0, AVANT lancement.

Pendant que le validateur est down : N prompts opencode uniformes × M=16
rollouts forced-seed (pipeline réel v4, cap prod), grading parallèle (le
goulot = subprocess grader, pas le GPU), sorties au format SAMPLE_DUMP :
  - corpus COMPLET → ETUDE_OUT (entraînement prior v5.0, cible = score) ;
  - sous-ensemble score ≥ MEMO_SEED_MIN → MEMO_SEED (amorce la table de
    vedettes du mémo au boot du mineur : PayableMemo.load_jsonl lit ce format).
Labels au ck0 = frais au lancement (premier checkpoint v4 = ce modèle).
Reprise : les prompt_idx déjà présents dans ETUDE_OUT sont sautés.
Usage : RELIQUARY_PROTOCOL_VERSION=4 python ops/probe_prior_v5.py
Env : PROBE_N (3000), PROBE_CHUNK (16 prompts = 256 seqs), PROBE_ENV
(opencodeinstruct), ETUDE_OUT, MEMO_SEED, MEMO_SEED_MIN (0.23), PROBE_SEED (81).
"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

ENV_NAME = os.environ.get("PROBE_ENV", "opencodeinstruct")
N = int(os.environ.get("PROBE_N", "3000"))
CHUNK = int(os.environ.get("PROBE_CHUNK", "16"))
OUT = os.environ.get("ETUDE_OUT", f"/workspace/probe_ck0_{ENV_NAME}.jsonl")
MEMO_SEED = os.environ.get("MEMO_SEED", "/workspace/samples_v4.jsonl")
MEMO_MIN = float(os.environ.get("MEMO_SEED_MIN", "0.23"))
SEED = int(os.environ.get("PROBE_SEED", "81"))
RANDOMNESS = "cafe81" + "deadbeef" * 7 + "42"  # fixe : H1 = corr 0.824 inter-randomness


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

    assert c.PROTOCOL_VERSION >= 4
    M, cap = c.M_ROLLOUTS, c.MAX_NEW_TOKENS_PROTOCOL_CAP

    done = set()
    try:
        for line in open(OUT):
            try:
                done.add(json.loads(line)["prompt_idx"])
            except Exception:
                pass
    except FileNotFoundError:
        pass
    print(f"[probe] reprise : {len(done)} prompts déjà étiquetés", flush=True)

    local = snapshot_download(c.DEFAULT_BASE_MODEL,
                              revision=c.DEFAULT_BASE_MODEL_REVISION)
    tok = load_tokenizer(local)
    env = load_environment(ENV_NAME)
    n_env = len(env)
    rng = random.Random(SEED)
    idxs = [i for i in rng.sample(range(n_env), min(N * 2, n_env))
            if i not in done][:N]
    print(f"[probe] env={ENV_NAME} len={n_env} — {len(idxs)} prompts à "
          f"étiqueter, chunks de {CHUNK} ({CHUNK*M} seqs)", flush=True)

    llm = LLM(
        model=local, tokenizer=local, trust_remote_code=True, dtype="bfloat16",
        max_model_len=cap + 1024, gpu_memory_utilization=0.76,
        enforce_eager=False, disable_log_stats=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"gdn_prefill_backend": "triton"},
        logits_processors=[build_forced_seed_logitsproc_class()],
        max_num_seqs=256,
    )
    pool = ThreadPoolExecutor(max_workers=16)
    t_start, n_done = time.time(), 0

    for c0 in range(0, len(idxs), CHUNK):
        chunk = idxs[c0:c0 + CHUNK]
        problems = {i: env.get_problem(i) for i in chunk}
        reqs, sps, meta = [], [], []
        for i in chunk:
            ptoks = encode_prompt(tok, problems[i]["prompt"])
            for r in range(M):
                reqs.append(TokensPrompt(prompt_token_ids=ptoks))
                sps.append(SamplingParams(
                    n=1, temperature=0.0, max_tokens=cap, ignore_eos=False,
                    extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness=RANDOMNESS, prompt_idx=i,
                        checkpoint_hash="probe-ck0", rollout_index=r,
                        base_offset=0, start_len=len(ptoks))},
                ))
                meta.append((i, r))
        outs = llm.generate(reqs, sampling_params=sps, use_tqdm=False)
        texts = {}
        for (i, r), o in zip(meta, outs):
            texts.setdefault(i, []).append(
                (tok.decode(o.outputs[0].token_ids), len(o.outputs[0].token_ids)))
        # grading PARALLÈLE (subprocess-bound → threads OK)
        futs = {i: [pool.submit(env.compute_reward, problems[i], t)
                    for t, _ in lst] for i, lst in texts.items()}
        with open(OUT, "a", encoding="utf-8") as fh, \
             open(MEMO_SEED, "a", encoding="utf-8") as fm:
            for i, lst in texts.items():
                rewards = [float(f.result()) for f in futs[i]]
                lens = sorted(L for _, L in lst)
                mean = sum(rewards) / len(rewards)
                sigma = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
                score = sigma * (1.0 - mean)
                row = {
                    "prompt": problems[i]["prompt"], "prompt_idx": i,
                    "rewards": rewards, "sigma": round(sigma, 4),
                    "window_n": None, "ts": round(time.time(), 1),
                    "score": round(score, 4),
                    "k": sum(1 for x in rewards if x >= 0.5),
                    "checkpoint_n": 0, "protocol_version": 4, "cap": cap,
                    "source": "probe_ck0",
                    "in_zone": bool(sigma >= 0.24), "sigma_min": 0.24,
                    "env": ENV_NAME,
                    "n_truncated": sum(1 for L in lens if L >= cap),
                    "completion_lens": [int(x) for x in lens],
                }
                fh.write(json.dumps(row) + "\n")
                if score >= MEMO_MIN and row["n_truncated"] == 0:
                    fm.write(json.dumps(row) + "\n")
        n_done += len(chunk)
        rate = n_done / max(1.0, time.time() - t_start)
        print(f"[probe] {n_done}/{len(idxs)} groupes ({rate*3600:.0f}/h) — "
              f"ETA {int((len(idxs)-n_done)/max(rate,1e-9)/60)} min", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
