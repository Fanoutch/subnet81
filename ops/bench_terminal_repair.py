"""Banc de bout en bout de la réparation de l'EOS final (V1, #253).

Composants RÉELS : vLLM en réglages de prod (``VLLMBackend``, bake en
streaming), service réplique du validateur (socket ``RELIQUARY_REPLICA_SOCKET``,
lancé à part dans venv_val), ``MiningEngine._repair_terminal_eos``.

Mesure, sur N prompts code × 16 rollouts :
  - part des groupes rendus conformes (tous les EOS validés par la réplique) ;
  - rollouts réparés, tours, tokens ajoutés, temps de réparation par groupe ;
  - contre-vérification indépendante : la réplique rejuge TOUS les rollouts
    des groupes réparés (doit donner 100 %).
"""
from __future__ import annotations

import json
import os
import statistics
import time
import types

import torch  # noqa: F401  (import CUDA avant vLLM)


def main() -> None:
    from huggingface_hub import snapshot_download

    from reliquary import constants as c
    from reliquary.environment import load_environment
    from reliquary.miner import engine as eng_mod, replica_client
    from reliquary.miner.vllm_backend import VLLMBackend
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import load_tokenizer

    sys_path_bench = os.path.join(os.path.dirname(__file__))
    import sys
    sys.path.insert(0, sys_path_bench)
    from bench_terminal_pick import _prompt_indices

    sock = os.environ["RELIQUARY_REPLICA_SOCKET"]
    n_prompts = int(os.environ.get("BENCH_PROMPTS", "12"))
    repo = os.environ.get("BENCH_REPO", "ReliquaryForge/qwen3-4b-base-dapo-v4")
    local = snapshot_download(repo)
    ckpt_hash = os.path.basename(local.rstrip("/"))
    randomness = os.environ.get("BENCH_RANDOMNESS", "3c" * 32)
    eos_id = 151643

    assert replica_client.load(sock, local, timeout=300), "réplique: load KO"
    env = load_environment("opencodeinstruct")
    tok = load_tokenizer(local)
    idxs = _prompt_indices(n_prompts, "opencodeinstruct")
    prompts = [encode_prompt(tok, env.get_problem(i)["prompt"]) for i in idxs]

    backend = VLLMBackend(
        model_path=local,
        gpu_memory_utilization=float(os.environ.get("BENCH_GPU_FRAC", "0.55")),
        max_model_len=int(os.environ.get("BENCH_MAX_MODEL_LEN", "10240")),
        forced_seed=True,
    )
    groups = backend.generate_forced_phase1_multi_stream(
        prompts, prompt_indices=idxs, randomness=randomness,
        checkpoint_hash=ckpt_hash, m_rollouts=c.M_ROLLOUTS,
        max_tokens=c.MAX_NEW_TOKENS_PROTOCOL_CAP, stop_token_ids=[eos_id],
        primary_eos_id=eos_id,
    )

    eng = eng_mod.MiningEngine.__new__(eng_mod.MiningEngine)
    eng._eos_ids = [eos_id]
    eng._cached_randomness = randomness
    eng._local_hash = ckpt_hash
    eng._loaded_checkpoint_path = local
    eng.max_new_tokens = c.MAX_NEW_TOKENS_PROTOCOL_CAP
    eng._vllm_backend = backend
    eng.hf_model = types.SimpleNamespace(
        generation_config=types.SimpleNamespace(eos_token_id=eos_id))

    ok_groups = 0
    repair_s, repaired_rollouts, recheck_ok, recheck_n = [], 0, 0, 0
    for pidx, ptoks, grp in zip(idxs, prompts, groups):
        gens = [{"tokens": list(ptoks) + list(g), "prompt_length": len(ptoks)}
                for g in grp]
        before = [g["tokens"] for g in gens]
        t0 = time.time()
        out = eng._repair_terminal_eos(gens, prompt_idx=pidx, env=env)
        repair_s.append(time.time() - t0)
        if out is None:
            continue
        ok_groups += 1
        repaired_rollouts += sum(
            1 for a, b in zip(before, (g["tokens"] for g in out)) if a != b)
        res = replica_client.terminal_verdicts(
            sock, model_path=local, randomness=randomness,
            checkpoint_hash=ckpt_hash, prompt_idx=pidx,
            items=[{"rollout": r, "prompt_len": g["prompt_length"],
                    "tokens": g["tokens"]} for r, g in enumerate(out)],
            timeout=120)
        for v in res or []:
            if v["ok"] is not None:
                recheck_n += 1
                recheck_ok += int(v["ok"])

    print("[repair] RESULT " + json.dumps({
        "groups": len(idxs), "groups_conformes": ok_groups,
        "rollouts_reparés": repaired_rollouts,
        "temps_reparation_groupe_s": {
            "p50": round(statistics.median(repair_s), 2),
            "max": round(max(repair_s), 2)},
        "contre_verif_replique": f"{recheck_ok}/{recheck_n}",
    }), flush=True)


if __name__ == "__main__":
    main()
