"""Comment vLLM termine-t-il vraiment ? — ignore_eos ON vs OFF, mesuré.

À lancer sur une box GPU AVANT de figer la config de terminaison. Répond à
trois questions qu'on ne peut pas trancher sans mesure (on s'est trompé 3 fois
en raisonnant sans elle) :

  1. Sans ignore_eos, vLLM RETIRE-t-il le token d'arrêt des token_ids ?
     (c'est ce qui a causé les 13 premiers bad_termination)
  2. Avec ignore_eos, la génération continue-t-elle APRÈS l'EOS du modèle ?
     (= GPU gaspillé + EOS au milieu, les 8 derniers bad_termination)
  3. Combien de tokens sont gaspillés en moyenne dans ce cas ?

Verdict attendu : si (1) est faux — l'EOS reste dans la sortie sans
ignore_eos — alors ignore_eos est INUTILE et doit être retiré (il ne fait plus
que gaspiller du GPU). Si (1) est vrai, on le garde et la restauration via
stop_reason reste nécessaire.

  cd /workspace/reliquary-miner-priv && HF_HOME=/workspace/hf PYTHONPATH=. \
    RELIQUARY_VLLM_FORCED_SEED=1 VLLM_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_HOME=/workspace/venv/lib/python3.12/site-packages/nvidia/cu13 \
    /workspace/venv/bin/python /workspace/probe_eos_behavior.py
"""
from __future__ import annotations

import os
import sys

CKPT = os.environ.get("PROBE_CKPT", "ReliquaryForge/qwen3.5-4b-reliquary-v4")
GPU_ID = int(os.environ.get("PROBE_GPU_ID", "1"))       # 1 = libre si le miner tourne sur 0
GPU_FRAC = float(os.environ.get("PROBE_GPU_FRAC", "0.35"))
MAX_TOKENS = int(os.environ.get("PROBE_MAX_TOKENS", "2600"))
N_PROMPTS = int(os.environ.get("PROBE_N_PROMPTS", "8"))


def main() -> None:
    from huggingface_hub import snapshot_download
    from reliquary.miner.vllm_backend import VLLMBackend
    from reliquary.shared.modeling import load_tokenizer, resolve_eos_token_ids
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.environment import load_environment
    from reliquary.miner.engine import validator_termination_ok

    local = snapshot_download(CKPT, allow_patterns=None)
    tok = load_tokenizer(CKPT)
    eos_ids = sorted(resolve_eos_token_ids(None, tok)) or [248044, 248046]
    print(f"[probe] EOS résolus: {eos_ids}", flush=True)

    env = load_environment("opencodeinstruct")
    n = len(env)
    prompts = [
        encode_prompt(tok, env.get_problem((i * 9973) % n)["prompt"])
        for i in range(N_PROMPTS)
    ]

    backend = VLLMBackend(
        model_path=local, gpu_id=GPU_ID, gpu_memory_utilization=GPU_FRAC,
        max_model_len=4096, tokenizer_path=CKPT, forced_seed=True,
    )

    import vllm
    real_sp = vllm.SamplingParams

    def patched(drop_ignore: bool):
        def factory(**kw):
            if drop_ignore:
                kw.pop("ignore_eos", None)
            return real_sp(**kw)
        return factory

    # 3 configurations. La 3e est la clé : arrêt PUREMENT naturel (l'EOS du
    # modèle stoppe, sans passer par stop_token_ids). C'est la seule qui ne
    # gaspille rien — encore faut-il qu'elle CONSERVE le token, sinon la
    # troncature n'a rien à couper (= le bug des fenêtres 27585/27586/27590).
    configs = (
        ("ignore_eos=ON  + stop_ids", False, True),
        ("ignore_eos=OFF + stop_ids", True, True),
        ("ignore_eos=OFF, SANS stop_ids (arrêt naturel pur)", True, False),
    )
    for label, drop, with_stop in configs:
        vllm.SamplingParams = patched(drop)
        groups = backend.generate_forced_phase1_multi(
            prompts, prompt_indices=list(range(1000, 1000 + len(prompts))),
            randomness="ab" * 32, checkpoint_hash="probe",
            m_rollouts=2, max_tokens=MAX_TOKENS,
            stop_token_ids=eos_ids if with_stop else None,
        )
        flat = [r for g in groups for r in g]
        eset = set(eos_ids)
        n_any = sum(1 for r in flat if any(t in eset for t in r))
        n_last = sum(1 for r in flat if r and r[-1] in eset)
        n_mid = sum(
            1 for r in flat
            if any(t in eset for t in r[:-1])
        )
        n_ok = sum(1 for r in flat if validator_termination_ok(r, eset))
        n_cap = sum(1 for r in flat if len(r) >= MAX_TOKENS)
        # gaspillage : tokens générés APRÈS le premier EOS
        waste = []
        for r in flat:
            for i, t in enumerate(r):
                if t in eset:
                    waste.append(len(r) - (i + 1))
                    break
        avg_waste = sum(waste) / len(waste) if waste else 0.0
        lens = sorted(len(r) for r in flat)
        print(
            f"\n=== {label} ({len(flat)} rollouts) ===\n"
            f"  contient un EOS      : {n_any}/{len(flat)}\n"
            f"  EOS en DERNIERE pos  : {n_last}/{len(flat)}\n"
            f"  EOS au MILIEU        : {n_mid}/{len(flat)}   <- cause bad_termination\n"
            f"  passe le validateur  : {n_ok}/{len(flat)}\n"
            f"  a atteint le cap     : {n_cap}/{len(flat)}\n"
            f"  tokens GASPILLES apres le 1er EOS : moyenne {avg_waste:.0f}\n"
            f"  longueurs: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}",
            flush=True,
        )

    vllm.SamplingParams = real_sp
    print(
        "\n[verdict] CHOISIR la config qui satisfait les DEUX criteres :\n"
        "  (a) CORRECTION : 'passe le validateur' ~= 'contient un EOS'\n"
        "      (si l'EOS est retire, la troncature n'a rien a couper => on\n"
        "       retombe sur les bad_termination des fenetres 27585/27586)\n"
        "  (b) ECONOMIE   : 'tokens GASPILLES apres le 1er EOS' ~= 0\n"
        "      (tout token genere apres la fin est jete = GPU brule, et sous\n"
        "       le tie-break v3 le debit = revenu)\n"
        "  La 3e config (arret naturel pur) est la candidate ideale : si elle\n"
        "  satisfait (a), elle gagne — c'est la seule sans gaspillage.\n"
        "  Si elle echoue (a), garder ignore_eos=ON + restauration + troncature\n"
        "  et accepter le gaspillage mesure en (b).",
        flush=True,
    )


if __name__ == "__main__":
    main()
