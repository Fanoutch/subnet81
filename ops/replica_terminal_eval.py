"""Évaluation « réplique validateur » de l'EOS final (à lancer dans le venv
réplique : torch 2.7.0+cu128, transformers 5.10.4, flash-attn 2.8.3, avec
PYTHONPATH sur le code upstream origin/main).

Rejoue EXACTEMENT la chaîne du validateur (verifier.py 0a69244) sur les
rollouts vLLM sauvegardés par ``bench_terminal_pick.py`` (BENCH_DUMP) :
forward ``[1, seq_len]`` sans masque via ``forward_single_layer``, projection
``lm_head`` de la seule ligne terminale, ``_gpu_terminal_forced_pick_
diagnostics``. Compare au verdict de notre garde locale (HF sdpa, 5.9.0).
"""
import json
import os
import sys

import torch


def main() -> None:
    from reliquary.environment.forced_sampling import u_at
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.modeling import (
        load_text_generation_model, load_tokenizer, resolve_eos_token_ids,
    )
    from reliquary.validator.verifier import _gpu_terminal_forced_pick_diagnostics
    import transformers

    dump = json.load(open(sys.argv[1]))
    local = dump["model_path"]
    impl = os.environ.get("REPLICA_ATTN", "flash_attention_2")
    model = load_text_generation_model(
        local, torch_dtype=torch.bfloat16, attn_implementation=impl,
    ).to("cuda").eval()
    tok = load_tokenizer(local)
    eos = resolve_eos_token_ids(model, tok)
    lm_head = model.lm_head
    rand, ckpt = dump["randomness"], dump["checkpoint_hash"]

    tab = {"both_ok": 0, "both_fail": 0, "local_ok_val_fail": 0,
           "local_fail_val_ok": 0}
    n = n_ok = 0
    groups: dict = {}
    for row in dump["rows"]:
        toks, plen, r = row["tokens"], row["prompt_len"], row["rollout"]
        with torch.no_grad():
            h, _ = forward_single_layer(
                model, torch.tensor([toks], device="cuda"), None, -1,
                materialize_logits=False)
        h = h[0]

        class _Rows:
            def __getitem__(self, key):
                with torch.no_grad():
                    return lm_head(h[key])

        us = [u_at(rand, row["prompt_idx"], ckpt, r, j)
              for j in range(len(toks) - plen)]
        ok, miss = _gpu_terminal_forced_pick_diagnostics(
            _Rows(), toks, plen, len(toks), set(eos), us, (0, 0))
        if ok is None:
            continue
        n += 1
        n_ok += int(bool(ok))
        groups.setdefault(row["prompt_idx"], []).append(bool(ok))
        key = ("both_ok" if ok and row["local_ok"] else
               "both_fail" if not ok and not row["local_ok"] else
               "local_ok_val_fail" if not ok else "local_fail_val_ok")
        tab[key] += 1
    print("[replica] RESULT " + json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "attn": impl, "rollouts_eos": n,
        "terminal_exact_replica": round(n_ok / n, 4) if n else None,
        "groups_all_ok": sum(all(v) for v in groups.values()),
        "groups": len(groups), "cross_tab_vs_local_sdpa": tab,
    }), flush=True)


if __name__ == "__main__":
    main()
