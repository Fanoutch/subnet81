"""Mesure : génération « mineur de référence » (HF generate en lot de 16,
flash-attn 2, pile du validateur) — accord de l'EOS final avec la vérification
exacte du validateur (forward complet + lm_head ligne), et temps par groupe.
À lancer dans venv_val avec PYTHONPATH upstream."""
import collections, json, os, sys, time
import torch
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor, forced_seed_generate_kwargs
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer, resolve_eos_token_ids, first_eos_index
from reliquary.validator.verifier import _gpu_terminal_forced_pick_diagnostics
from reliquary.environment.forced_sampling import u_at

dump = json.load(open(sys.argv[1]))
NP = int(os.environ.get("NPROMPTS", "4")); MAXNEW = int(os.environ.get("MAXNEW", "2048"))
path = dump["model_path"]
model = load_text_generation_model(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
tok = load_tokenizer(path)
eos = sorted(resolve_eos_token_ids(model, tok))
R, C = dump["randomness"], dump["checkpoint_hash"]
prompts = {}
for row in dump["rows"]:
    prompts.setdefault(row["prompt_idx"], row["tokens"][:row["prompt_len"]])
n = ok = eos_end = 0; groups_ok = 0; times = []
for pidx, ptoks in list(prompts.items())[:NP]:
    plen = len(ptoks)
    t0 = time.time()
    with torch.no_grad():
        inp = torch.tensor([ptoks] * 16, device="cuda")
        proc = ForcedSeedLogitsProcessor(randomness=R, hotkey="x", prompt_idx=pidx, checkpoint_hash=C,
                                         rollout_indices=list(range(16)), base_offsets=[0] * 16, start_len=plen)
        out = model.generate(inp, **forced_seed_generate_kwargs(
            {"max_new_tokens": MAXNEW, "pad_token_id": eos[0], "attention_mask": torch.ones_like(inp), "eos_token_id": eos}, proc))
    times.append(time.time() - t0)
    gok = True
    for r in range(16):
        gen = out[r].tolist()[plen:]
        fe = first_eos_index(gen, eos)
        if fe is None:
            continue
        toks = ptoks + gen[:fe + 1]
        eos_end += 1
        with torch.no_grad():
            h, _ = forward_single_layer(model, torch.tensor([toks], device="cuda"), None, -1, materialize_logits=False)
        hh = h[0]
        class Rows:
            def __getitem__(self, k):
                with torch.no_grad():
                    return model.lm_head(hh[k])
        us = [u_at(R, pidx, C, r, j) for j in range(len(toks) - plen)]
        v, _ = _gpu_terminal_forced_pick_diagnostics(Rows(), toks, plen, len(toks), set(eos), us, (0, 0))
        n += 1; ok += int(bool(v)); gok = gok and bool(v)
    groups_ok += int(gok)
print("[hfgen] prompts=%d rollouts_eos=%d terminal_exact=%.4f groups_all_ok=%d temps_groupe=%s"
      % (NP, n, ok / max(1, n), groups_ok, [round(t, 1) for t in times]), flush=True)
