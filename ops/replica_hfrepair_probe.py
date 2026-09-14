"""Mesure : réparation HYBRIDE — pour les rollouts vLLM dont l'EOS final est
refusé par la réplique, régénérer la fin avec HF generate (pile validateur),
en lot par prompt avec padding à gauche, à partir du préfixe tokens[:-1].
Vérifie l'EOS final obtenu avec le calcul exact du validateur ; mesure le temps.
venv_val + PYTHONPATH upstream."""
import collections, json, os, sys, time
import torch
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor, forced_seed_generate_kwargs
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer, resolve_eos_token_ids, first_eos_index
from reliquary.validator.verifier import _gpu_terminal_forced_pick_diagnostics
from reliquary.environment.forced_sampling import u_at

dump = json.load(open(sys.argv[1]))
MAXNEW = int(os.environ.get("MAXNEW", "1536"))
path = dump["model_path"]
model = load_text_generation_model(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
tok = load_tokenizer(path)
eos = sorted(resolve_eos_token_ids(model, tok)); pad = eos[0]
R, C = dump["randomness"], dump["checkpoint_hash"]

def verdict(toks, plen, r, pidx):
    with torch.no_grad():
        h, _ = forward_single_layer(model, torch.tensor([toks], device="cuda"), None, -1, materialize_logits=False)
    hh = h[0]
    class Rows:
        def __getitem__(self, k):
            with torch.no_grad():
                return model.lm_head(hh[k])
    us = [u_at(R, pidx, C, r, j) for j in range(len(toks) - plen)]
    return _gpu_terminal_forced_pick_diagnostics(Rows(), toks, plen, len(toks), set(eos), us, (0, 0))[0]

fails = collections.defaultdict(list)
t_check = time.time()
for row in dump["rows"]:
    if verdict(row["tokens"], row["prompt_len"], row["rollout"], row["prompt_idx"]) is False:
        fails[row["prompt_idx"]].append(row)
t_check = time.time() - t_check
n = ok = trunc = 0; times = []
for pidx, rows in fails.items():
    prefixes = [r["tokens"][:-1] for r in rows]
    L = max(len(p) for p in prefixes)
    ids = torch.tensor([[pad] * (L - len(p)) + p for p in prefixes], device="cuda")
    mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in prefixes], device="cuda")
    offs = [len(p) - r["prompt_len"] for p, r in zip(prefixes, rows)]
    t0 = time.time()
    with torch.no_grad():
        proc = ForcedSeedLogitsProcessor(randomness=R, hotkey="x", prompt_idx=pidx, checkpoint_hash=C,
                                         rollout_indices=[r["rollout"] for r in rows], base_offsets=offs, start_len=L)
        out = model.generate(ids, **forced_seed_generate_kwargs(
            {"max_new_tokens": MAXNEW, "pad_token_id": pad, "attention_mask": mask, "eos_token_id": eos}, proc))
    times.append(time.time() - t0)
    for i, r in enumerate(rows):
        gen = out[i].tolist()[L:]
        fe = first_eos_index(gen, eos)
        if fe is None:
            trunc += 1; continue
        toks = prefixes[i] + gen[:fe + 1]
        n += 1; ok += int(bool(verdict(toks, r["prompt_len"], r["rollout"], pidx)))
print("[hfrepair] refusés=%d réparés_eos=%d exacts=%d tronqués@%d=%d | temps lots=%s | vérif 191 rollouts=%.1fs"
      % (sum(len(v) for v in fails.values()), n, ok, MAXNEW, trunc, [round(t, 1) for t in times], t_check), flush=True)
