"""Mesure : longueur de la « queue » après réparation d'un EOS final refusé.

Pour chaque rollout du BENCH_DUMP dont l'EOS final n'est pas le pick de la
réplique : on remplace l'EOS par ce pick et on continue la génération forced-
seed avec la pile du validateur (forward complet [1, seq] par pas + lm_head
d'une ligne = exactement son calcul), jusqu'à un stop ou REPAIR_MAX pas.
À lancer dans venv_val avec PYTHONPATH upstream.
"""
import collections, json, os, sys, time
import torch
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, warp, u_at
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer, resolve_eos_token_ids

dump = json.load(open(sys.argv[1]))
MAX = int(os.environ.get("REPAIR_MAX", "400"))
model = load_text_generation_model(dump["model_path"], torch_dtype=torch.bfloat16,
                                   attn_implementation="flash_attention_2").to("cuda").eval()
eos = set(resolve_eos_token_ids(model, load_tokenizer(dump["model_path"])))
R, C = dump["randomness"], dump["checkpoint_hash"]

def row_for(tokens):
    with torch.no_grad():
        h, _ = forward_single_layer(model, torch.tensor([tokens], device="cuda"), None, -1,
                                    materialize_logits=False)
        return model.lm_head(h[0][len(tokens) - 1])   # prédit le token suivant

tails, times, outcomes = [], [], collections.Counter()
for row in dump["rows"]:
    toks, plen, r, pidx = row["tokens"], row["prompt_len"], row["rollout"], row["prompt_idx"]
    j = len(toks) - 1 - plen
    probs = warp(row_for(toks[:-1]).float(), t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    p0 = pick(probs, u_at(R, pidx, C, r, j))
    if p0 == toks[-1]:
        continue
    t0 = time.time()
    seq = toks[:-1] + [p0]
    n = 0
    while n < MAX:
        jj = len(seq) - plen
        probs = warp(row_for(seq).float(), t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        tk = pick(probs, u_at(R, pidx, C, r, jj))
        seq.append(tk); n += 1
        if tk in eos:
            break
    outcomes["eos" if seq[-1] in eos else "max"] += 1
    tails.append(n); times.append(time.time() - t0)
tails.sort(); times.sort()
q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None
print("[tail] n=%d outcomes=%s queue(tokens) p50=%s p90=%s max=%s | temps p50=%.2fs p90=%.2fs max=%.2fs | %.1f ms/pas"
      % (len(tails), dict(outcomes), q(tails, .5), q(tails, .9), tails[-1] if tails else None,
         q(times, .5), q(times, .9), times[-1], 1000 * sum(times) / max(1, sum(tails))), flush=True)
