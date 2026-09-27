"""Banc du mineur corpus (27/09) : débit batché + auto-audit TOPLOC.

Phase gen   : générateur d'UPSTREAM (reliquary.miner.corpus_miner.VllmGenerator),
              sampling EXACT du job, vrais prompts du job (rendered_prompt tirés
              de R2). Lot de K prompts × n=4 en un seul appel vLLM.
              BENCH_K = liste de tailles de lot, ex. "1,16".
Phase check : chargeur et vérification du VALIDATEUR corpus
              (load_text_only_model + batch_completion_hidden_states +
              audit_completion). PASS = 100 % des complétions passent.
"""
import json
import os
import sys
import time

MODEL_DIR = os.environ["BENCH_MODEL"]
OUT = os.environ.get("BENCH_OUT", "/workspace/corpus_bench_gen.json")
PROMPTS = os.environ.get("BENCH_PROMPTS", "/workspace/corpus_prompts.json")
JOB = {"max_new_tokens": 32768, "min_new_tokens": 16, "n": 4,
       "temperature": 1.0, "top_k": 20, "top_p": 0.95}
EOS = 248046


class _Sampling:
    def __init__(self, d):
        self.__dict__.update(d)


def gen():
    from transformers import AutoTokenizer
    from reliquary.miner.corpus_miner import VllmGenerator
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = json.load(open(PROMPTS))
    ids = [tok.encode(p["rendered_prompt"], add_special_tokens=False) for p in prompts]
    frac = float(os.environ.get("BENCH_GPU_FRACTION", "0.9"))
    g = VllmGenerator(MODEL_DIR, _Sampling(JOB), PROOF, EOS, gpu_memory_utilization=frac)
    llm, params = g._llm, g._params
    from vllm.inputs import TokensPrompt
    from reliquary.miner.vllm_hidden_capture import completion_rows
    from reliquary.protocol.toploc_proof import build_chunk_proofs
    import base64

    out = {"runs": []}
    pos = 0
    for k in [int(x) for x in os.environ.get("BENCH_K", "1,16").split(",")]:
        batch = ids[pos:pos + k]
        pos += k
        reqs = [TokensPrompt(prompt_token_ids=p) for p in batch for _ in range(JOB["n"])]
        t0 = time.time()
        res = llm.generate(reqs, params)
        wall = time.time() - t0
        comps, preempt = [], 0
        for i, o in enumerate(res):
            p = batch[i // JOB["n"]]
            toks = list(o.outputs[0].token_ids)
            try:
                rows = completion_rows(g._capture.pop(o.request_id), len(p), len(p) + len(toks))
            except (ValueError, KeyError):
                preempt += 1
                continue
            proofs = build_chunk_proofs(rows, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
            comps.append({"prompt_ids": p, "tokens": toks,
                          "proofs": [base64.b64encode(x).decode() for x in proofs]})
        ntok = sum(len(c["tokens"]) for c in comps)
        lens = sorted(len(c["tokens"]) for c in comps)
        run = {"k": k, "seqs": len(reqs), "wall_s": round(wall, 1), "tokens": ntok,
               "tok_s": round(ntok / wall, 1), "preempted": preempt,
               "len_med": lens[len(lens) // 2] if lens else 0, "len_max": lens[-1] if lens else 0,
               "ended_eos": sum(1 for c in comps if c["tokens"] and c["tokens"][-1] == EOS)}
        print("[gen]", json.dumps(run), flush=True)
        out["runs"].append({**run, "completions": comps})
    json.dump(out, open(OUT, "w"))
    print("GEN_DONE", flush=True)


def check():
    import torch
    from reliquary.shared.modeling import load_text_only_model
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from reliquary.validator.corpus_audit import batch_completion_hidden_states, audit_completion

    data = json.load(open(OUT))
    model = load_text_only_model(MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    total = fails = 0
    worst = {}
    for run in data["runs"]:
        for c in run["completions"]:
            seq = c["prompt_ids"] + c["tokens"]
            hidden = batch_completion_hidden_states(model, [(seq, len(c["prompt_ids"]))])[0]
            res = audit_completion(hidden, c["proofs"], PROOF)
            total += 1
            if not res.passed:
                fails += 1
                print("[check] ÉCHEC", run["k"], res.reason, flush=True)
            for r in res.results:
                for key in ("exp_mismatches", "mant_err_mean", "mant_err_median"):
                    v = getattr(r, key, None)
                    if v is not None:
                        worst[key] = max(worst.get(key, 0), v)
            del hidden
            torch.cuda.empty_cache()
    print(f"[check] {total - fails}/{total} complétions passent | pires valeurs {worst} | seuils "
          f"{PROOF.thresholds()}", flush=True)
    print("CHECK_PASS" if fails == 0 and total > 0 else "CHECK_FAIL", flush=True)


if __name__ == "__main__":
    {"gen": gen, "check": check}[sys.argv[1]]()
