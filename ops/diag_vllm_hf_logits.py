"""Diagnostic : les distributions de vLLM et de HF divergent-elles au-delà du
bruit numérique ? (14/09, suite au banc EOS final)

Le banc a montré des ratés à −0,16 de probabilité cumulée du bord de
l'intervalle EOS : trop gros pour des arrondis bf16. Ce script compare, sur
un vrai prompt code enveloppé v6 et sa complétion vLLM :
  - la largeur des logits vus par le processeur forced-seed de vLLM ;
  - HF vs vLLM ``prompt_logprobs`` (distribution RAW, avant processeur) sur
    les mêmes positions : écart max de logprob du token, écart de masse top-20 ;
  - la masse HF portée par les ids >= EOS (fin de vocabulaire : l'ordre CDF
    forced-seed est croissant par id, l'EOS 151643 est presque en dernier).
"""
from __future__ import annotations

import json
import os

import torch


def main() -> None:
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from reliquary.environment import load_environment
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

    repo = os.environ.get("BENCH_REPO", "ReliquaryForge/qwen3-4b-base-dapo-v4")
    local = snapshot_download(repo)
    tok = load_tokenizer(local)
    env = load_environment("opencodeinstruct")
    idx = int(os.environ.get("DIAG_PROMPT_IDX", "1631303"))
    ptoks = encode_prompt(tok, env.get_problem(idx)["prompt"])

    llm = LLM(model=local, gpu_memory_utilization=0.5, max_model_len=4096,
              dtype="bfloat16", enforce_eager=True,
              disable_cascade_attn=True, max_num_seqs=8)
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=300,
                        seed=1234, prompt_logprobs=20, logprobs=20)
    out = llm.generate([TokensPrompt(prompt_token_ids=ptoks)], sp)[0]
    gen = list(out.outputs[0].token_ids)
    gen_lps = out.outputs[0].logprobs            # list[dict[token -> Logprob]]
    vocab_v = llm.llm_engine.model_config.get_vocab_size()
    del llm
    torch.cuda.empty_cache()

    model = load_text_generation_model(
        local, torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("BENCH_HF_ATTN", "sdpa"),
    ).to("cuda").eval()
    full = ptoks + gen
    with torch.no_grad():
        logits = model(torch.tensor([full], device="cuda")).logits[0].float()
    lsm = torch.log_softmax(logits, dim=-1)
    plen = len(ptoks)
    eos = 151643

    diffs_tok, diffs_mass, tail_mass = [], [], []
    for j, d in enumerate(gen_lps or []):
        pos = plen - 1 + j
        hf = lsm[pos]
        tail_mass.append(float(hf[eos:].exp().sum()))
        chosen = gen[j]
        if chosen in d:
            diffs_tok.append(abs(float(d[chosen].logprob) - float(hf[chosen])))
        top = sorted(d.items(), key=lambda kv: -kv[1].logprob)[:20]
        mv = sum(float(torch.tensor(lp.logprob).exp()) for _, lp in top)
        mh = float(sum(hf[t].exp() for t, _ in top))
        diffs_mass.append(abs(mv - mh))

    # où divergent-ils ? décodage (logprobs de génération) par position, et
    # prefill (prompt_logprobs) sur les positions du prompt.
    big = [(j, round(abs(float(d[gen[j]].logprob) - float(lsm[plen - 1 + j][gen[j]])), 3),
            round(float(lsm[plen - 1 + j][gen[j]].exp()), 3))
           for j, d in enumerate(gen_lps or []) if gen[j] in d
           and abs(float(d[gen[j]].logprob) - float(lsm[plen - 1 + j][gen[j]])) > 0.05]
    print("[diag] DECODE_BIG (offset, |dlp|, p_hf) " + json.dumps(big[:60]), flush=True)
    pre = []
    for i, d in enumerate(out.prompt_logprobs or []):
        if not d or i == 0:
            continue
        t = ptoks[i]
        if t in d:
            pre.append(abs(float(d[t].logprob) - float(lsm[i - 1][t])))
    ps = sorted(pre)
    print("[diag] PREFILL |dlp| p50=%.5f p90=%.5f max=%.5f n=%d" % (
        ps[len(ps)//2], ps[int(.9*len(ps))], ps[-1], len(ps)), flush=True)

    def q(v, p):
        s = sorted(v)
        return round(s[min(len(s) - 1, int(p * len(s)))], 5) if s else None

    print("[diag] RESULT " + json.dumps({
        "prompt_idx": idx, "plen": plen, "gen_len": len(gen),
        "vocab_vllm": vocab_v, "vocab_hf_logits": int(logits.shape[-1]),
        "tokenizer_len": len(tok),
        "logprob_diff_chosen": {"p50": q(diffs_tok, .5), "p90": q(diffs_tok, .9),
                                "max": q(diffs_tok, 1.0)},
        "top20_mass_diff": {"p50": q(diffs_mass, .5), "p90": q(diffs_mass, .9),
                            "max": q(diffs_mass, 1.0)},
        "hf_mass_ids_ge_eos": {"p50": q(tail_mass, .5), "p90": q(tail_mass, .9),
                               "max": q(tail_mass, 1.0)},
    }), flush=True)


if __name__ == "__main__":
    main()
