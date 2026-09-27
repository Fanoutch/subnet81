import json, sys
import httpx
from huggingface_hub import snapshot_download
from reliquary.corpus.job import parse_job
from reliquary.corpus.encoding import checkpoint_fingerprint
from reliquary.shared.modeling import load_tokenizer
from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
print("profil actif:", ACTIVE_PROTOCOL_PROFILE.profile_id, "| preuve:", toploc_proof(ACTIVE_PROTOCOL_PROFILE))
job = parse_job(httpx.get("http://62.238.81.36:8000/corpus/job", timeout=30).json())
d = snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision)
fp = checkpoint_fingerprint(d)
print("empreinte", "OK" if fp == job.checkpoint_sha256 else f"DIFFÉRENTE {fp}")
tok = load_tokenizer(d)
enc = lambda t: list(getattr(tok.encode(t, add_special_tokens=False), "ids", tok.encode(t, add_special_tokens=False)))
r = renderer_for_job(job, enc, tokenizer=tok)
prompts = prompt_job_for_spec(job)
ref = json.load(open("/workspace/corpus_prompts.json"))
ok = sum(1 for p in ref[:20] if r.initial_text(prompts.task_for(p["prompt_index"])) == p["rendered_prompt"])
print(f"rendu identique aux soumissions R2 : {ok}/20")
print("RENDER_OK" if ok == 20 and fp == job.checkpoint_sha256 else "RENDER_FAIL")
