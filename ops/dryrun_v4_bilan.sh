#!/bin/bash
# Bilan du dry-run : verdicts du mock + signaux mineur.
echo "===== checks du mock (mock_checks.jsonl) ====="
python3 - <<'EOF'
import json
from collections import Counter
rows = []
try:
    rows = [json.loads(l) for l in open("/workspace/mock_checks.jsonl")]
except FileNotFoundError:
    print("aucun check — le mineur n'a encore rien soumis")
subs = [r for r in rows if r.get("kind") == "submit"]
pres = [r for r in rows if r.get("kind") == "precommit"]
print(f"precommits: {len(pres)} (ok={sum(1 for r in pres if r['ok'])})")
print(f"submits:    {len(subs)} (ok={sum(1 for r in subs if r['ok'])})")
print("raisons:", dict(Counter((r.get('reason') or 'ACCEPTED') for r in subs)))
for r in subs[:4]:
    print("  ex:", {k: r.get(k) for k in ("ok","reason","prompt_idx","sigma","n_bad","n_trunc","envelope_sig_ok","merkle_ok","max_completion")})
EOF
echo; echo "===== signaux mineur ====="
grep -cE "submitted window" /workspace/dryrun_miner.log 2>/dev/null | xargs echo "soumissions loggées:"
grep -E "v4 constants|ERROR|Traceback|pre_bake\[(termination|malformed|uncertain|out_of_zone)\]" /workspace/dryrun_miner.log 2>/dev/null | tail -8
echo; echo "===== dumps étude (les 4 writers B1-B5) ====="
for f in dryrun_samples dryrun_verdicts dryrun_submits dryrun_windows; do
  echo "  $f: $(wc -l < /workspace/$f.jsonl 2>/dev/null || echo 0) lignes"
done
