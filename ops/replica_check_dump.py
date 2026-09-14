"""Rejoue un BENCH_DUMP à travers le service réplique (client du mineur) et
compare au verdict attendu ; mesure la latence par groupe de rollouts."""
import collections, json, sys, time
from reliquary.miner import replica_client

sock, dump_path = sys.argv[1], sys.argv[2]
dump = json.load(open(dump_path))
by = collections.defaultdict(list)
for row in dump["rows"]:
    by[row["prompt_idx"]].append(row)
assert replica_client.load(sock, dump["model_path"]), "load KO"
n = n_ok = 0; lat = []
for pidx, rows in by.items():
    t0 = time.time()
    res = replica_client.terminal_verdicts(
        sock, model_path=dump["model_path"], randomness=dump["randomness"],
        checkpoint_hash=dump["checkpoint_hash"], prompt_idx=pidx,
        items=[{"rollout": r["rollout"], "prompt_len": r["prompt_len"],
                "tokens": r["tokens"]} for r in rows], timeout=120)
    lat.append(time.time() - t0)
    for r in res:
        if r["ok"] is not None:
            n += 1; n_ok += int(r["ok"])
lat.sort()
print("[check] rollouts=%d ok=%d rate=%.4f latence_groupe p50=%.2fs max=%.2fs (items/groupe ~%d)"
      % (n, n_ok, n_ok / n, lat[len(lat)//2], lat[-1], len(dump["rows"]) // len(by)), flush=True)
