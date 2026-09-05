#!/usr/bin/env python3
"""Moissonne les métriques par fenêtre depuis miner.log (box) et les cumule
localement (dédup par fenêtre) — le log est tronqué à chaque restart."""
import json, re, subprocess, time, os

OUT = "/root/subnet81/data/window_timing_v4.jsonl"
seen = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: seen.add(json.loads(l)["window_n"])
        except Exception: pass

raw = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=15", "-p", "20098", "root@38.255.28.21",
     "cat /workspace/miner.log"],
    capture_output=True, text=True, timeout=120).stdout
def ts(l):
    try: return time.mktime(time.strptime(l[:19], "%Y-%m-%d %H:%M:%S"))
    except Exception: return None

flips, fires, posts = {}, {}, {}
for l in raw.splitlines():
    t = ts(l)
    if t is None: continue
    m = re.search(r"randomness flip \(window=(\d+)\)", l)
    if m: flips[int(m.group(1))] = t; continue
    m = re.search(r"fire_for_window=(\d+): finalizing", l)
    if m: fires.setdefault(int(m.group(1)), []).append(t); continue
    m = re.search(r"submitted window=(\d+) prompt=\d+ accepted=(\w+) reason=(\w+)", l)
    if m: posts.setdefault(int(m.group(1)), []).append((t, m.group(2) == "True", m.group(3)))

new = 0
allw = sorted(flips)
with open(OUT, "a") as fh:
    for w in allw:
        if w in seen or w == allw[-1]:  # la dernière peut être incomplète
            continue
        f = flips[w]
        acc = [round(t - f) for t, a, r in posts.get(w, []) if a]
        bf = [round(t - f) for t, a, r in posts.get(w, []) if r == "batch_filled"]
        fh.write(json.dumps({
            "window_n": w, "flip_ts": f,
            "first_fire_s": round(min(fires[w]) - f) if w in fires else None,
            "accepted_offsets_s": acc, "n_accepted": len(acc),
            "seal_s": min(bf) if bf else None,
        }) + "\n")
        new += 1
print(f"{time.strftime('%H:%M')} harvest: +{new} fenêtres (total {len(seen) + new})")
