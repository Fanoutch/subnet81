#!/usr/bin/env python3
"""Régénère l'ensemble des index BRÛLÉS PAR CONTENU et le pousse sur la box.

Source : l'instantané R2 du cooldown de contenu du validateur
(``content_cooldown_snapshots/<run_id>.json.gz``, rafraîchi toutes les 10 fen),
croisé avec la table des digests de tous les prompts (``prompt_digests.npy``,
sha256("reliquary/prompt-content/v1\\0"+env+"\\0"+prompt rendu), parité 600/600
vérifiée le 04/09). La clé R2 reste sur la dev box (.env.r2, jamais commitée).

Usage : python3 scripts/refresh_burned_idx.py [--run-id ID] [--push]
"""
import argparse, gzip, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOX = "root@157.10.162.245"; PORT = "20301"


def env_r2():
    for line in open(ROOT / ".env.r2"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k, v)
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["R2_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["R2_SECRET_ACCESS_KEY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="qwen3-4b-base-dapo-reasoning-v5-basereset-20260825")
    ap.add_argument("--digests", default=str(ROOT / "data" / "prompt_digests.npy"))
    ap.add_argument("--out", default=str(ROOT / "data" / "burned_idx.npy"))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    import boto3, numpy as np
    from botocore.config import Config
    env_r2()
    c = boto3.session.Session().client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                                       region_name="auto", config=Config(read_timeout=120))
    key = f"content_cooldown_snapshots/{a.run_id}.json.gz"
    raw = c.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)["Body"].read()
    snap = json.loads(gzip.decompress(raw))
    ct = snap["envs"]["opencodeinstruct"]
    dig = np.load(a.digests, allow_pickle=False)
    burn = np.frombuffer(b"".join(bytes.fromhex(k) for k in ct), dtype=np.uint8).reshape(-1, 32)
    v = np.ascontiguousarray(dig).view(np.dtype((np.void, 32))).ravel()
    bv = np.ascontiguousarray(burn).view(np.dtype((np.void, 32))).ravel()
    idx = np.nonzero(np.isin(v, bv))[0].astype(np.int64)
    tmp = a.out + ".tmp.npy"; np.save(tmp, idx); os.replace(tmp, a.out)
    print(f"{time.strftime('%F %T')} snapshot fen {snap.get('snapshot_window')} run {snap.get('run_id')} "
          f"digests {len(ct)} → {len(idx)} index brûlés -> {a.out}")
    if a.push:
        subprocess.run(["scp", "-q", "-P", PORT, a.out, f"{BOX}:/workspace/burned_idx.npy.tmp"], check=True)
        subprocess.run(["ssh", "-p", PORT, BOX, "mv -f /workspace/burned_idx.npy.tmp /workspace/burned_idx.npy"], check=True)
        print("poussé sur la box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
