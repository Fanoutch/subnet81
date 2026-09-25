#!/usr/bin/env python3
"""Empreintes de contenu BRÛLÉES d'un env, depuis l'instantané R2 du
validateur, poussées sur la box (24/09, mineur math).

Le cooldown de contenu dure 1 000 000 de fenêtres : toute empreinte de
l'instantané est définitivement refusée (``content_in_cooldown``). Le mineur
lit le fichier (une empreinte hex par ligne) via ``RELIQUARY_BURNED_DIGESTS``
et le recharge tout seul quand il change. La clé R2 reste sur la dev box
(.env.r2, jamais commitée).

Usage : python3 scripts/refresh_burned_digests.py [--env openmathinstruct]
        [--run-id ID] [--push --box root@IP --port N [--dest CHEMIN]]
"""
import argparse, gzip, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def env_r2():
    for cand in (ROOT / ".env.r2", pathlib.Path("/root/subnet81/.env.r2")):
        if cand.exists():
            for line in open(cand):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)
            break
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["R2_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["R2_SECRET_ACCESS_KEY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="openmathinstruct")
    ap.add_argument("--run-id", default="qwen3-4b-base-dapo-reliquary-v1-20260910")
    ap.add_argument("--out", default=str(ROOT / "data" / "burned_digests_math.txt"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--box", default="")
    ap.add_argument("--port", default="")
    ap.add_argument("--dest", default="/workspace/math/burned_digests_math.txt")
    a = ap.parse_args()
    import boto3
    from botocore.config import Config
    env_r2()
    c = boto3.session.Session().client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
        config=Config(read_timeout=120))
    key = f"content_cooldown_snapshots/{a.run_id}.json.gz"
    snap = json.loads(gzip.decompress(
        c.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)["Body"].read()))
    digests = sorted(k.lower() for k in snap["envs"][a.env])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(digests) + "\n")
    os.replace(tmp, a.out)
    print(f"{time.strftime('%F %T')} run {snap.get('run_id')} fen "
          f"{snap.get('snapshot_window')} env {a.env} : {len(digests)} empreintes -> {a.out}")
    if a.push:
        if not (a.box and a.port):
            sys.exit("--push exige --box et --port")
        subprocess.run(["scp", "-q", "-P", a.port, a.out, f"{a.box}:{a.dest}.tmp"], check=True)
        subprocess.run(["ssh", "-p", a.port, a.box, f"mv -f {a.dest}.tmp {a.dest}"], check=True)
        print("poussé sur la box :", a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
