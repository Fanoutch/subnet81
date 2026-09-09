#!/usr/bin/env python3
"""Rapatrie les archives de fenêtre du bucket R2 du validateur (lecture seule).

Une archive = reliquary/dataset/window-<N>.json.gz : batch retenu (avec prompts,
rollouts, rewards), rejets de TOUT le marché avec leurs horodatages d'arrivée, et
l'émission par hotkey. Voir _archive_window() côté validateur.

Usage:
    R2_ENDPOINT=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
    python3 scripts/r2_pull_windows.py --debut 32600 --fin 34160 --pas 16 --out /chemin/cache
"""
import argparse
import concurrent.futures as cf
import os
import pathlib

import boto3
from botocore.config import Config


def client():
    return boto3.session.Session().client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        region_name="auto",
        config=Config(read_timeout=120, max_pool_connections=32,
                      retries={"max_attempts": 3}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debut", type=int, required=True)
    ap.add_argument("--fin", type=int, required=True)
    ap.add_argument("--pas", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "reliquary"))
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wins = list(range(a.debut, a.fin, a.pas))

    def get(w):
        dest = out / f"{w}.json.gz"
        if dest.exists():
            return dest.stat().st_size
        try:
            raw = client().get_object(
                Bucket=a.bucket, Key=f"reliquary/dataset/window-{w}.json.gz"
            )["Body"].read()
            dest.write_bytes(raw)
            return len(raw)
        except Exception:
            return 0

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        sizes = list(ex.map(get, wins))

    ok = [s for s in sizes if s]
    print(f"{len(ok)}/{len(wins)} fenêtres  {sum(ok)/1e6:.1f} Mo  ->  {out}")


if __name__ == "__main__":
    main()
