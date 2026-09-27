#!/usr/bin/env python3
"""Vigie du mineur SFT (27/09) — cron dev box toutes les 10 min.

Lit l'état de notre hotkey dans la tâche corpus (R2, lecture seule) :
- premier échec d'audit CONFIRMÉ, suspicion ou ban ⇒ pose /workspace/SFT_STOP
  sur la box et arrête le mineur (3 échecs en 7 j = ban de 7 j pour la hotkey) ;
- sinon journalise audits passés, curseur et part de la dernière archive payée.
"""
import gzip
import json
import os
import subprocess
import sys
import time

HK = "5Gy8EzpC6PXZX1XyCfRUmuYp1Qk6ZJwtFE2GHXudiBMtiJaW"          # hotkey81.2
BOX, PORT = "root@162.243.212.30", "20299"                       # 27/09 : brave-wolf-54
JOB = "code-qwen38-27b-v1"
LOG = "/root/subnet81/data/vigie_sft.log"


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def stop_miner(reason):
    cmd = ("touch /workspace/SFT_STOP; for p in $(pgrep -f 'sft_miner[.]miner'); do kill $p; done; "
           "echo stoppé")
    r = subprocess.run(["ssh", "-p", PORT, "-o", "ConnectTimeout=20", BOX, cmd],
                       capture_output=True, text=True, timeout=60)
    log(f"ARRÊT DU MINEUR ({reason}) : {r.stdout.strip()} {r.stderr.strip()[:200]}")


def main():
    import boto3
    s3 = boto3.session.Session().client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])
    get = lambda k: s3.get_object(Bucket="reliquary", Key=k)["Body"].read()  # noqa: E731
    miners = json.loads(get(f"reliquary/corpus/jobs/{JOB}/miners.json"))
    me = (miners.get("miners", miners) or {}).get(HK)
    ledgers = json.loads(get(f"reliquary/corpus/jobs/{JOB}/ledgers.json"))
    cursor = ledgers.get("cursors", {}).get(HK)
    keys = sorted(o["Key"] for o in s3.list_objects_v2(
        Bucket="reliquary", Prefix="reliquary/tasks/corpus-code-v1/dataset/")["Contents"])
    last = json.loads(gzip.decompress(get(keys[-1])))
    r = last.get("rewards_by_hotkey") or {}
    share = r.get(HK, 0.0) / max(sum(r.values()), 1e-12)
    if me is None:
        log(f"hotkey pas encore dans miners.json | curseur {cursor} | part {share:.1%} ({keys[-1][-18:]})")
        return 0
    fails = me.get("confirmed_failures") or []
    passed = me.get("audited_passed")
    log(f"audités OK {passed} | échecs confirmés {len(fails)} | suspect {me.get('suspect_until')} "
        f"| ban {me.get('banned_until')} | curseur {cursor} | part {share:.1%} ({keys[-1][-18:]})")
    if fails or me.get("banned_until") or me.get("suspect_until"):
        stop_miner(f"échecs {len(fails)}, suspect {me.get('suspect_until')}, ban {me.get('banned_until')}")
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # la vigie ne doit jamais planter en silence
        log(f"ERREUR vigie : {type(exc).__name__}: {exc}")
        sys.exit(1)
