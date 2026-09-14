"""Gate du mode parallèle de la réplique (``REPLICA_WORKERS`` > 1).

La réplique doit rendre EXACTEMENT le verdict du validateur : le mode
parallèle (un flux CUDA par fil) n'est acceptable que si chaque ligne de
logits terminale est identique AU BIT PRÈS à celle du mode sérialisé validé
(flux par défaut). Mesure aussi le débit des deux modes.

    PYTHONPATH=/workspace/reliquary_upstream RELIQUARY_PROTOCOL_VERSION=6 \\
      /workspace/venv_val/bin/python ops/replica_parallel_gate.py \\
      <model_path> /workspace/bench_dump.json [workers=4] [rows=96]

Sortie : ``[gate] PASS|FAIL ...`` (code retour 0 si PASS).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path


def _service():
    spec = importlib.util.spec_from_file_location(
        "replica_service", Path(__file__).with_name("replica_service.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import torch

    model_path, dump_path = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 96
    dump = json.load(open(dump_path))
    rows = [r for r in dump["rows"] if len(r["tokens"]) >= 2][:n_rows]
    svc = _service()
    be = svc.UpstreamBackend()
    be.load(model_path)

    def row_of(tokens):
        with be.worker_context():
            out = be.terminal_row(tokens).float().cpu()
        return out

    for r in rows[:2]:                      # préchauffe (kernels, allocateur)
        row_of(r["tokens"])

    os.environ["REPLICA_WORKERS"] = "1"
    t0 = time.monotonic()
    serial = [row_of(r["tokens"]) for r in rows]
    torch.cuda.synchronize()
    t_serial = time.monotonic() - t0

    os.environ["REPLICA_WORKERS"] = str(workers)
    par: list = [None] * len(rows)
    nxt = {"i": 0}
    lk = threading.Lock()

    def work():
        while True:
            with lk:
                i = nxt["i"]
                nxt["i"] += 1
            if i >= len(rows):
                return
            par[i] = row_of(rows[i]["tokens"])

    t0 = time.monotonic()
    ths = [threading.Thread(target=work) for _ in range(workers)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    torch.cuda.synchronize()
    t_par = time.monotonic() - t0

    diff = [i for i in range(len(rows)) if not torch.equal(serial[i], par[i])]
    toks = sum(len(r["tokens"]) for r in rows)
    verdict = "PASS" if not diff else "FAIL"
    print(f"[gate] {verdict} rows={len(rows)} tokens={toks} workers={workers} "
          f"lignes_differentes={len(diff)} serie={t_serial:.2f}s "
          f"parallele={t_par:.2f}s gain=x{t_serial / max(t_par, 1e-9):.2f}",
          flush=True)
    if diff:
        i = diff[0]
        print(f"[gate] 1er écart row={i} max|Δ|="
              f"{(serial[i] - par[i]).abs().max().item():.3e}", flush=True)
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
