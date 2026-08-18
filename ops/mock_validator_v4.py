#!/usr/bin/env python3
"""Validateur FACTICE v4 — dry-run de conformité du mineur AVANT activation.

Sert le contrat v4 sur 127.0.0.1:9999 (fenêtres de 150 s, randomness
déterministe par fenêtre) et JUGE chaque soumission avec NOS ports des checks
validateur : pydantic BatchSubmissionRequest (schéma), protocol_version=4,
generation_profile_id, 16 rollouts, cap 8192, merkle LEGACY, signature
d'enveloppe, terminaison (EOS final / budget truncated 1 math / 3 code), zone
σ≥0.24. Tout est écrit dans /workspace/mock_checks.jsonl ; /verdicts rejoue
les jugements (teste aussi le polling B3 du mineur).
⚠️ Ne vérifie PAS : GRAIL/seed-consistency (gate GPU déjà PASS), timing réel,
cooldown. C'est un test de CBLAGE wire, pas d'économie.
"""
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reliquary import constants as c
from reliquary.protocol.submission import (
    BatchSubmissionRequest, SubmissionPrecommitRequest,
)
from reliquary.protocol.signatures import verify_envelope_signature
from reliquary.miner.engine import (
    _compute_merkle_root, termination_partition, validator_termination_ok,
)

PORT = 9999
WINDOW_S = 150
T0 = time.time()
CKPT_HASH = ""  # pas de checkpoint publié → le mineur part sur le fallback pinné
CHECKS = "/workspace/mock_checks.jsonl"
VERDICTS = []
LOCK = threading.Lock()


def now_window():
    n = int((time.time() - T0) // WINDOW_S)
    randomness = hashlib.sha256(f"mock-v4-{n}".encode()).hexdigest()
    return n, randomness


def log_check(row):
    row["ts"] = round(time.time(), 1)
    with LOCK:
        with open(CHECKS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def judge(req: BatchSubmissionRequest, raw: dict):
    """Rejoue les checks v4 ; renvoie (reason, details). reason=None = OK."""
    d = {}
    d["protocol_version"] = raw.get("protocol_version")
    if raw.get("protocol_version") != 4:
        return "protocol_version_mismatch", d
    d["profile"] = raw.get("generation_profile_id")
    if raw.get("generation_profile_id") != "qwen3-4b-base-dapo-v4":
        return "generation_contract_mismatch", d
    rollouts = req.rollouts
    d["n_rollouts"] = len(rollouts)
    if len(rollouts) != 16:
        return "bad_schema", d
    # merkle LEGACY (celui que le validateur calcule)
    legacy = _compute_merkle_root(rollouts)
    d["merkle_ok"] = (legacy == req.merkle_root)
    if not d["merkle_ok"]:
        return "merkle_root_mismatch", d
    # signature d'enveloppe — la préimage inclut la randomness : on essaie la
    # fenêtre courante puis la précédente (grâce d'upload). Aucune des deux =
    # signature fausse OU mauvaise randomness.
    win_n, rand_now = now_window()
    rand_prev = hashlib.sha256(f"mock-v4-{win_n-1}".encode()).hexdigest()
    sig_ok, rand_used = False, None
    for rnd in (rand_now, rand_prev):
        try:
            if verify_envelope_signature(
                miner_hotkey=req.miner_hotkey,
                window_start=req.window_start,
                prompt_idx=req.prompt_idx,
                merkle_root=req.merkle_root,
                checkpoint_hash=req.checkpoint_hash or "",
                drand_round=req.drand_round,
                randomness=rnd,
                nonce=req.nonce,
                envelope_signature=req.envelope_signature,
            ):
                sig_ok, rand_used = True, ("courante" if rnd == rand_now else "précédente")
                break
        except Exception as e:
            d["sig_err"] = f"{type(e).__name__}: {e}"
    d["envelope_sig_ok"] = sig_ok
    d["randomness_fenetre"] = rand_used
    if not sig_ok:
        return "bad_envelope_signature", d
    env_name = getattr(rollouts[0], "env_name", None) or raw.get("env_name")
    comps, rewards, lens = [], [], []
    for r in rollouts:
        tokens = list(r.tokens)
        meta = (r.commit.get("rollout", {}) or {}) if isinstance(r.commit, dict) else {}
        plen = int(meta.get("prompt_length", 0))
        comp = tokens[plen:]
        comps.append(comp)
        lens.append(len(comp))
        rewards.append(float(meta.get("reward", 0.0)))
    d["max_completion"] = max(lens)
    if max(lens) > c.MAX_NEW_TOKENS_PROTOCOL_CAP:
        return "bad_schema", d
    eos = (151643, 151645)
    n_bad, n_trunc = termination_partition(comps, eos, c.MAX_NEW_TOKENS_PROTOCOL_CAP)
    d["n_bad"], d["n_trunc"] = n_bad, n_trunc
    budget = c.max_truncated_for_environment(env_name or "openmathinstruct")
    if n_bad > 0 or n_trunc > budget:
        return "bad_termination", d
    mean = sum(rewards) / len(rewards)
    sigma = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
    d["sigma"] = round(sigma, 4)
    if sigma < 1e-8 or sigma < c.SIGMA_MIN:
        return "out_of_zone", d
    return None, d


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        n, randomness = now_window()
        if self.path.startswith("/state"):
            self._json(200, {
                "state": "open", "window_n": n, "anchor_block": 1,
                "cooldown_prompts": [], "valid_submissions": 0,
                "checkpoint_n": 0, "checkpoint_repo_id": None,
                "checkpoint_revision": None,
                "protocol_version": 4,
                "generation_profile_id": "qwen3-4b-base-dapo-v4",
                "generation_contract": {"protocol_version": 4},
                "randomness": randomness,
            })
        elif self.path.startswith("/health"):
            self._json(200, {"status": "ok", "image_revision": "mock-v4",
                             "protocol_version": 4,
                             "generation_profile_id": "qwen3-4b-base-dapo-v4"})
        elif self.path.startswith("/verdicts/"):
            with LOCK:
                vs = list(VERDICTS[-200:])
            self._json(200, {"verdicts": vs})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        n, _ = now_window()
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(raw)
        except Exception:
            self._json(400, {"detail": "bad json"}); return
        if self.path == "/submit/precommit":
            try:
                SubmissionPrecommitRequest.model_validate(payload)
                log_check({"kind": "precommit", "ok": True, "window_n": n})
                self._json(200, {"accepted": True, "reason": "submitted",
                                 "receipt_id": hashlib.sha256(raw).hexdigest()[:16],
                                 "upload_deadline_ts": time.time() + 33})
            except Exception as e:
                log_check({"kind": "precommit", "ok": False, "err": str(e)[:400]})
                self._json(200, {"accepted": False, "reason": "bad_schema"})
        elif self.path == "/submit":
            try:
                req = BatchSubmissionRequest.model_validate(payload)
            except Exception as e:
                log_check({"kind": "submit", "ok": False, "stage": "schema",
                           "err": str(e)[:600]})
                self._json(200, {"accepted": False, "reason": "bad_schema"})
                return
            reason, details = judge(req, payload)
            ok = reason is None
            log_check({"kind": "submit", "ok": ok, "reason": reason,
                       "prompt_idx": req.prompt_idx, "window_n": n, **details})
            with LOCK:
                VERDICTS.append({
                    "merkle_root": req.merkle_root, "window_n": n,
                    "accepted": ok, "reason": reason or "accepted",
                    "ts": time.time(), "canonical_rank": 1 if ok else None,
                    "accepted_into_pool": ok, "selected_for_batch": ok,
                    "rewarded": ok,
                })
            self._json(200, {"accepted": ok, "reason": reason or "submitted"})
        else:
            self._json(404, {"detail": "not found"})


if __name__ == "__main__":
    print(f"[mock] validateur v4 factice sur :{PORT} — fenêtres {WINDOW_S}s, "
          f"checks → {CHECKS}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
