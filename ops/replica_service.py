"""Service « réplique validateur » : verdict exact de l'EOS final d'un rollout.

À lancer dans le venv réplique (même pile que le proof worker du validateur,
lue sur son /health : torch 2.7.0+cu128, transformers 5.10.4, flash-attn
2.8.3), avec PYTHONPATH sur le code upstream (origin/main) :

    PYTHONPATH=/workspace/reliquary_upstream RELIQUARY_PROTOCOL_VERSION=6 \\
      /workspace/venv_val/bin/python ops/replica_service.py /workspace/replica.sock

Pourquoi (mesuré 14/09) : depuis #253 le validateur exige que le dernier token
stop soit EXACTEMENT le pick forced-seed recalculé sur SON forward. Ce modèle
est si sensible à la précision qu'aucune autre pile ne reproduit ce verdict
(HF sdpa torch 2.11 : d'accord sur 161/191 rollouts seulement).

Protocole : une requête JSON par ligne sur un socket Unix, une réponse JSON.
  {"op": "ping"}
  {"op": "terminal", "model_path", "randomness", "checkpoint_hash",
   "prompt_idx", "items": [{"rollout", "prompt_len", "tokens"}, ...]}
→ {"ok": true, "results": [{"ok": bool|null, "pick": int|null,
                            "cdf_miss": float|null}, ...]}
``ok`` null = le rollout ne finit pas sur un stop (non concerné). ``pick`` =
le token que le validateur tire à la position terminale (sert à réparer).

Chaque forward reste une séquence ``[1, seq_len]`` sans masque, exactement
comme ``verifier.py``. Connexions acceptées en parallèle (fils) ; forwards
simultanés bornés par ``REPLICA_WORKERS`` (défaut 1 = sérialisés sur le flux
CUDA par défaut, le mode validé le 14/09). Au-delà de 1, chaque fil calcule
sur son propre flux CUDA — à n'activer qu'après ``ops/replica_parallel_gate.py``
(lignes de logits identiques au bit près au mode sérialisé).
"""
from __future__ import annotations

import json
import logging
import contextlib
import os
import socketserver
import sys
import threading

logger = logging.getLogger("replica")


class UpstreamBackend:
    """Forward et pick réalisés par le CODE UPSTREAM du validateur."""

    def __init__(self) -> None:
        self.model = None
        self.lm_head = None

    def load(self, model_path: str) -> set[int]:
        import torch

        from reliquary.shared.modeling import (
            load_text_generation_model, load_tokenizer, resolve_eos_token_ids,
        )

        self.model = None
        torch.cuda.empty_cache()
        attn = os.environ.get("REPLICA_ATTN", "flash_attention_2")
        self.model = load_text_generation_model(
            model_path, torch_dtype=torch.bfloat16, attn_implementation=attn,
        ).to("cuda").eval()
        self.lm_head = self.model.lm_head
        return set(resolve_eos_token_ids(self.model, load_tokenizer(model_path)))

    def worker_context(self):
        """Mode parallèle : un flux CUDA par fil (kernels identiques, exécutés
        en concurrence). Mode sérialisé : flux par défaut, inchangé."""
        if _workers_from_env() <= 1:
            return contextlib.nullcontext()
        import torch

        local = _THREAD_STREAMS
        stream = getattr(local, "stream", None)
        if stream is None:
            stream = local.stream = torch.cuda.Stream()
        return _stream_scope(stream)

    def terminal_row(self, tokens: list[int]):
        import torch

        from reliquary.shared.forward import forward_single_layer

        with torch.no_grad():
            h, _ = forward_single_layer(
                self.model, torch.tensor([tokens], device="cuda"), None, -1,
                materialize_logits=False,
            )
            # verifier._LazyLogitRows.__getitem__(t - 1) : projection d'UNE ligne
            return self.lm_head(h[0][len(tokens) - 2])

    def diagnose(self, row, token: int, u: float):
        from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
        from reliquary.environment.forced_sampling import pick, warp
        from reliquary.validator.verifier import _forced_pick_diagnostics

        exact, miss = _forced_pick_diagnostics(row, int(token), u)
        probs = warp(row.float(), t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        return bool(exact), float(miss), int(pick(probs, u))

    def u_at(self, randomness, prompt_idx, checkpoint_hash, rollout, j):
        from reliquary.environment.forced_sampling import u_at

        return u_at(randomness, int(prompt_idx), checkpoint_hash, int(rollout), int(j))


_THREAD_STREAMS = threading.local()


@contextlib.contextmanager
def _stream_scope(stream):
    import torch

    with torch.cuda.stream(stream):
        yield
    stream.synchronize()


def _workers_from_env() -> int:
    try:
        return max(1, int(os.environ.get("REPLICA_WORKERS", "1") or 1))
    except ValueError:
        return 1


class ReplicaState:
    """Logique pure du service (le backend porte le GPU et le code upstream).

    ``workers`` forwards au plus en même temps ; un chargement de checkpoint
    prend TOUS les jetons (aucun forward pendant qu'on change de modèle)."""

    def __init__(self, backend, workers: int | None = None) -> None:
        self.backend = backend
        self.model_path: str | None = None
        self.eos: set[int] = set()
        self.workers = _workers_from_env() if workers is None else max(1, int(workers))
        self._slots = threading.BoundedSemaphore(self.workers)
        self._load_lock = threading.Lock()

    def _ensure(self, model_path: str) -> None:
        if model_path == self.model_path:
            return
        with self._load_lock:
            if model_path == self.model_path:
                return
            for _ in range(self.workers):
                self._slots.acquire()
            try:
                logger.info("réplique: chargement %s", model_path)
                self.eos = set(self.backend.load(model_path))
                self.model_path = model_path
            finally:
                for _ in range(self.workers):
                    self._slots.release()

    @contextlib.contextmanager
    def _slot(self, model_path: str):
        """Un jeton de forward, sur le BON modèle (un chargement concurrent a
        pu passer entre ``_ensure`` et l'obtention du jeton)."""
        while True:
            self._ensure(model_path)
            self._slots.acquire()
            if self.model_path == model_path:
                break
            self._slots.release()
        try:
            ctx = getattr(self.backend, "worker_context", None)
            with (ctx() if ctx is not None else contextlib.nullcontext()):
                yield
        finally:
            self._slots.release()

    def handle(self, req: dict) -> dict:
        try:
            op = req.get("op")
            if op == "ping":
                return {"ok": True, "model_path": self.model_path}
            if op == "load":
                self._ensure(req["model_path"])
                return {"ok": True, "model_path": self.model_path}
            if op != "terminal":
                return {"ok": False, "error": f"op inconnue: {op!r}"}
            with self._slot(req["model_path"]):
                results = self._terminal(req)
            return {"ok": True, "results": results}
        except Exception as exc:          # une requête ne tue jamais le service
            logger.exception("réplique: requête en échec")
            return {"ok": False, "error": repr(exc)}

    def _terminal(self, req: dict) -> list[dict]:
        results = []
        for item in req["items"]:
            toks = list(item["tokens"])
            plen = int(item["prompt_len"])
            j = len(toks) - 1 - plen
            if len(toks) < 2 or j < 0 or int(toks[-1]) not in self.eos:
                results.append({"ok": None, "pick": None, "cdf_miss": None})
                continue
            row = self.backend.terminal_row(toks)
            u = self.backend.u_at(req["randomness"], req["prompt_idx"],
                                  req["checkpoint_hash"], item["rollout"], j)
            exact, miss, picked = self.backend.diagnose(row, int(toks[-1]), u)
            results.append({"ok": bool(exact), "pick": picked,
                            "cdf_miss": round(miss, 8)})
        return results


def make_server(socket_path: str, state: ReplicaState):
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    class _Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                except ValueError as exc:
                    resp = {"ok": False, "error": f"json: {exc}"}
                else:
                    resp = state.handle(req)
                self.wfile.write((json.dumps(resp) + "\n").encode())
                self.wfile.flush()

    class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        # le mineur envoie jusqu'à GRADE_CONCURRENCY requêtes d'un coup : la
        # file d'écoute par défaut (5) les refusait en EAGAIN
        daemon_threads = True
        request_queue_size = 256

    return _Server(socket_path, _Handler)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | replica | %(levelname)s | %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/replica.sock"
    server = make_server(path, ReplicaState(UpstreamBackend()))
    logger.info("réplique prête sur %s (workers=%d)", path, _workers_from_env())
    server.serve_forever()


if __name__ == "__main__":
    main()
