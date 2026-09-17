"""Service « réplique validateur » : verdict exact de l'EOS final d'un rollout.

À lancer dans le venv réplique (même pile que le proof worker du validateur,
lue sur son /health : torch 2.7.0+cu128, transformers 5.10.4, flash-attn
2.8.3), avec PYTHONPATH sur le code upstream (origin/main) :

    PYTHONPATH=/workspace/reliquary_upstream RELIQUARY_PROTOCOL_VERSION=6 \\
      RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reliquary-v1 \\
      RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED=1 \\
      /workspace/venv_val/bin/python ops/replica_service.py /workspace/replica.sock

⚠️ Sans ``RELIQUARY_PROTOCOL_PROFILE``, le code upstream retombe sur son profil
par défaut (``qwen35-2b-auction-v2`` : T=0,6, top-k 20, domaine forced-seed-v2)
et tous les verdicts sont faux (vécu le 14/09). Le service refuse de démarrer
si le profil actif n'est pas ``REPLICA_EXPECTED_PROFILE``.

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
  {"op": "chosen_logprobs", "model_path", "items": [{"prompt_len", "tokens"}]}
→ {"ok": true, "results": [[logprob par position de complétion] | null, ...]}
(rollouts < CHALLENGE_K : le validateur compare notre revendication à CE calcul
à toutes les positions — mesuré 15/09, 4 logprob_mismatch sur des rollouts
d'un token).

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
import time

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

    def chosen_logprobs(self, tokens: list[int], prompt_len: int, *,
                        device: str = "cuda") -> list[float | None]:
        """Logprob du token choisi à chaque position de complétion, calculé
        comme ``verifier._gpu_completion_token_stats`` → ``batcher.
        _verify_logprobs_for_training`` : lignes ``t-1`` projetées ENSEMBLE
        (``_LazyLogitRows.index_select``), ``softmax(lignes.float() / T)``,
        puis ``math.log`` du float. C'est la valeur à laquelle le validateur
        compare notre revendication sous ``CHALLENGE_K`` tokens."""
        import math

        import torch

        from reliquary.constants import T_PROTO
        from reliquary.shared.forward import forward_single_layer

        valid_t = [t for t in range(int(prompt_len), len(tokens)) if t > 0]
        if not valid_t:
            return []
        with torch.no_grad():
            h, _ = forward_single_layer(
                self.model, torch.tensor([tokens], device=device), None, -1,
                materialize_logits=False,
            )
            pos = torch.tensor([t - 1 for t in valid_t], device=device,
                               dtype=torch.long)
            tok = torch.tensor([int(tokens[t]) for t in valid_t], device=device,
                               dtype=torch.long)
            rows = self.lm_head(h[0].index_select(0, pos))
            probs = (rows.float() / float(T_PROTO)).softmax(dim=-1)
            chosen = probs.gather(1, tok.unsqueeze(1)).squeeze(1).tolist()
        return [math.log(float(c)) if c > 0.0 else None for c in chosen]

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
            if op == "chosen_logprobs":
                with self._slot(req["model_path"]):
                    results = self._chosen_logprobs(req)
                return {"ok": True, "results": results}
            if op != "terminal":
                return {"ok": False, "error": f"op inconnue: {op!r}"}
            # 17/09 (mesure pure) : attente du jeton vs calcul, renvoyés au
            # mineur (clés ignorées par les anciens clients).
            _t0 = time.monotonic()
            with self._slot(req["model_path"]):
                _t1 = time.monotonic()
                results = self._terminal(req)
                _t2 = time.monotonic()
            return {"ok": True, "results": results,
                    "t_wait": round(_t1 - _t0, 4), "t_compute": round(_t2 - _t1, 4)}
        except Exception as exc:          # une requête ne tue jamais le service
            logger.exception("réplique: requête en échec")
            return {"ok": False, "error": repr(exc)}

    def _chosen_logprobs(self, req: dict) -> list:
        results = []
        for item in req["items"]:
            toks = [int(x) for x in item["tokens"]]
            plen = int(item["prompt_len"])
            if plen < 1 or len(toks) <= plen:
                results.append(None)
                continue
            lps = self.backend.chosen_logprobs(toks, plen)
            ok = len(lps) == len(toks) - plen and all(v is not None for v in lps)
            results.append([float(v) for v in lps] if ok else None)
        return results

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


EXPECTED_PROFILE_DEFAULT = "qwen3-4b-base-dapo-reliquary-v1"


def profile_error(constants, expected: str) -> str | None:
    """Message d'erreur si le profil upstream actif n'est pas celui du
    validateur (``None`` si conforme)."""
    active = getattr(constants, "PROTOCOL_PROFILE_ID", None)
    if active == expected:
        return None
    return (f"profil de protocole actif {active!r} (T={getattr(constants, 'T_PROTO', None)}, "
            f"top_k={getattr(constants, 'TOP_K_PROTO', None)}, "
            f"top_p={getattr(constants, 'TOP_P_PROTO', None)}) != attendu {expected!r} — "
            "poser RELIQUARY_PROTOCOL_PROFILE (et RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED=1)")


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
    import reliquary.constants as constants

    expected = os.environ.get("REPLICA_EXPECTED_PROFILE", EXPECTED_PROFILE_DEFAULT)
    err = profile_error(constants, expected)
    if err:
        logger.error("réplique: %s", err)
        sys.exit(2)
    logger.info("réplique: profil %s (T=%s top_k=%s top_p=%s)",
                constants.PROTOCOL_PROFILE_ID, constants.T_PROTO,
                constants.TOP_K_PROTO, constants.TOP_P_PROTO)
    server = make_server(path, ReplicaState(UpstreamBackend()))
    logger.info("réplique prête sur %s (workers=%d)", path, _workers_from_env())
    server.serve_forever()


if __name__ == "__main__":
    main()
