"""Mineur SFT (tâche corpus) à remplissage continu — 27/09.

Le mineur d'upstream (``reliquary corpus mine``) génère UN prompt à la fois
(n = 4 séquences) : mesuré 100 tok/s sur une H100 avec Qwen3.8-27B, carte
quasi vide. Ici le fil GPU garde jusqu'à ``max_in_flight`` prompts dans vLLM
et en ajoute un dès qu'un autre se termine ; le fil de soumission les envoie
STRICTEMENT dans l'ordre du curseur (``CursorBook``).

Tout ce qui touche au protocole est celui d'upstream, importé tel quel :
``build_submission`` (corps + signature), ``_retry`` / ``issue_corpus_request``
(relances), ``walk_index`` (prompt d'un curseur), ``prompt_token_ids``,
``VllmGenerator`` (vLLM, graine aléatoire, capture des états cachés, sampling
du job), ``completion_rows`` + ``build_chunk_proofs`` (preuve TOPLOC). Même
garde qu'upstream : une séquence dont les lignes capturées ne correspondent pas
(préemption KV) n'est JAMAIS soumise — le prompt est régénéré.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import secrets
import threading
import time
from collections import Counter

from sft_miner.book import CursorBook

logger = logging.getLogger("sft_miner")


class VllmStream:
    """vLLM en flux : on ajoute des prompts, ``step`` rend ceux qui sont finis."""

    def __init__(self, generator, n: int, admit_below: float = 1.0) -> None:
        from vllm.sampling_params import RequestOutputKind

        self._engine = generator._llm.llm_engine
        self._capture = generator._capture
        self._proof = generator._proof
        self._params = generator._params.clone()
        self._params.output_kind = RequestOutputKind.FINAL_ONLY
        self.n = n
        self._req: dict[str, tuple[int, int]] = {}
        self._entry: dict[int, dict] = {}
        self.tokens_done = 0
        self.failures = 0
        self.admit_below = admit_below

    def kv_usage(self) -> float | None:
        """Occupation du cache KV (0..1), moteur en processus ; None si illisible."""
        try:
            return float(self._engine.engine_core.engine_core.scheduler.kv_cache_manager.usage)
        except Exception:
            return None

    def can_admit(self) -> bool:
        """Démarrer un prompt seulement s'il reste de la place : au-delà, vLLM
        évince et recalcule des séquences et leur preuve devient inutilisable
        (mesuré 27/09 : 66 régénérés sur 135 admis à 16 prompts fixes)."""
        if not self._entry:
            return True
        usage = self.kv_usage()
        return usage is None or usage < self.admit_below

    def add(self, cursor: int, prompt_ids: list[int]) -> None:
        from vllm.inputs import TokensPrompt

        nonce = secrets.token_hex(4)
        ids = []
        for i in range(self.n):
            # Toujours l'id EXTERNE : vLLM 0.30 renomme la requête en interne
            # (`<id>-<8 car.>`, valeur de retour ignorée), ses sorties portent
            # l'id externe, et la capture d'upstream retrouve l'interne par
            # préfixe `<id>-` — exactement comme `VllmGenerator.generate`.
            rid = f"c{cursor}-{i}-{nonce}"
            self._engine.add_request(rid, TokensPrompt(prompt_token_ids=prompt_ids), self._params)
            self._req[rid] = (cursor, i)
            ids.append(rid)
        self._entry[cursor] = {"ids": ids, "plen": len(prompt_ids), "gens": [None] * self.n,
                               "done": 0}

    def _drop(self, cursor: int, abort: bool) -> None:
        entry = self._entry.pop(cursor, None)
        if entry is None:
            return
        pending = [rid for rid in entry["ids"] if rid in self._req]
        for rid in pending:
            self._req.pop(rid, None)
            try:
                self._capture.pop(rid)
            except KeyError:
                pass
        if abort and pending:
            self._engine.abort_request(pending)

    def abort(self, cursor: int) -> None:
        self._drop(cursor, abort=True)

    def busy(self) -> bool:
        return self._engine.has_unfinished_requests()

    def step(self) -> list[tuple[int, list | None]]:
        from reliquary.miner.corpus_miner import Generation
        from reliquary.miner.vllm_hidden_capture import completion_rows
        from reliquary.protocol.toploc_proof import build_chunk_proofs

        results = []
        for out in self._engine.step():
            if not out.finished:
                continue
            key = self._req.pop(out.request_id, None)
            if key is None:
                # Requête d'un prompt déjà abandonné : libérer ses lignes.
                try:
                    self._capture.pop(out.request_id)
                except KeyError:
                    pass
                continue
            cursor, i = key
            entry = self._entry.get(cursor)
            if entry is None:
                continue
            tokens = list(out.outputs[0].token_ids)
            try:
                rows = completion_rows(self._capture.pop(out.request_id),
                                       entry["plen"], entry["plen"] + len(tokens))
            except (ValueError, KeyError) as exc:
                # Préemption/recalcul sous pression KV : aucune preuve fiable.
                logger.warning("curseur %d : génération inutilisable (%s), régénéré", cursor, exc)
                self.failures += 1
                self._drop(cursor, abort=True)
                results.append((cursor, None))
                continue
            proofs = build_chunk_proofs(rows, chunk_tokens=self._proof.chunk_tokens,
                                        topk=self._proof.topk)
            entry["gens"][i] = Generation(tokens, [base64.b64encode(p).decode() for p in proofs])
            entry["done"] += 1
            self.tokens_done += len(tokens)
            if entry["done"] == self.n:
                del self._entry[cursor]
                results.append((cursor, entry["gens"]))
        return results


def submit_loop(*, book: CursorBook, client, job, hotkey: str, tokenizer, sign, stats: Counter,
                dry_run: bool = False, sleep=time.sleep, dump=None) -> None:
    """Soumet dans l'ordre du curseur ; se réaligne sur le validateur après
    tout refus. Même traitement des raisons que ``mine_steps`` d'upstream."""
    from reliquary.miner import corpus_miner as cm

    retry = dict(sleep=sleep, counts=stats, max_consecutive_failures=cm._MAX_CONSECUTIVE_FAILURES)
    while not book.stopped:
        item = book.take_submittable()
        if item is None:
            sleep(0.05)
            continue
        cursor, (prompt_index, rendered, generations) = item
        body = cm.build_submission(job=job, hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
                                   rendered_prompt=rendered, generations=generations,
                                   tokenizer=tokenizer, sign=sign)
        if dump is not None:
            dump(prompt_index, rendered, generations)
        if dry_run:
            stats["dry_run_ready"] += 1
            stats["dry_run_tokens"] += sum(len(g.tokens) for g in generations)
            book.on_accepted(cursor)
            continue
        try:
            answer = cm._retry(lambda: client.submit(body), **retry)
        except cm.CorpusMinerHalted as exc:
            book.stop(f"halted: {exc}")
            break
        reason = str(answer.get("reason"))
        stats[reason] += 1
        if reason == "job_complete":
            book.stop("job_complete")
        elif reason in cm._HALT:
            logger.error("le validateur refuse cette hotkey : %s", reason)
            book.stop(reason)
        elif answer.get("accepted"):
            stats["tokens_accepted"] += sum(len(g.tokens) for g in generations)
            book.on_accepted(cursor)
        else:
            logger.warning("curseur %d refusé : %s %s", cursor, reason, answer.get("detail"))
            try:
                book.realign(int(cm._retry(lambda: client.cursor(hotkey), **retry)))
            except cm.CorpusMinerHalted as exc:
                book.stop(f"halted: {exc}")


def gpu_loop(*, book: CursorBook, stream, prompt_for, stats: Counter, deadline: float | None,
             log_every: float = 60.0, sleep=time.sleep, clock=time.time) -> None:
    """Remplit vLLM en continu, rend les prompts finis au carnet."""
    t0 = last = clock()
    tok0 = 0
    while not book.stopped:
        if deadline is not None and clock() >= deadline:
            book.stop("deadline")
            break
        for cursor in book.pop_aborts():
            stream.abort(cursor)
        # Au plus UN nouveau prompt par pas : la jauge du cache ne compte un
        # prompt qu'une fois préremplie, en admettre une rafale sous le seuil
        # le fait déborder (mesuré 27/09 : 5 → 24 en vol d'un coup, cache 99 %).
        can_admit = getattr(stream, "can_admit", lambda: True)
        if can_admit() and (cursor := book.next_admission()) is not None:
            prompt_index, rendered, ids = prompt_for(cursor)
            stream.add(cursor, ids)
            stats["admitted"] += 1
        if stream.busy():
            for cursor, generations in stream.step():
                if generations is None:
                    stats["regenerated"] += 1
                    book.on_failed(cursor)
                else:
                    prompt_index, rendered, _ = prompt_for(cursor)
                    book.on_generated(cursor, (prompt_index, rendered, generations))
        else:
            sleep(0.05)
        now = clock()
        if hasattr(prompt_for, "prune"):
            prompt_for.prune(book.snapshot()["next_submit"])
        if now - last >= log_every:
            rate = (stream.tokens_done - tok0) / (now - last)
            usage = getattr(stream, "kv_usage", lambda: None)()
            logger.info("débit %.0f tok/s (moyenne %.0f) | cache KV %s | carnet %s | compteurs %s",
                        rate, stream.tokens_done / (now - t0),
                        "?" if usage is None else f"{usage:.0%}", book.snapshot(), dict(stats))
            last, tok0 = now, stream.tokens_done


def _own_logging() -> None:
    fmt = logging.Formatter("%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s")
    for name in ("sft_miner", "reliquary.miner.corpus_miner"):
        lg = logging.getLogger(name)
        lg.handlers = []
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg.disabled = False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validator-url", default="http://62.238.81.36:8000")
    ap.add_argument("--wallet-name", default="camille81-v2")
    ap.add_argument("--hotkey", default="hotkey81.2")
    ap.add_argument("--max-in-flight", type=int, default=32, help="plafond de prompts dans vLLM (×4 séquences)")
    ap.add_argument("--max-ahead", type=int, default=256, help="curseurs d'avance sur la soumission")
    ap.add_argument("--admit-below", type=float, default=0.6,
                    help="n'admettre un prompt que si le cache KV est rempli à moins de cette part")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true", help="générer et prouver, ne rien soumettre")
    ap.add_argument("--duration", type=float, default=0, help="secondes (0 = sans fin)")
    ap.add_argument("--dump", type=int, default=0,
                    help="écrire les N premiers prompts prêts (ids + tokens + preuves) pour l'auto-audit")
    ap.add_argument("--dump-path", default="/workspace/sft_dump.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(threadName)s | %(message)s")
    if not os.environ.get("RELIQUARY_TASK_CONTRACT"):
        raise SystemExit("RELIQUARY_TASK_CONTRACT doit pointer le contrat de la tâche corpus")

    import httpx
    from huggingface_hub import snapshot_download
    from reliquary.corpus.encoding import checkpoint_fingerprint, prompt_token_ids
    from reliquary.corpus.job import parse_job
    from reliquary.corpus.walk import walk_index
    from reliquary.miner import corpus_miner as cm
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job

    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    if proof is None:
        raise SystemExit(f"le profil {ACTIVE_PROTOCOL_PROFILE.profile_id} ne déclare pas de preuve toploc")
    http = httpx.Client(base_url=a.validator_url, timeout=120.0)

    class _Client:
        def job(self):
            r = http.get("/corpus/job")
            r.raise_for_status()
            return r.json()

        def cursor(self, hk):
            return int(cm.issue_corpus_request(lambda: http.get(f"/corpus/cursor/{hk}"))["cursor"])

        def submit(self, body):
            return cm.issue_corpus_request(lambda: http.post("/corpus/submit", json=body))

    client = _Client()
    job = parse_job(client.job())
    directory = snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision)
    if checkpoint_fingerprint(directory) != job.checkpoint_sha256:
        raise SystemExit("le checkpoint téléchargé ne correspond pas à l'empreinte du job")
    tokenizer = load_tokenizer(directory)

    def encode(text):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    renderer = renderer_for_job(job, encode, tokenizer=tokenizer)
    prompts = prompt_job_for_spec(job)

    if a.dry_run:
        hotkey, sign = f"dry-run-{a.hotkey}", (lambda body: "")
        start = 0
    else:
        import bittensor as bt
        from reliquary.protocol.signatures import sign_corpus_submission

        wallet = bt.Wallet(name=a.wallet_name, hotkey=a.hotkey)
        # bittensor remet toute la journalisation au niveau WARNING à l'import :
        # nos journaux (et la raison d'un arrêt) disparaîtraient. On les rebranche.
        _own_logging()
        hotkey = wallet.hotkey.ss58_address
        sign = lambda body: sign_corpus_submission(wallet, body)  # noqa: E731
        start = client.cursor(hotkey)
    logger.info("job %s | hotkey %s | curseur de départ %d | %d prompts en vol max | dry_run=%s",
                job.job_id, hotkey, start, a.max_in_flight, a.dry_run)

    cache: dict[int, tuple] = {}

    def prompt_for(cursor):
        if cursor not in cache:
            index = walk_index(job.job_id, hotkey, cursor, job.prompt_count)
            rendered = renderer.initial_text(prompts.task_for(index))
            cache[cursor] = (index, rendered, prompt_token_ids(tokenizer, rendered))
        return cache[cursor]

    def _prune(below):
        for c in [c for c in cache if c < below]:
            del cache[c]

    prompt_for.prune = _prune

    generator = cm.VllmGenerator(directory, job.sampling, proof, job.eos_token_id,
                                 gpu_memory_utilization=a.gpu_memory_utilization)
    stream = VllmStream(generator, job.sampling.n, admit_below=a.admit_below)
    book = CursorBook(start=start, max_in_flight=a.max_in_flight, max_ahead=a.max_ahead)
    stats: Counter = Counter()
    dumped: list = []

    def dump(prompt_index, rendered, generations):
        # Format du banc (corpus_bench.py check) : une complétion par ligne.
        if len(dumped) >= a.dump:
            return
        ids = prompt_token_ids(tokenizer, rendered)
        dumped.extend({"prompt_ids": ids, "tokens": list(g.tokens), "proofs": list(g.proofs)}
                      for g in generations)
        if len(dumped) >= a.dump:
            import json
            with open(a.dump_path, "w") as fh:
                json.dump({"runs": [{"k": "flux", "completions": dumped}]}, fh)
            logger.info("auto-audit : %d complétions écrites dans %s", len(dumped), a.dump_path)

    sub = threading.Thread(target=submit_loop, name="soumission", daemon=True,
                           kwargs=dict(book=book, client=client, job=job, hotkey=hotkey,
                                       tokenizer=tokenizer, sign=sign, stats=stats,
                                       dry_run=a.dry_run, dump=dump if a.dump else None))
    sub.start()
    deadline = time.time() + a.duration if a.duration else None
    try:
        gpu_loop(book=book, stream=stream, prompt_for=prompt_for, stats=stats, deadline=deadline)
    finally:
        book.stop(book.stopped or "sortie")
        sub.join(timeout=150)
    logger.info("arrêt : %s | compteurs %s | %d tokens générés, %d prompts régénérés",
                book.stopped, dict(stats), stream.tokens_done, stream.failures)
    print(f"arrêt : {book.stopped} | compteurs {dict(stats)}", flush=True)
    return 0 if book.stopped in ("job_complete", "deadline") else 1


if __name__ == "__main__":
    raise SystemExit(main())
