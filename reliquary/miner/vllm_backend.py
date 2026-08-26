"""vLLM generation backend for the private miner.

Wraps vllm.LLM with a thin synchronous API:
    backend = VLLMBackend(model_path, gpu_id=0)
    completions = backend.generate(prompt_tokens=[1,2,3], n=8, temperature=0.9, ...)
    backend.reload(new_model_path)

The engine calls `generate` from an asyncio coroutine via
`await asyncio.to_thread(backend.generate, ...)` to avoid blocking the loop.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

# vLLM sync ``LLM.generate`` n'est PAS thread-safe. Le pipeline chunké du
# bake (prefetch du chunk N+1 PENDANT le grading du chunk N) peut faire
# arriver un fallback per-prompt (cache phase-1 stale) en parallèle du
# prefetch — deux .generate() concurrents = corruption/crash moteur. Ce
# verrou sérialise uniquement l'appel moteur ; sans contention il ne
# coûte rien.
_VLLM_CALL_LOCK = threading.Lock()


def _kill_stale_engine_cores(grace_s: float = 3.0) -> int:
    """Tue les EngineCore vLLM zombies (enfants directs de CE processus).

    Incident 2026-08-16 13:24 : au reload de checkpoint, l'ancien EngineCore
    a gardé ses 115 GiB ~4 min au lieu de ~30 s → 4 _build_llm en init-OOM,
    fuite VRAM du processus principal, puis TOUTES les preuves GRAIL en OOM
    (revenu zéro, invisible du watchdog). N'est appelé que quand
    ``self._llm is None`` : tout EngineCore encore vivant est par définition
    indésirable (son pool est déjà jeté par drop-on-ckpt), le tuer ne perd
    aucun travail légitime. SIGTERM d'abord, SIGKILL après ``grace_s``.
    Ne lève jamais ; renvoie le nombre de processus tués.
    """
    import time as _time

    me = os.getpid()
    victims: list[int] = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat", "rb") as f:
                    # champ 4 = ppid ; comm (champ 2) peut contenir des
                    # espaces mais est parenthésé → couper après ')'.
                    stat = f.read().decode("ascii", "replace")
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
                if ppid != me:
                    continue
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
                if "EngineCore" in cmdline:
                    victims.append(pid)
            except (OSError, ValueError, IndexError):
                continue
        for pid in victims:
            try:
                os.kill(pid, 15)  # SIGTERM
            except OSError:
                pass
        if victims:
            deadline = _time.monotonic() + max(0.0, grace_s)
            while _time.monotonic() < deadline:
                if not any(os.path.exists(f"/proc/{p}") for p in victims):
                    break
                _time.sleep(0.2)
            for pid in victims:
                try:
                    os.kill(pid, 9)  # SIGKILL les survivants
                except OSError:
                    pass
            logger.warning(
                "_kill_stale_engine_cores: %d EngineCore zombie(s) tué(s) "
                "(pids=%s)", len(victims), victims,
            )
    except Exception:
        return 0
    return len(victims)


def _with_stop_token(completion_output, primary_eos_id: int | None = None) -> list[int]:
    """token_ids d'une CompletionOutput, stop-token RESTAURÉ s'il a été retiré.

    Observé live (fenêtres 27585/27586, verdicts bad_termination 7/7) : quand
    la génération s'arrête sur un stop-token SPÉCIAL (l'EOS <|im_end|>), vLLM
    l'exclut des token_ids — ``include_stop_str_in_output`` ne couvre que les
    stop STRINGS. Le validateur exige exactement UN EOS en DERNIÈRE position
    (admission._classify_termination) ; sans lui, 100% bad_termination.

    Le token est légitime : c'est le pick forcé réellement généré (c'est lui
    qui a déclenché l'arrêt), vLLM le signale via ``stop_reason``. Idempotent :
    si vLLM l'a déjà inclus, on ne double pas ; arrêt par cap
    (finish_reason='length', stop_reason=None) : on ne touche à rien.
    """
    ids = list(completion_output.token_ids)
    stop = getattr(completion_output, "stop_reason", None)
    if isinstance(stop, int):
        if not ids or ids[-1] != stop:
            ids.append(stop)
        return ids
    # stop_reason absent MAIS la generation s'est arretee => c'est l'EOS du
    # MODELE qui a stoppe, et vLLM l'a retire sans le nommer. Son identifiant
    # est connu et unique pour ce checkpoint (generation_config), donc on le
    # RECONSTRUIT : c'est le token que le forced-seed a reellement tire a cette
    # position, le teacher-forcing du validateur retombera dessus.
    # Sans cette branche ces rollouts n'ont AUCUN EOS -> la troncature n'a rien
    # a couper, la garde locale les jette, et le mineur ne soumet plus rien
    # (panne silencieuse). C'est la difference entre « ne pas envoyer de
    # mauvais » et « envoyer du bon ».
    # ⚠️ DESACTIVE PAR DEFAUT (RELIQUARY_RECONSTRUCT_EOS=1 pour activer).
    # Le checkpoint 4B declare DEUX EOS : 248044 <|endoftext|> (generation_
    # config) et 248046 <|im_end|> (tokenizer). On ne sait pas encore lequel
    # arrete reellement la generation par ce chemin. Apposer le mauvais =
    # soumettre un token JAMAIS genere -> le teacher-forcing du validateur
    # trouverait autre chose (SEED_MISMATCH / soupcon de falsification), pire
    # qu'un groupe jete. ops/probe_eos_behavior.py tranche en 2 min ; activer
    # SEULEMENT une fois le token confirme.
    if (
        primary_eos_id is not None
        and os.environ.get("RELIQUARY_RECONSTRUCT_EOS", "0") == "1"
        and getattr(completion_output, "finish_reason", None) == "stop"
        and (not ids or ids[-1] != primary_eos_id)
    ):
        ids.append(int(primary_eos_id))
    return ids

logger = logging.getLogger(__name__)


def _build_llm(
    model_path: str,
    gpu_id: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    tokenizer_path: Optional[str] = None,
    forced_seed: bool = False,
):
    """Construct a vllm.LLM. Wrapped in a function for mock-ability in tests.

    ``tokenizer_path`` lets us load weights from a trained checkpoint that
    doesn't ship a tokenizer (we strip tokenizer files from the snapshot to
    decouple the miner's transformers version from the validator's). When
    provided, vLLM reads the tokenizer from there instead of ``model_path``.

    ``forced_seed=True`` loads the qwen3.5-2b hybrid-GDN checkpoint the way it
    actually boots on vLLM 0.24 (proven by scripts/vllm_smoke_test.py): text-only
    multimodal, Triton/FLA GDN prefill (the validator's own path — flashinfer's
    GDN JIT crashes on ptxas), eager, remote code. It also DROPS ngram
    speculative decoding: under forced-seed every token must equal the public
    inverse-CDF pick, and spec-decoded tokens would fail the validator's
    seed-consistency check.
    """
    # Pin which CUDA device vLLM uses BEFORE importing it. vLLM picks up
    # CUDA_VISIBLE_DEVICES at import / engine init time.
    # Only set if the shell hasn't already pinned a device: with the dual
    # miner launcher, ``CUDA_VISIBLE_DEVICES=1`` is exported per-process
    # before launching, and overwriting it here would force both miners
    # back onto GPU 0.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # transformers 5.x removed `Tokenizer.all_special_tokens_extended` but
    # vLLM 0.7.3 still calls it during LLM init. Restore the attribute as
    # a thin alias of `all_special_tokens` before vLLM imports the
    # tokenizer wrapper. Idempotent: only patches if missing.
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    from vllm import LLM
    # vLLM 0.10+ accepts `speculative_config` as a dict; ngram speculation
    # gives ~1.5-2x throughput on math reasoning workloads. Compatible with
    # transformers 5.x as long as the all_special_tokens_extended shim
    # below has been applied before vLLM imports the tokenizer wrapper.
    # Disable via RELIQUARY_DISABLE_SPECULATIVE=1 if the validator's
    # distribution check (q10 of chosen-token probabilities) flags our
    # submissions — spec-decoded sequences can drift slightly off the
    # validator's HF-computed distribution and tank q10.
    kwargs = dict(
        model=model_path,
        tokenizer=tokenizer_path or model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        kv_cache_dtype="auto",
    )
    mns = vllm_max_num_seqs()
    if mns is not None:
        kwargs["max_num_seqs"] = mns
    if forced_seed:
        # qwen3.5-2b hybrid-GDN bring-up recipe (see reference_vllm_qwen35_2b
        # bringup): text-only, Triton/FLA GDN prefill, eager, remote code. NO
        # speculative_config — spec-decode breaks forced-seed consistency.
        # logits_processors: the forced-seed processor CLASS must be registered
        # at LLM init or per-request extra_args are silently ignored (tokens
        # would come out UNFORCED → SEED_MISMATCH). The async backend always
        # registered it; this sync path relied on callers doing it themselves.
        from reliquary.miner.vllm_forced_seed import (
            build_forced_seed_logitsproc_class,
        )
        kwargs.update(
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            additional_config={"gdn_prefill_backend": "triton"},
            enforce_eager=vllm_enforce_eager(),
            logits_processors=[build_forced_seed_logitsproc_class()],
        )
        # Guérison divergence (19/08 soir, audit parité) : nos 16 rollouts
        # partagent le même prompt → vLLM active le kernel CASCADE (calcul
        # partagé du préfixe), dont la numérique dépend de la forme du batch
        # — le validateur, lui, vérifie en teacher-forcing séquentiel, jamais
        # en cascade. Le désactiver aligne nos logits de décodage sur son
        # forward aux positions frontière de la CDF forcée (suspect n°1 des
        # token_tampered honnêtes). Coût : préfixe recalculé par rollout
        # (~-5-10 % de débit sur nos prompts courts). Kill-switch env.
        if os.environ.get("RELIQUARY_VLLM_DISABLE_CASCADE", "0") == "1":
            kwargs["disable_cascade_attn"] = True
    elif os.environ.get("RELIQUARY_DISABLE_SPECULATIVE", "0") != "1":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        }
    # Cache torch.compile FIGÉ (facultatif, cf. vllm_compile_cache_dir).
    # Sans lui, chaque avancée de checkpoint change le chemin du modèle donc
    # le config_hash donc le répertoire de cache → recompilation intégrale.
    # Mesuré sur la box le 25/08, deux démarrages du MÊME jour :
    #   cache MANQUÉ  : torch.compile 22,41 s, init engine 36,90 s
    #   cache TOUCHÉ  : torch.compile  4,23 s, init engine 11,32 s
    # soit 25,6 s de moins par rechargement. Le graphe ne dépend pas des
    # poids : `computation_graph.py` est identique octet pour octet entre
    # deux caches consécutifs, seul `config_hash` diffère.
    _cache_dir = vllm_compile_cache_dir(model_path, kwargs)
    if _cache_dir is not None:
        # vLLM ne calcule son hash QUE si cache_dir est absent
        # (compilation/backends.py ~1053 : `if not
        # self.compilation_config.cache_dir:`). Le fournir court-circuite le
        # hash sans toucher au reste de la CompilationConfig (mode, tailles
        # de capture, passes inductor restent aux défauts).
        kwargs["compilation_config"] = {"cache_dir": _cache_dir}
    return LLM(**kwargs)


def _build_sampling_params(
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    stop_token_ids: Optional[list[int]] = None,
):
    """Construct a vllm.SamplingParams. Wrapped for mock-ability in tests.

    The vllm import is local so the module imports cleanly on machines
    where vllm is not installed.
    """
    from vllm import SamplingParams
    # CRITICAL: pass EOS tokens explicitly. vLLM falls back to
    # ``tokenizer.eos_token_id`` only (e.g. Qwen3-4B reports just 151645
    # ``<|im_end|>``; 151643 ``<|endoftext|>`` is the pad_token and absent from
    # the R0mAI fine-tune's (missing) generation_config). The validator's
    # ``has_eos_padding`` check rejects any rollout with an EOS NOT at the last
    # position, so if the model samples an EOS that vLLM doesn't stop on, the
    # rollout becomes EOS-in-middle → ``bad_termination``.
    #
    # ``stop_token_ids`` is now resolved dynamically by the caller (the engine,
    # via ``resolve_eos_token_ids(hf_model, tokenizer)`` — covers Qwen3.5's
    # generation_config + nested text_config EOS set). Falls back to the
    # Qwen3-4B pair {151643, 151645} when the caller passes nothing, so older
    # call paths keep their previous behaviour.
    if not stop_token_ids:
        stop_token_ids = [151643, 151645]
    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_token_ids=list(stop_token_ids),
        include_stop_str_in_output=True,
    )


def vllm_max_num_seqs(env=None):
    """Optional vLLM scheduler cap (``max_num_seqs``) from
    ``RELIQUARY_VLLM_MAX_NUM_SEQS``. Bench 2026-08-03 (4B, H100, batched
    forced-seed processor): 512 → 5859 tok/s at batch 40 — concurrency scaling
    reopened once the per-row Python hook was gone. Unset/invalid → None (keep
    vLLM's default)."""
    src = os.environ if env is None else env
    raw = src.get("RELIQUARY_VLLM_MAX_NUM_SEQS")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def vllm_enforce_eager() -> bool:
    """``enforce_eager`` a passer a vLLM. True = pas de CUDA graphs (defaut).

    Les CUDA graphs valent **+23% de debit a batch egal** (banc 2026-07-22,
    H100 128 sequences : 2116 vs 1725 tok/s) et battent meme un batch de 256
    sequences en eager. Conformite forced-seed verifiee : seed_consistency
    0.9880 groupe / 0.9531 pire rollout (planchers 0.80 / 0.75).

    Le defaut reste EAGER : c'est le reglage du bring-up, adopte parce que le
    JIT des kernels GDN de ce modele hybride crashait. On n'active les graphes
    que sur demande explicite (``RELIQUARY_VLLM_CUDA_GRAPHS=1``), pour qu'une
    box neuve demarre toujours sur le chemin eprouve — et re-passer le gate
    avant de s'y fier sur un nouveau GPU.
    """
    return os.environ.get("RELIQUARY_VLLM_CUDA_GRAPHS", "0") != "1"


class VLLMBackend:
    """Lazy-initialized vLLM engine on a specific GPU."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        tokenizer_path: Optional[str] = None,
        forced_seed: bool = False,
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._tokenizer_path = tokenizer_path
        self._forced_seed = forced_seed
        self._llm: Optional[object] = None
        # Drapeau d'interruption engine-wide (fix hot-swap 2026-08-18) : le
        # driver multi-stream le vérifie à chaque step — permet au chemin
        # checkpoint-advance d'obtenir _VLLM_CALL_LOCK en ~1 step au lieu
        # d'attendre la fin complète du bake (gels du 15/08).
        self._interrupt = threading.Event()

    def _ensure_loaded(self) -> None:
        """Lazy-build the vLLM LLM, with retry on ckpt-advance OOM.

        H100 80GB hardening: at ckpt-advance the old vLLM's EngineCore
        subprocess takes 3-10 s to actually release its KV cache + weights
        back to the device. The first attempt can therefore see less free
        VRAM than ``gpu_memory_utilization * total`` requests and raise::

            ValueError: Free memory on device (N/total GiB) on startup is
            less than desired GPU memory utilization ...

        We catch that specific class of init failure (ValueError +
        RuntimeError from vLLM's launch_core_engines wrapper), force gc +
        empty_cache, sleep, and retry. Without this the generator loop
        re-enters _ensure_loaded() in a tight loop, each attempt racing
        the same stale memory, and the miner stays in a permanent
        zero-vLLM state for the rest of the process lifetime.
        """
        if self._llm is not None:
            return
        last_exc: Optional[BaseException] = None
        for attempt in range(5):
            try:
                self._llm = _build_llm(
                    model_path=self._model_path,
                    gpu_id=self._gpu_id,
                    gpu_memory_utilization=self._gpu_memory_utilization,
                    max_model_len=self._max_model_len,
                    dtype=self._dtype,
                    tokenizer_path=self._tokenizer_path,
                    forced_seed=self._forced_seed,
                )
                if attempt > 0:
                    logger.info(
                        "vLLM _build_llm succeeded on retry attempt %d",
                        attempt + 1,
                    )
                return
            except (ValueError, RuntimeError) as exc:
                last_exc = exc
                msg = str(exc)
                # Only retry on the OOM-during-init signature; let other
                # build errors (bad path, bad config, etc.) fail fast.
                is_oom = (
                    "Free memory on device" in msg
                    or "Engine core initialization failed" in msg
                )
                if not is_oom:
                    raise
                logger.warning(
                    "vLLM _build_llm attempt %d/5 hit init-time OOM "
                    "(%s); gc+sleep+retry",
                    attempt + 1,
                    msg.split("\n", 1)[0][:200],
                )
                try:
                    import gc as _gc
                    import time as _time
                    # Dès le 2e échec, la mort naturelle de l'ancien moteur a
                    # eu sa chance : on tue les EngineCore zombies plutôt que
                    # de retenter dans le mur (incident 2026-08-16 13:24 —
                    # 4 échecs en cascade + fuite VRAM du processus principal).
                    if attempt >= 1:
                        _kill_stale_engine_cores()
                    _gc.collect()
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Backoff: 3s, 6s, 12s, 24s. Caps total at ~45s — past
                    # which the stale vLLM is either dead or genuinely stuck.
                    _time.sleep(3.0 * (2 ** attempt))
                except Exception:
                    pass
        # All retries exhausted; surface the original error so the engine
        # loop can fall back to a no-op iteration rather than spin.
        assert last_exc is not None
        raise last_exc

    def generate(
        self,
        prompt_token_ids: list[int],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """Sample `n` completions for the given prompt; return token-id lists.

        Returns a list of length `n`; each element is the list of generated
        token IDs (NOT including the prompt). ``stop_token_ids`` should be the
        model's full EOS set (resolved by the engine); falls back to the
        Qwen3-4B pair inside ``_build_sampling_params`` when omitted.
        """
        self._ensure_loaded()
        sampling_params = _build_sampling_params(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )
        # vLLM 0.10+ no longer accepts ``prompt_token_ids=`` as a kwarg on
        # ``LLM.generate``; pass a TokensPrompt instead. Backwards-compat
        # with older vLLM versions is left to the operator.
        from vllm.inputs import TokensPrompt
        with _VLLM_CALL_LOCK:
            outputs = self._llm.generate(
                [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling_params=sampling_params,
        )
        # outputs is a list of length 1 (we passed one prompt).
        # outputs[0].outputs is a list of n CompletionOutput objects.
        return [list(co.token_ids) for co in outputs[0].outputs]

    def generate_forced_phase1(
        self,
        prompt_token_ids: list[int],
        *,
        randomness: str,
        prompt_idx: int,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
        primary_eos_id: Optional[int] = None,
    ) -> list[list[int]]:
        """BFT phase-1 under forced-seed: ``m_rollouts`` completions, each forced
        onto its own ``rollout_index`` stream by the engine-registered
        VLLMForcedSeedLogitsProcessor. Greedy (``temperature=0``) so the argmax is
        the forced token. One batched ``LLM.generate`` call (continuous batching
        across the M rollouts). Returns generated token lists (prompt excluded);
        phase-2 continues on HF via ``engine._bft_from_seqs``.
        """
        self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )
        start_len = len(prompt_token_ids)
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)
                   for _ in range(m_rollouts)]
        sps = [
            SamplingParams(
                n=1, temperature=0.0, max_tokens=max_tokens,
                stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                # EOS d'arrêt INCLUS dans les token_ids — sans ça vLLM le retire,
                # le rollout soumis ne finit pas par EOS → bad_termination au
                # verdict (3/3 observés fenêtre 27585). Même piège documenté dans
                # _build_sampling_params.
                include_stop_str_in_output=True,
                # ignore_eos: l'arret par l'EOS MOTEUR (config modele) strippe
                # le token avec stop_reason=None -> impossible a restaurer
                # (verdicts bad_termination 27590 malgre le re-append). En ne
                # laissant QUE stop_token_ids stopper, stop_reason porte
                # toujours l'ID exact a restaurer.
                ignore_eos=True,
                extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                    randomness=randomness, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash, rollout_index=r,
                    base_offset=0, start_len=start_len)},
            )
            for r in range(m_rollouts)
        ]
        with _VLLM_CALL_LOCK:
            outputs = self._llm.generate(prompts, sampling_params=sps)
        return [_with_stop_token(out.outputs[0], primary_eos_id)
                for out in outputs]

    def generate_forced_phase1_multi(
        self,
        prompts_token_ids: list[list[int]],
        *,
        prompt_indices: list[int],
        randomness: str,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
        primary_eos_id: Optional[int] = None,
    ) -> list[list[list[int]]]:
        """Forced-seed phase-1 for MANY prompts in ONE batched ``generate`` call.

        Same contract as ``generate_forced_phase1`` but across prompts: every
        (prompt, rollout) pair becomes its own sequence carrying its own
        ``prompt_idx``/``start_len`` in ``extra_args``, so the engine-registered
        processor forces each stream independently.

        Why this exists: under forced-seed the bake loop used to call
        ``generate_forced_phase1`` once per prompt, so N prompts cost N x ~44s.
        With a 100s collection window that meant a 6-prompt bake (~264s) was
        always flushed stale at the randomness flip and never submitted.
        Batching turns it into roughly one 44s call.

        Returns ``[[rollout_tokens] * m_rollouts] * len(prompts)`` in input order.
        """
        if len(prompts_token_ids) != len(prompt_indices):
            raise ValueError(
                f"prompts_token_ids ({len(prompts_token_ids)}) and prompt_indices "
                f"({len(prompt_indices)}) must have the same length; zip would "
                f"silently mislabel forced-seed streams"
            )
        self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )

        prompts = []
        sps = []
        for tokens, prompt_idx in zip(prompts_token_ids, prompt_indices):
            start_len = len(tokens)
            for r in range(m_rollouts):
                prompts.append(TokensPrompt(prompt_token_ids=tokens))
                sps.append(SamplingParams(
                    n=1, temperature=0.0, max_tokens=max_tokens,
                    # ignore_eos : seul stop_token_ids stoppe → stop_reason
                    # porte toujours l'ID du token d'arrêt à restaurer (l'EOS
                    # moteur strippe avec stop_reason=None, irrécupérable —
                    # verdicts bad_termination 27590 malgré le ré-append).
                    ignore_eos=True,
                    stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                    # EOS d'arrêt INCLUS dans les token_ids — sans ça vLLM le retire,
                    # le rollout soumis ne finit pas par EOS → bad_termination au
                    # verdict (3/3 observés fenêtre 27585). Même piège documenté dans
                    # _build_sampling_params.
                    include_stop_str_in_output=True,
                    extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness=randomness, prompt_idx=prompt_idx,
                        checkpoint_hash=checkpoint_hash, rollout_index=r,
                        base_offset=0, start_len=start_len)},
                ))

        with _VLLM_CALL_LOCK:
            outputs = self._llm.generate(prompts, sampling_params=sps)
        flat = [_with_stop_token(out.outputs[0], primary_eos_id)
                for out in outputs]
        _o0 = outputs[0].outputs[0] if outputs and outputs[0].outputs else None
        if _o0 is not None:
            _ids0 = flat[0] if flat else []
            logger.info(
                "phase1[diag] finish=%s stop_reason=%r last_tok=%s in_stop=%s n_seq=%d",
                getattr(_o0, "finish_reason", None),
                getattr(_o0, "stop_reason", None),
                _ids0[-1] if _ids0 else None,
                bool(_ids0) and stop_token_ids and _ids0[-1] in set(stop_token_ids),
                len(outputs),
            )
        # regroup: sequences were emitted prompt-major, m_rollouts per prompt
        return [
            flat[i * m_rollouts:(i + 1) * m_rollouts]
            for i in range(len(prompts_token_ids))
        ]

    def _stream_request_id(self, pos: int, r: int) -> str:
        """Request id du streaming — surchargable par les tests pour scripter
        l'ordre d'achèvement. Compteur propre : unique par appel ET par vie du
        backend (deux streams successifs ne doivent jamais réutiliser un id
        encore vivant dans le moteur)."""
        c = getattr(self, "_stream_counter", 0) + 1
        self._stream_counter = c
        return f"fs{c}-p{pos}-r{r}"

    def generate_forced_phase1_multi_stream(
        self,
        prompts_token_ids: list[list[int]],
        *,
        prompt_indices: list[int],
        randomness: str,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
        primary_eos_id: Optional[int] = None,
        on_group=None,
        should_abort=None,
        sprint_size: int = 0,
        sprint_max_wait_s: float = 20.0,
    ) -> list[list[list[int]]]:
        """Phase-1 forcée en STREAMING : chaque groupe livré dès qu'il finit.

        Pourquoi : le départage d'enchère v3 est ``min(tokens, cap) /
        (round_arrivée - round_ouverture)`` — chaque seconde d'attente divise le
        rang. ``generate`` (monolithique) ne rend rien tant que TOUTES les
        séquences ne sont pas finies : un k=2 prêt à ~25 s attendait les
        traînards du plafond (~50 s) puis le grading — nos soumissions
        partaient à 60-110 s de l'ouverture contre 20-45 s pour le peloton
        (rang 26 sur un k=2 à score maximal, fenêtre 27872).

        Comment : on pilote LE MÊME moteur sync pas à pas (``add_request`` +
        ``step()`` — exactement ce que ``generate`` fait en interne). Mêmes
        kernels, même processeur forced-seed engine-level, mêmes ``extra_args``
        que ``generate_forced_phase1_multi`` : la parité certifiée par la gate
        4B (PASS 0.9793, 2026-08-06) transfère telle quelle.

        ``on_group(pos, prompt_idx, [tokens]*m)`` est invoqué à la COMPLÉTION
        de chaque prompt. ``should_abort()`` (vérifié à chaque step) permet
        d'interrompre au flip de fenêtre : le reste est avorté (il serait jeté
        hors-tranche de toute façon) et le GPU passe à la nouvelle tranche.

        SPRINT (``sprint_size`` > 0) : les ``sprint_size`` PREMIERS prompts
        (l'ordre d'entrée est l'ordre du classement prédicteur) sont enfilés
        SEULS — moins de séquences en vol = décodage par séquence plus rapide,
        le meilleur candidat arrive des rounds plus tôt (le départage entre
        k=2 à égalité est tokens/rounds-depuis-ouverture). Le BALAYAGE (le
        reste) est enfilé dès la livraison du dernier groupe du sprint, ou
        après ``sprint_max_wait_s`` si un prompt du sprint file vers le
        plafond — un tronqueur ne doit pas retenir la couverture de fenêtre.
        Les extra_args sont identiques au chemin non-sprinté (mêmes flux).

        Retourne l'agrégat du chemin batché (drop-in) ; les groupes avortés
        sont absents du retour.
        """
        if len(prompts_token_ids) != len(prompt_indices):
            raise ValueError(
                f"prompts_token_ids ({len(prompts_token_ids)}) and "
                f"prompt_indices ({len(prompt_indices)}) must have the same "
                f"length; zip would silently mislabel forced-seed streams"
            )
        self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )

        engine = self._llm.llm_engine
        n = len(prompts_token_ids)
        rid_to_slot: dict[str, tuple[int, int]] = {}
        groups: list[list] = [[None] * m_rollouts for _ in range(n)]
        remaining = [m_rollouts] * n
        delivered = [False] * n

        n_sprint = max(0, min(int(sprint_size or 0), n))
        if n_sprint >= n:
            n_sprint = 0        # sprint = tout le lot : aucun sens, tout part
        scan_started = n_sprint == 0

        def _enqueue(pos_range):
            for pos in pos_range:
                tokens = prompts_token_ids[pos]
                prompt_idx = prompt_indices[pos]
                start_len = len(tokens)
                for r in range(m_rollouts):
                    rid = self._stream_request_id(pos, r)
                    rid_to_slot[rid] = (pos, r)
                    engine.add_request(
                        rid,
                        TokensPrompt(prompt_token_ids=tokens),
                        SamplingParams(
                            n=1, temperature=0.0, max_tokens=max_tokens,
                            ignore_eos=True,
                            stop_token_ids=(
                                list(stop_token_ids) if stop_token_ids else None
                            ),
                            include_stop_str_in_output=True,
                            extra_args={
                                FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                                    randomness=randomness,
                                    prompt_idx=prompt_idx,
                                    checkpoint_hash=checkpoint_hash,
                                    rollout_index=r,
                                    base_offset=0, start_len=start_len,
                                )
                            },
                        ),
                    )

        import time as _time_mod
        with _VLLM_CALL_LOCK:
            _enqueue(range(n_sprint if not scan_started else n))
            sprint_t0 = _time_mod.monotonic()

            def _maybe_start_scan(reason):
                nonlocal scan_started
                if scan_started:
                    return
                scan_started = True
                _enqueue(range(n_sprint, n))
                logger.info(
                    "sprint: balayage enclenché à %.1fs (%s) — %d prompts de "
                    "sprint, %d de balayage",
                    _time_mod.monotonic() - sprint_t0, reason, n_sprint,
                    n - n_sprint,
                )

            aborted = False
            # getattr défensif : les tests (et tout outillage) construisent des
            # backends partiels via __new__ qui sautent __init__ — sans lui, la
            # boucle crashe AttributeError au lieu de streamer (réconciliation
            # hot-swap 18/08).
            _interrupt = getattr(self, "_interrupt", None)
            while engine.has_unfinished_requests():
                if (_interrupt is not None and _interrupt.is_set()) or (
                        should_abort is not None and should_abort()):
                    pending = [
                        rid for rid, (pos, _r) in rid_to_slot.items()
                        if not delivered[pos]
                        and any(
                            groups[pos][rr] is None
                            for rr in range(m_rollouts)
                        )
                    ]
                    # n'avorte que les requêtes pas encore terminées
                    pending = [
                        rid for rid in pending
                        if groups[rid_to_slot[rid][0]][rid_to_slot[rid][1]]
                        is None
                    ]
                    if pending:
                        try:
                            engine.abort_request(pending)
                        except TypeError:
                            for rid in pending:
                                engine.abort_request(rid)
                    aborted = True
                    break
                if not scan_started and (
                    _time_mod.monotonic() - sprint_t0 >= sprint_max_wait_s
                ):
                    # un prompt de sprint file vers le plafond : il ne doit
                    # pas retenir la couverture de la fenêtre
                    _maybe_start_scan("délai")
                for out in engine.step():
                    if not getattr(out, "finished", False):
                        continue
                    slot = rid_to_slot.get(out.request_id)
                    if slot is None:
                        continue
                    pos, r = slot
                    groups[pos][r] = _with_stop_token(
                        out.outputs[0], primary_eos_id,
                    )
                    remaining[pos] -= 1
                    if remaining[pos] == 0 and not delivered[pos]:
                        delivered[pos] = True
                        if not scan_started and all(
                            delivered[q] for q in range(n_sprint)
                        ):
                            _maybe_start_scan("sprint livré")
                        if on_group is not None:
                            try:
                                on_group(
                                    pos, prompt_indices[pos], groups[pos],
                                )
                            except Exception:
                                # un callback qui lève ne doit jamais tuer le
                                # décodage des autres groupes
                                logger.exception(
                                    "on_group failed for prompt=%d",
                                    prompt_indices[pos],
                                )
            if aborted:
                logger.info(
                    "phase1 stream: abandon demandé (flip de fenêtre) — "
                    "%d/%d groupes livrés", sum(delivered), n,
                )

        return [
            groups[pos] if delivered[pos] else []
            for pos in range(n)
        ]

    def generate_multi(
        self,
        prompts_token_ids: list[list[int]],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[list[int]]]:
        """Generate ``n`` completions for each prompt in ``prompts_token_ids``.

        Issued as a single ``LLM.generate`` call so vLLM's continuous
        batching can interleave the rollouts of all prompts on the same
        GPU step — keeps the SM busy while individual rollouts trickle in
        and out. This is the win on a single big GPU (H200) where one
        prompt × 8 rollouts leaves the device under-utilised between
        cycles.

        Returns a list parallel to ``prompts_token_ids``: index ``i``
        holds the ``n`` token-id lists generated for prompt ``i``.
        """
        self._ensure_loaded()
        sampling_params = _build_sampling_params(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )
        from vllm.inputs import TokensPrompt
        with _VLLM_CALL_LOCK:
            outputs = self._llm.generate(
                [TokensPrompt(prompt_token_ids=p) for p in prompts_token_ids],
            sampling_params=sampling_params,
        )
        # outputs is parallel to prompts. outputs[i].outputs is the list
        # of n CompletionOutput objects for prompt i.
        return [
            [list(co.token_ids) for co in out.outputs]
            for out in outputs
        ]

    def reload(self, new_model_path: str) -> None:
        """Swap to a new checkpoint. Drops the current LLM and BLOCKS
        until its VRAM is actually released.

        The next call to `generate()` will rebuild against `new_model_path`.

        H100 80GB hardening: a naive ``del self._llm; empty_cache()`` is
        racy — vLLM's EngineCore subprocess shutdown is async (Popen
        signal, OS reaps, CUDA driver returns memory) and takes 3-10 s.
        If the next ``_ensure_loaded()`` fires before the old subprocess
        releases, the new ``LLM()`` init OOMs (only ~10 GiB free while
        it wants ~32 GiB) and the miner enters a crash loop. We poll
        ``torch.cuda.mem_get_info`` and wait until the device has enough
        free memory for the new vLLM to fit (``gpu_memory_utilization``
        × total + 5 GiB safety), or 30 s max — whichever comes first.

        Cost on the happy path: an extra ~3-8 s of synchronous wait at
        ckpt-advance. Worth it vs the indefinite zombie state we get
        without the wait.
        """
        if self._llm is not None:
            # Drop the Python reference so refcount can hit zero and
            # vLLM's __del__ chain (which signals the EngineCore subprocess
            # to terminate) starts.
            self._llm = None
            try:
                import gc as _gc
                import time as _time
                import torch as _torch

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _, total_b = _torch.cuda.mem_get_info()
                        total_gib = total_b / (1024 ** 3)
                    except Exception:
                        total_gib = 80.0  # H100 fallback
                    target_free_gib = self._gpu_memory_utilization * total_gib + 5.0

                    waited = 0.0
                    deadline = 30.0
                    # Plateau = l'ancien EngineCore est mort et la mémoire ne
                    # bougera plus. La cible uti*total+5 est INATTEIGNABLE
                    # quand le modèle de preuve HF réside sur le même GPU
                    # (mesuré 2026-08-12 13:28 : free plafonné à 112.5 GiB
                    # pour cible 114 → 30 s brûlées à CHAQUE reload). On ne
                    # veut attendre que la mort de l'ancien moteur, pas un
                    # niveau absolu : 3 lectures stables (±0.25 GiB) suffisent.
                    last_free = None
                    initial_free = None
                    stable = 0
                    while waited < deadline:
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                            if initial_free is None:
                                initial_free = free_gib
                            if free_gib >= target_free_gib:
                                logger.info(
                                    "vllm_backend.reload: %.1f/%.1f GiB free "
                                    "(target %.1f) after %.1fs — proceeding",
                                    free_gib, total_gib, target_free_gib, waited,
                                )
                                break
                            # Un plateau ne vaut que si la LIBÉRATION a été
                            # observée (free remonté d'au moins 10 GiB) ou si
                            # on est déjà près de la cible. Incident 22:22 le
                            # 2026-08-12 : stable à 10 GiB AVANT le début de
                            # la libération → init à 10 GiB → OOM (retry 1/5).
                            plateau_credible = (
                                free_gib >= initial_free + 10.0
                                or free_gib >= target_free_gib - 5.0
                            )
                            if (
                                plateau_credible
                                and last_free is not None
                                and abs(free_gib - last_free) < 0.25
                            ):
                                stable += 1
                                if stable >= 3:
                                    logger.info(
                                        "vllm_backend.reload: free plafonné à "
                                        "%.1f/%.1f GiB (cible %.1f inatteignable"
                                        " — modèle de preuve résident) après "
                                        "%.1fs — proceeding",
                                        free_gib, total_gib, target_free_gib,
                                        waited,
                                    )
                                    break
                            else:
                                stable = 0
                            last_free = free_gib
                        except Exception:
                            break
                        _time.sleep(1.0)
                        waited += 1.0
                        _gc.collect()
                        try:
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        # Loop exhausted without break — log and let the
                        # next _ensure_loaded() retry-loop handle it.
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                        except Exception:
                            free_gib = -1
                        logger.warning(
                            "vllm_backend.reload: timed out waiting for "
                            "VRAM release (free=%.1f, target=%.1f); "
                            "_ensure_loaded will retry",
                            free_gib, target_free_gib,
                        )
                        # L'ancien moteur ne mourra plus tout seul : on le
                        # tue et on laisse ~10 s à la VRAM pour revenir,
                        # pour que le premier _build_llm parte propre
                        # (incident 2026-08-16 13:24).
                        if _kill_stale_engine_cores() > 0:
                            for _ in range(10):
                                _time.sleep(1.0)
                                try:
                                    free_b, _ = _torch.cuda.mem_get_info()
                                    if (free_b / (1024 ** 3)
                                            >= target_free_gib - 5.0):
                                        break
                                except Exception:
                                    break
            except Exception:
                # Best-effort cleanup; never raise from reload.
                pass
        self._model_path = new_model_path

    def reload_weights_inplace(self, new_model_path: str,
                               lock_timeout_s: float = 15.0) -> bool:
        """Échange des poids À CHAUD dans le moteur vivant (checkpoint-advance).

        Le rebuild complet coûte ~150 s (poids + graphs + warmup) → toute
        fenêtre dont la collecte chevauche un reload est perdue (28964 le
        2026-08-15, pendant que d'autres mineurs la servaient). Le nouveau
        checkpoint = même architecture → vLLM 0.24 sait recharger les poids
        en place (``Worker.reload_weights``), sans recapture des graphs :
        ~5-15 s. À n'appeler que génération QUIESCÉE (chemin ckpt-advance :
        le bake en vol est déjà abandonné/périmé).

        Retourne True si l'échange a réussi ; False (jamais d'exception) →
        l'appelant retombe sur ``reload()`` complet, l'état d'avant.
        L'appelant DOIT ensuite passer sa propre gate de cohérence forced-seed
        (cf. engine._hot_swap_self_gate) avant de re-générer pour de vrai.
        """
        import os as _os
        if _os.environ.get("RELIQUARY_HOT_SWAP", "0") != "1":
            return False
        if self._llm is None:
            return False  # rien de chargé : le chemin normal construira
        import time as _time
        # Fix 2026-08-18 (gels du 15/08) : le driver multi-stream tient
        # _VLLM_CALL_LOCK pendant TOUT le bake, et un ckpt-advance arrive
        # souvent en pleine fenêtre (should_abort ne flippe qu'au flip de
        # randomness). On lève donc le drapeau d'interruption (le driver
        # avorte ses requêtes engine et libère le verrou en ~1 step) puis on
        # prend le verrou AVEC TIMEOUT — jamais d'attente non bornée : à
        # défaut, False → l'appelant fait le rebuild complet, comme avant.
        self._interrupt.set()
        acquired = _VLLM_CALL_LOCK.acquire(timeout=float(lock_timeout_s))
        self._interrupt.clear()
        if not acquired:
            logger.warning(
                "reload_weights_inplace: verrou moteur non obtenu en %.0fs "
                "— repli sur le rebuild complet", lock_timeout_s,
            )
            return False
        t0 = _time.monotonic()
        try:
            self._llm.collective_rpc(
                "reload_weights", kwargs={"weights_path": new_model_path},
            )
            # ⛔ SÛRETÉ FORCED-SEED — SANS CECI L'ÉCHANGE À CHAUD EST FAUX.
            # `enable_prefix_caching=True` est le DÉFAUT de vLLM 0.24 et
            # notre moteur tourne bien avec (dump de config du 25/08). Or
            # `GPUModelRunner.reload_weights` remet à zéro le cache encodeur
            # et le cache multimodal, mais PAS le cache de préfixe : des
            # blocs KV calculés avec les ANCIENS poids survivent à l'échange
            # et seraient réutilisés par le nouveau modèle. Les logits
            # divergeraient de ceux que le validateur reconstitue en
            # teacher-forcing → SEED_MISMATCH en masse.
            # Le rebuild complet n'a jamais eu ce problème : il détruit le
            # processus EngineCore, donc le cache avec.
            # Le risque est CONCRET chez nous : le slot mémo re-sélectionne
            # délibérément des prompts déjà joués, donc un préfixe déjà
            # caché est re-soumis peu après l'échange.
            self._llm.reset_prefix_cache()
            self._model_path = new_model_path
            logger.info(
                "vllm_backend.reload_weights_inplace: poids échangés à chaud "
                "en %.1fs (%s)", _time.monotonic() - t0, new_model_path,
            )
            return True
        except Exception:
            logger.exception(
                "reload_weights_inplace: échec — repli sur le rebuild complet"
            )
            return False
        finally:
            _VLLM_CALL_LOCK.release()

    def generate_forced_probe(self, prompt_ids, n_tokens: int, *,
                              randomness: str, checkpoint_hash: str):
        """Sonde de l'auto-gate hot-swap : n_tokens forcés (greedy, ignore_eos)
        sur un prompt arbitraire — les ids retournés sont vérifiés par
        teacher-forcing contre le modèle de preuve HF (engine._hot_swap_self_gate).
        """
        self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )
        sp = SamplingParams(
            n=1, temperature=0.0, max_tokens=int(n_tokens), ignore_eos=True,
            extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                randomness=randomness, prompt_idx=0,
                checkpoint_hash=checkpoint_hash, rollout_index=0,
                base_offset=0, start_len=len(prompt_ids),
            )},
        )
        # Fix 2026-08-18 : la sonde steppait le moteur SANS _VLLM_CALL_LOCK,
        # en concurrence avec le driver du bake (sync generate non
        # thread-safe) — un des deux ingrédients des gels du 15/08.
        with _VLLM_CALL_LOCK:
            out = self._llm.generate(
                [TokensPrompt(prompt_token_ids=list(prompt_ids))], sp)
        return list(out[0].outputs[0].token_ids)

    def warmup(self, max_tokens: int = 16) -> Optional[float]:
        """Paie le coût de (re)construction TOUT DE SUITE au lieu du premier
        bake de la fenêtre suivante : build moteur (poids + capture CUDA
        graphs, dans ``_ensure_loaded``) puis une mini-génération forced-seed
        pour déclencher les JIT Triton (prefill GDN + processeur).

        Sans ça, le premier ``generate()`` post-reload paie 130-400 s
        (mesuré : fenêtre 28569 perdue le 2026-08-12, premier cycle de boot
        à 132 s) — appelé au checkpoint-advance, ce coût tombe dans le temps
        mort post-flush du pool où rien d'autre n'est possible.

        Best-effort : ne lève JAMAIS (un warmup raté laisse exactement
        l'état d'avant le fix — le premier bake paiera). Retourne la durée
        en secondes, ou None sur échec.
        """
        import time as _time
        t0 = _time.monotonic()
        try:
            self._ensure_loaded()
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from reliquary.miner.vllm_forced_seed import (
                FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
            )
            ids = [1] * 8
            sp = SamplingParams(
                n=1, temperature=0.0, max_tokens=int(max_tokens),
                ignore_eos=True,
                extra_args={FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                    randomness="00" * 32, prompt_idx=0,
                    checkpoint_hash="warmup", rollout_index=0,
                    base_offset=0, start_len=len(ids),
                )},
            )
            self._llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)
            dt = _time.monotonic() - t0
            logger.info("vllm_backend.warmup: moteur chaud en %.1fs", dt)
            return dt
        except Exception:
            logger.exception(
                "vllm_backend.warmup: échec (non fatal — le premier bake "
                "paiera le rebuild comme avant)"
            )
            return None


# ---------------------------------------------------------------------------
# AsyncLLM backend — feature-flagged, opt-in via RELIQUARY_ASYNC_LLM=1.
#
# vLLM's ``AsyncLLMEngine`` lets us push prompt requests at any time and
# stream completed rollouts as they finish. This unlocks pipeline overlap:
# while one prompt's rollouts are running through HF proof + submit, the
# next prompt is already generating on the GPU. On a single H200 (where
# the sync miner sits idle ~25-35% of the cycle waiting on HF/HTTP), this
# closes most of that gap. Sync ``VLLMBackend`` above is unchanged so
# Targon (vllm 0.7.3 — different async API) keeps working as-is.
# ---------------------------------------------------------------------------


class AsyncVLLMBackend:
    """Async vLLM engine for pipeline-overlapped mining.

    API mirrors ``VLLMBackend`` but ``generate`` is an async coroutine,
    and a streaming variant ``add_request`` returns an async iterator over
    completed ``RequestOutput`` objects.

    Validated against vllm 0.10.2 only. Older vllm (0.7.x) has a
    different ``AsyncLLMEngine`` import path / signature; if you flip the
    feature flag on a 0.7.x box, expect import errors.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        dtype: str = "bfloat16",
        tokenizer_path: Optional[str] = None,
        forced_seed: bool = False,
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._tokenizer_path = tokenizer_path
        self._forced_seed = forced_seed
        self._engine = None
        self._lock = None  # asyncio.Lock, lazy-init on first call
        self._req_counter = 0

    async def _ensure_loaded(self) -> None:
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._engine is not None:
                return

            # Same shim + CUDA pinning logic as the sync path. The shim
            # patches transformers 5.x for vllm 0.10.2 compat and must run
            # before any vllm submodule import.
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_id)

            from transformers import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
                PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens
                )

            from vllm import AsyncEngineArgs, AsyncLLMEngine
            engine_kwargs = dict(
                model=self._model_path,
                tokenizer=self._tokenizer_path or self._model_path,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                dtype=self._dtype,
                kv_cache_dtype="auto",
                disable_log_stats=True,
            )
            mns = vllm_max_num_seqs()
            if mns is not None:
                engine_kwargs["max_num_seqs"] = mns
            if self._forced_seed:
                # qwen3.5-2b hybrid-GDN bring-up recipe + register the forced-seed
                # batch logits processor (per-request payload via extra_args). NO
                # speculative_config — spec-decode breaks seed-consistency.
                from reliquary.miner.vllm_forced_seed import (
                    build_forced_seed_logitsproc_class,
                )
                engine_kwargs.update(
                    trust_remote_code=True,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    additional_config={"gdn_prefill_backend": "triton"},
                    enforce_eager=vllm_enforce_eager(),
                    logits_processors=[build_forced_seed_logitsproc_class()],
                )
            elif os.environ.get("RELIQUARY_DISABLE_SPECULATIVE", "0") != "1":
                engine_kwargs["speculative_config"] = {
                    "method": "ngram",
                    "num_speculative_tokens": 5,
                    "prompt_lookup_max": 4,
                }
            args = AsyncEngineArgs(**engine_kwargs)
            # Mirror sync _VLLMBackend._ensure_loaded: at ckpt-advance the
            # previous EngineCore subprocess takes 3-10 s to release its KV
            # cache. If reload()'s VRAM-poll didn't catch the full release
            # (or wasn't called — first build path), retry on init-time OOM
            # rather than die and leave the miner generator-less.
            last_exc: Optional[BaseException] = None
            for attempt in range(5):
                try:
                    self._engine = AsyncLLMEngine.from_engine_args(args)
                    if attempt > 0:
                        logger.info(
                            "AsyncLLMEngine.from_engine_args succeeded on "
                            "retry attempt %d", attempt + 1,
                        )
                    return
                except (ValueError, RuntimeError) as exc:
                    last_exc = exc
                    msg = str(exc)
                    is_oom = (
                        "Free memory on device" in msg
                        or "Engine core initialization failed" in msg
                    )
                    if not is_oom:
                        raise
                    logger.warning(
                        "AsyncLLMEngine.from_engine_args attempt %d/5 hit "
                        "init-time OOM (%s); gc+sleep+retry",
                        attempt + 1,
                        msg.split("\n", 1)[0][:200],
                    )
                    try:
                        import asyncio as _asyncio
                        import gc as _gc
                        _gc.collect()
                        try:
                            import torch as _torch
                            if _torch.cuda.is_available():
                                _torch.cuda.empty_cache()
                        except Exception:
                            pass
                        await _asyncio.sleep(3.0 * (2 ** attempt))
                    except Exception:
                        pass
            assert last_exc is not None
            raise last_exc

    def _next_request_id(self) -> str:
        self._req_counter += 1
        return f"reliquary-{os.getpid()}-{self._req_counter}"

    async def generate(
        self,
        prompt_token_ids: list[int],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """Sample ``n`` completions, await them all, return token lists.

        Workaround for vLLM 0.10.2 AsyncLLMEngine bug: with ``n>1`` and
        concurrent requests, the engine silently drops samples (returns
        partial ``output.outputs`` under KV cache preemption). We submit
        ``n`` independent ``n=1`` requests instead and gather their
        results. vLLM internally still continuous-batches them all.
        """
        await self._ensure_loaded()
        from vllm.inputs import TokensPrompt

        async def _one_sample() -> list[int]:
            sp = _build_sampling_params(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
            )
            request_id = self._next_request_id()
            results_iter = self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sp,
                request_id=request_id,
            )
            final_output = None
            async for output in results_iter:
                final_output = output
            if final_output is None or not final_output.outputs:
                raise RuntimeError(
                    f"AsyncLLMEngine yielded no output for {request_id}"
                )
            return list(final_output.outputs[0].token_ids)

        import asyncio
        results = await asyncio.gather(*[_one_sample() for _ in range(n)])
        return results

    async def generate_forced_phase1(
        self,
        prompt_token_ids: list[int],
        *,
        randomness: str,
        prompt_idx: int,
        checkpoint_hash: str,
        m_rollouts: int,
        max_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
    ) -> list[list[int]]:
        """BFT phase-1 on vLLM under forced-seed: ``m_rollouts`` completions,
        each forced onto its own ``rollout_index`` stream by the engine-registered
        ``VLLMForcedSeedLogitsProcessor`` (payload threaded via extra_args).

        Sampling is greedy (``temperature=0``): the processor masks every logit
        to ``-inf`` except the inverse-CDF pick (set to ``0.0``), so the greedy
        argmax IS the forced token. Returns generated token lists (prompt
        excluded); phase-2 (answer) continues on the HF path via
        ``engine._bft_from_seqs``.
        """
        await self._ensure_loaded()
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from reliquary.miner.vllm_forced_seed import (
            FORCED_SEED_EXTRA_KEY, forced_seed_extra_args,
        )

        start_len = len(prompt_token_ids)

        async def _one_rollout(rollout_index: int) -> list[int]:
            sp = SamplingParams(
                n=1,
                temperature=0.0,  # greedy → picks the forced (0.0) token
                max_tokens=max_tokens,
                stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                # EOS d'arrêt INCLUS dans les token_ids — sans ça vLLM le retire,
                # le rollout soumis ne finit pas par EOS → bad_termination au
                # verdict (3/3 observés fenêtre 27585). Même piège documenté dans
                # _build_sampling_params.
                include_stop_str_in_output=True,
                extra_args={
                    FORCED_SEED_EXTRA_KEY: forced_seed_extra_args(
                        randomness=randomness, prompt_idx=prompt_idx,
                        checkpoint_hash=checkpoint_hash,
                        rollout_index=rollout_index,
                        base_offset=0, start_len=start_len,
                    )
                },
            )
            request_id = self._next_request_id()
            results_iter = self._engine.generate(
                TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sp,
                request_id=request_id,
            )
            final_output = None
            async for output in results_iter:
                final_output = output
            if final_output is None or not final_output.outputs:
                raise RuntimeError(
                    f"AsyncLLMEngine yielded no output for {request_id}"
                )
            return list(final_output.outputs[0].token_ids)

        import asyncio
        return await asyncio.gather(
            *[_one_rollout(r) for r in range(m_rollouts)]
        )

    async def reload(self, new_model_path: str) -> None:
        """Swap checkpoint. Drains the engine, rebuilds on next ``generate``.

        Mirrors the sync ``_VLLMBackend.reload`` hardening: after dropping
        the engine reference we poll ``torch.cuda.mem_get_info`` until the
        device actually has enough free VRAM for the next ``_ensure_loaded``
        to fit (``gpu_memory_utilization × total + 5 GiB``), or 30 s max.
        Without this, the next ``_ensure_loaded`` races the still-alive
        EngineCore subprocess and OOMs at init.
        """
        if self._engine is not None:
            try:
                await self._engine.shutdown_background_loop()
            except Exception:
                pass
            self._engine = None
            try:
                import asyncio
                import gc as _gc
                import torch as _torch

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _, total_b = _torch.cuda.mem_get_info()
                        total_gib = total_b / (1024 ** 3)
                    except Exception:
                        total_gib = 80.0
                    target_free_gib = self._gpu_memory_utilization * total_gib + 5.0

                    waited = 0.0
                    deadline = 30.0
                    while waited < deadline:
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                            if free_gib >= target_free_gib:
                                logger.info(
                                    "AsyncVLLMBackend.reload: %.1f/%.1f GiB free "
                                    "(target %.1f) after %.1fs — proceeding",
                                    free_gib, total_gib, target_free_gib, waited,
                                )
                                break
                        except Exception:
                            break
                        await asyncio.sleep(1.0)
                        waited += 1.0
                        _gc.collect()
                        try:
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        try:
                            free_b, _ = _torch.cuda.mem_get_info()
                            free_gib = free_b / (1024 ** 3)
                        except Exception:
                            free_gib = -1
                        logger.warning(
                            "AsyncVLLMBackend.reload: timed out waiting for "
                            "VRAM release (%.1f GiB free, target %.1f); "
                            "_ensure_loaded will retry",
                            free_gib, target_free_gib,
                        )
            except Exception:
                pass
        self._model_path = new_model_path


def vllm_compile_cache_key(model_path: str, kwargs: dict, env=None) -> str:
    """Clé d'identité du GRAPHE compilé, indépendante du checkpoint.

    vLLM dérive son ``config_hash`` (donc le répertoire de cache
    torch.compile) du CHEMIN du modèle. Comme chaque checkpoint est un
    snapshot HF différent, le hash change à chaque avancée → recompilation
    intégrale (22,4 s) alors que le graphe est identique. Mesuré sur la box
    le 25/08 : deux répertoires de cache consécutifs ne diffèrent que par
    ``config_hash`` ; ``computation_graph.py`` est identique octet pour
    octet.

    On remplace donc le chemin du modèle par ce qui détermine RÉELLEMENT le
    graphe :
      * le sha256 du ``config.json`` du snapshot (l'architecture) — vérifié
        byte-identique sur 4 révisions consécutives du repo de checkpoints ;
      * les kwargs moteur qui changent les formes/kernels compilés.
    Si le validateur publie un jour une architecture différente, le sha
    change → nouveau répertoire → aucune réutilisation illégitime.
    """
    import hashlib
    import json as _json

    h = hashlib.sha256()
    try:
        with open(os.path.join(model_path, "config.json"), "rb") as fh:
            h.update(fh.read())
    except OSError:
        # Pas de config.json lisible : on retombe sur le chemin, donc sur le
        # comportement actuel (une clé par checkpoint). Jamais d'exception.
        h.update(str(model_path).encode())
    shape_keys = (
        "max_model_len", "max_num_seqs", "dtype", "enforce_eager",
        "disable_cascade_attn", "kv_cache_dtype", "trust_remote_code",
    )
    shape = {k: kwargs.get(k) for k in shape_keys}
    shape["additional_config"] = kwargs.get("additional_config")
    h.update(_json.dumps(shape, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def vllm_compile_cache_dir(model_path: str, kwargs: dict, env=None):
    """Répertoire de cache torch.compile FIGÉ, ou None (défaut = vLLM décide).

    Armé par ``RELIQUARY_VLLM_COMPILE_CACHE_DIR`` (racine). Non défini →
    retourne None → ``_build_llm`` ne passe rien à vLLM → comportement
    byte-identique à aujourd'hui. C'est le repli : vider la variable.

    Le chemin porte la version de vLLM : une mise à jour de vLLM ne doit
    JAMAIS réutiliser des artefacts compilés par la version précédente.
    """
    src = os.environ if env is None else env
    root = src.get("RELIQUARY_VLLM_COMPILE_CACHE_DIR")
    if not root:
        return None
    try:
        import vllm as _vllm
        version = str(getattr(_vllm, "__version__", "unknown"))
    except Exception:
        version = "unknown"
    key = vllm_compile_cache_key(model_path, kwargs, env=src)
    return os.path.join(root, f"vllm-{version}", key)
