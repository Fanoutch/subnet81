"""Reliquary CLI — mine and validate commands."""

import asyncio
import logging
import os
import threading

import typer

from reliquary.constants import DEFAULT_BASE_MODEL, DEFAULT_HF_REPO_ID, ENVIRONMENT_NAME, VALIDATOR_HTTP_PORT

app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    # ``%(threadName)s`` distinguishes the main asyncio loop from the
    # dedicated ``weight-setter`` thread (see ``validate`` below) when
    # tailing logs.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environment: str = typer.Option(ENVIRONMENT_NAME, help="Environment name"),
    validator_url: str = typer.Option(
        "",
        help=(
            "Override the validator URL (otherwise discovered from the metagraph). "
            "Useful for local testing — e.g. http://127.0.0.1:8888"
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary miner."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, env=%s)",
        network, netuid, environment,
    )

    async def _run():
        import bittensor as bt
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environment
        from reliquary.infrastructure.chain import get_subtensor, get_metagraph, NETUID
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2
        from reliquary.miner.vllm_backend import VLLMBackend
        from reliquary.shared.modeling import (
            MODEL_SNAPSHOT_ALLOW_PATTERNS,
            load_text_generation_model,
            load_tokenizer,
        )

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        # --- Resolve initial checkpoint from validator if available ---
        # CRITICAL: if snapshot_download fails here, vLLM initializes with the
        # BASE model. The first ckpt-advance path is supposed to call
        # backend.reload(snapshot_path), but on the bootstrap transition it
        # often doesn't (observed: vllm_backend.reload only fires on the
        # SECOND advance, leaving 5-10 min of generation against the base
        # model — every rollout fails distribution_suspicious because the
        # validator verifies under the trained checkpoint). Retry 5x with
        # exponential backoff (3s, 6s, 12s, 24s) to swallow transient HF
        # outages / network blips, and refuse to fall back to base unless
        # all retries genuinely failed.
        import asyncio as _async
        import time as _time
        initial_path = checkpoint  # fallback to --checkpoint arg
        snapshot_resolved = False
        for attempt in range(5):
            try:
                if validator_url:
                    url = validator_url
                else:
                    metagraph = await get_metagraph(subtensor, NETUID)
                    url = discover_validator_url(metagraph)

                import httpx
                from huggingface_hub import snapshot_download
                async with httpx.AsyncClient(timeout=30) as client:
                    state = await get_window_state_v2(url, client=client)
                if state.checkpoint_repo_id and state.checkpoint_revision:
                    logger.info(
                        "Validator at %s is on checkpoint %d (%s@%s). "
                        "Downloading to seed the miner model (attempt %d/5).",
                        url, state.checkpoint_n, state.checkpoint_repo_id,
                        state.checkpoint_revision[:12], attempt + 1,
                    )
                    initial_path = snapshot_download(
                        repo_id=state.checkpoint_repo_id,
                        revision=state.checkpoint_revision,
                        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
                    )
                    logger.info("Using initial checkpoint path: %s", initial_path)
                    snapshot_resolved = True
                    break
                else:
                    logger.info(
                        "Validator has no published checkpoint yet — using --checkpoint=%s",
                        checkpoint,
                    )
                    break  # nothing to download, normal startup
            except Exception as e:
                wait_s = 3.0 * (2 ** attempt)  # 3, 6, 12, 24, 48
                logger.warning(
                    "Bootstrap attempt %d/5 failed (%s); retrying in %.0fs",
                    attempt + 1, e, wait_s,
                )
                if attempt < 4:
                    await _async.sleep(wait_s)
        if not snapshot_resolved and initial_path == checkpoint:
            logger.warning(
                "WARNING: bootstrap could not pull validator snapshot after 5 "
                "retries. vLLM will start with --checkpoint=%s (BASE model). "
                "All rollouts will FAIL distribution_suspicious until the "
                "next checkpoint advance forces a backend.reload — expect "
                "5-10 min of unproductive generation.",
                checkpoint,
            )
        try:
            pass  # no-op so the original except below stays balanced
        except Exception as e:
            logger.warning(
                "Could not fetch validator checkpoint (%s); falling back to "
                "--checkpoint=%s",
                e, checkpoint,
            )

        # --- Load models from resolved path ---
        logger.info("Loading models from %s...", initial_path)
        tokenizer = load_tokenizer(initial_path)  # ensure_tokenizer_padding inside

        # Single H200: co-locate vLLM generation and HF GRAIL proof on cuda:0.
        # The 0.55 cap was calibrated on an 80 GiB card, where it left ~36 GiB
        # for the HF proof model plus a ~7 GiB fragmentation margin against the
        # reload cliff (0.60 failed ~1-in-5 reloads there). On a 143 GiB H200
        # the same fraction leaves ~64 GiB while the HF model uses ~12 GiB —
        # far more slack than intended, and measured 2026-07-21 at 60% memory /
        # 26% compute.
        #
        # ⚠ 0.75 WAS TESTED AND FAILS on this box: vLLM took 106.7 GiB and the
        # per-rollout GRAIL proof forward (~8 GiB spikes, engine.py
        # _pre_bake_entry) hit torch.OutOfMemoryError 45 times in 2h, silently
        # losing those bakes. The binding constraint is not the model footprint
        # but the proof forward's transient allocations living beside the KV
        # cache. 0.65 leaves ~50 GiB for a ~25 GiB HF process plus its spikes.
        # NOTE: bake batch size does NOT move this boundary — vLLM preallocates
        # the KV cache from the fraction alone, so more prompts fill an existing
        # pool rather than reserving more memory. Only this fraction does.
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in
        # scripts/launch_miner.sh) additionally keeps fragmentation from
        # re-creating the cliff. Override with RELIQUARY_VLLM_GPU_FRACTION.
        proof_device = "cuda:0"

        # Under forced-seed enforcement the engine ALWAYS runs the sync HF
        # `_generator_loop` (vLLM cannot apply the forced-seed LogitsProcessor,
        # so its tokens would fail seed-consistency) and reloads via hf_model —
        # the engine tolerates vllm_backend=None on both the generation and the
        # checkpoint-reload paths. Constructing a vLLM backend here is therefore
        # pure waste, AND on the qwen3_5 hybrid-GDN model vLLM's default GDN
        # prefill kernel JIT-crashes (ptxas). Skip it entirely under enforcement.
        from reliquary.constants import FORCED_SEED_ENFORCE as _FSE
        from reliquary.miner.engine import vllm_forced_seed_enabled
        if _FSE and not vllm_forced_seed_enabled():
            vllm_backend = None
            logger.info(
                "forced-seed enforcement ON → HF-only generation; vLLM backend "
                "disabled (avoids qwen3_5 GDN JIT crash, frees GPU memory)"
            )
        else:
            # forced_seed=True registers VLLMForcedSeedLogitsProcessor + loads
            # qwen3_5 on the Triton GDN backend, so phase-1 can run forced-seed on
            # vLLM (RELIQUARY_VLLM_FORCED_SEED=1). When enforcement is OFF this is
            # the legacy fast path (forced_seed False).
            vllm_backend = VLLMBackend(
                model_path=initial_path,
                tokenizer_path=checkpoint,  # base tokenizer (immutable across ckpts)
                gpu_id=0,
                gpu_memory_utilization=float(
                    os.environ.get("RELIQUARY_VLLM_GPU_FRACTION", "0.78")
                ),
                max_model_len=16384,
                dtype="bfloat16",
                forced_seed=(_FSE and vllm_forced_seed_enabled()),
            )
            if _FSE:
                logger.info(
                    "forced-seed enforcement ON + RELIQUARY_VLLM_FORCED_SEED=1 → "
                    "phase-1 generation on vLLM (forced-seed processor registered)"
                )
        vllm_model = None  # legacy attribute kept for fallback compatibility

        hf_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(proof_device).eval()

        env = load_environment(environment)
        engine = MiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            env,
            proof_gpu=0,
            validator_url_override=validator_url or None,
            vllm_backend=vllm_backend,
        )

        # Seed engine's _loaded_checkpoint_path so the first
        # maybe_pull_checkpoint sees we're already synced (skips redundant reload).
        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path

        logger.info("Miner ready. Entering main loop.")
        try:
            await engine.mine_window(subtensor, 0, use_drand=use_drand)
        except KeyboardInterrupt:
            logger.info("Miner interrupted by user")
        except Exception as e:
            logger.error("Mining loop crashed: %s", e, exc_info=True)
            raise

    asyncio.run(_run())


@app.command()
def validate(
    train: bool = typer.Option(
        True,
        "--train/--no-train",
        help=(
            "Run full trainer mode (default). "
            "Pass --no-train for weight-only mode: reads R2 archives, "
            "computes EMA, submits weights. No GPU, no HF, no HTTP server."
        ),
    ),
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(DEFAULT_BASE_MODEL, help="HF repo id or local path of the model to load (trainer mode only)"),
    environment: str = typer.Option(ENVIRONMENT_NAME, help="Environment name (trainer mode only)"),
    http_host: str = typer.Option("0.0.0.0", help="HTTP bind address (trainer mode only)"),
    http_port: int = typer.Option(VALIDATOR_HTTP_PORT, help="HTTP listen port (trainer mode only)"),
    external_ip: str = typer.Option(
        "",
        help=(
            "Public IP this validator is reachable at. Published on-chain via "
            "serve_axon so miners can discover it through the metagraph. "
            "Leave empty to skip publishing (miners then need --validator-url). "
            "Trainer mode only."
        ),
    ),
    external_port: int = typer.Option(
        0,
        help="Public port to advertise on-chain; defaults to --http-port when 0. Trainer mode only.",
    ),
    hf_repo_id: str = typer.Option(
        DEFAULT_HF_REPO_ID,
        help="HuggingFace repo ID to publish checkpoints to (must be writable with HF_TOKEN). Trainer mode only.",
    ),
    resume_from: str = typer.Option(
        os.getenv("RELIQUARY_RESUME_FROM", ""),
        help=(
            "Resume trainer from a checkpoint instead of the base model. "
            "Accepts 'sha:<40-hex>' (HF commit on --hf-repo-id) or "
            "'path:<dir>' (local ckpt_<N> directory). Trainer mode only."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary validator (trainer mode by default; --no-train for weight-only)."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    if train:
        logger.info(
            "Starting Reliquary validator [trainer] (network=%s, netuid=%d, env=%s, http=%s:%d)",
            network, netuid, environment, http_host, http_port,
        )
    else:
        logger.info(
            "Starting Reliquary validator [weight-only] (network=%s, netuid=%d)",
            network, netuid,
        )

    async def _run():
        import bittensor as bt

        from reliquary.infrastructure.chain import get_subtensor

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        if train:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from reliquary.constants import ATTN_IMPLEMENTATION
            from reliquary.environment import load_environment
            from reliquary.validator.service import ValidationService

            logger.info("Loading model from %s...", checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to("cuda:0").eval()

            env = load_environment(environment)
            service = ValidationService(
                wallet,
                model,
                tokenizer,
                env,
                netuid,
                use_drand=use_drand,
                http_host=http_host,
                http_port=http_port,
                external_ip=external_ip or None,
                external_port=(external_port or http_port) if external_ip else None,
                hf_repo_id=hf_repo_id,
                resume_from=resume_from or None,
            )
            # Run the weight setter in a dedicated OS thread with its own
            # event loop. asyncio is single-threaded, so any sync blocking
            # call on the trainer's loop (e.g. /state acquiring a lock the
            # GRAIL verifier is holding) would stall set_weights too. The
            # weight setter's own subtensor (see WeightOnlyValidator.run)
            # plus its own loop here means neither side can block the other.
            from reliquary.validator.weight_only import WeightOnlyValidator

            def _run_weight_setter() -> None:
                try:
                    worker = WeightOnlyValidator(wallet=wallet, netuid=netuid)
                    asyncio.run(worker.run())
                except Exception:
                    logger.exception("weight-setter thread crashed")

            threading.Thread(
                target=_run_weight_setter,
                name="weight-setter",
                daemon=True,
            ).start()
            await service.run(subtensor)
        else:
            from reliquary.validator.weight_only import WeightOnlyValidator

            validator = WeightOnlyValidator(wallet=wallet, netuid=netuid)
            await validator.run()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
