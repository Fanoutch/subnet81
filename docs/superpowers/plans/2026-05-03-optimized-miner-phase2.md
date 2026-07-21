# Optimized Miner — Phase 2 Implementation Plan (vLLM generation backend)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HF `.generate()` on GPU 0 with a real vLLM engine to cut generation time by ~2-3× on 2× H100, while keeping HF teacher-forcing on GPU 1 for GRAIL proof bit-identicality.

**Architecture:** Introduce `reliquary/miner/vllm_backend.py` exposing a `VLLMBackend` class that wraps vLLM's sync `LLM` API. `MiningEngine` swaps its `vllm_model` HF attribute for a `VLLMBackend` instance; `_generate_m_rollouts` calls `backend.generate(prompt_tokens, n=8, ...)` and adapts the token-id output into the existing rollout dict format. Checkpoint reload drops the HF-on-GPU-0 path and triggers `backend.reload(new_path)` instead, which kills+recreates the vLLM `LLM` instance (no hot-reload, since vLLM's `update_weights_from_disk` isn't reliable across versions).

**Tech Stack:** Python 3.11, PyTorch, HuggingFace transformers, vLLM, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-03-optimized-miner-design.md` sections 5.1 (vLLM config), 5.2 (HF proof), 8 Phase 2 (criteria).

**Hardware assumption:** 2× H100 80 GB. GPU 0 hosts vLLM (~8 GB weights + ~50-60 GB KV cache). GPU 1 hosts HF proof (~8 GB weights + workspace). Single-GPU setup is **out of scope for Phase 2** (would require co-locating vLLM + HF on one card with capped KV cache; deferred to a possible Phase 2.5).

**Phase 1 prerequisite:** branch `priv` at commit `13aa4b5` or later (Phase 1 code complete, 46 tests passing in `tests/miner_priv/`).

---

## Architecture decisions

1. **vLLM sync API wrapped in `asyncio.to_thread`** — simpler than `AsyncLLMEngine`, avoids async runtime conflicts, and the `await asyncio.to_thread(...)` cost is negligible vs the 3-5 s of generation.

2. **vLLM API: `LLM.generate(prompt_token_ids=..., sampling_params=...)`** — pass pre-tokenized prompts to avoid tokenizer divergence. The HF tokenizer is the canonical one (loaded by upstream code via `AutoTokenizer`). vLLM internally also has a tokenizer but we never use it for prompts.

3. **Hot-reload: kill + recreate** — `del backend._llm; backend._llm = None; backend._model_path = new_path`, lazy-reinit on next `generate()`. Cost ~20-30 s per checkpoint advance. Checkpoints publish ~every 5-10 min, so the miner loses at most ~1 window per advance. Acceptable.

4. **Output format unchanged** — `_generate_m_rollouts` still returns `list[dict]` with `{"tokens": prompt_tokens + completion_tokens, "prompt_length": int}`. Downstream `_build_rollout_submission` and GRAIL proof are untouched.

5. **No CLI flag changes** — vLLM config (`gpu_memory_utilization`, `max_model_len`, etc.) hardcoded in this plan. A future config pass can expose them.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `reliquary/miner/vllm_backend.py` | Create | `VLLMBackend` class — lazy init, generate, reload |
| `reliquary/miner/engine.py` | Modify | Hold `VLLMBackend` instead of HF model on GPU 0; route `_generate_m_rollouts` to backend; drop HF reload on GPU 0 |
| `reliquary/cli/main.py` | Modify | Stop loading HF model on GPU 0; pass `initial_path` and `vllm_gpu_id` to `MiningEngine` instead |
| `tests/miner_priv/test_vllm_backend.py` | Create | Unit tests for `VLLMBackend` with `vllm.LLM` mocked |
| `tests/miner_priv/test_engine_phase2.py` | Create | Integration smoke test for engine wiring (skipped if vllm not installed) |

---

## Task 0: Verify environment

**Files:** none (env check only).

**IMPORTANT:** This plan supports **dev on a CPU-only box** (mocked tests) **+ smoke test later on a 2× H100 box** (Task 8). vLLM is NOT installed during dev — all unit tests mock the vLLM seams. The first time vLLM is actually installed is on the GPU box, just before Task 8.

- [ ] **Step 1: Confirm Python + dev deps are intact**

```bash
cd ~/reliquary-miner-priv && source .venv/bin/activate
python -c "import torch, transformers, pytest, hypothesis; print('deps ok')"
```

Expected: `deps ok`. If anything is missing, run `pip install -e . pytest hypothesis` to recover.

- [ ] **Step 2: Detect environment (GPU vs CPU)**

```bash
python -c "import torch; n = torch.cuda.device_count(); print('GPU count:', n, '|', 'mode:', 'GPU-test' if n >= 2 else 'CPU-dev')"
```

- If `mode: GPU-test`: proceed to optionally install vLLM (Step 3). Task 8 will run.
- If `mode: CPU-dev`: skip Step 3, proceed to Task 1. Task 8 will be skipped with a STOP-AND-REPORT message.

- [ ] **Step 3: (GPU mode only) Install vLLM and pin version**

Skip this step if Step 2 reported CPU-dev mode.

```bash
pip install "vllm>=0.6.5,<0.7"
python -c "import vllm; print(vllm.__version__)" > .vllm-version
git add .vllm-version
git commit -m "chore(miner-priv): pin vllm version for Phase 2"
```

If the install fails on a GPU box (CUDA mismatch, etc.), report verbatim and stop.

---

## Task 1: `VLLMBackend` skeleton + lazy init

**Files:**
- Create: `reliquary/miner/vllm_backend.py`
- Test: `tests/miner_priv/test_vllm_backend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/miner_priv/test_vllm_backend.py`:

```python
"""Unit tests for VLLMBackend with vllm.LLM mocked.

These tests do NOT require a GPU — they patch the vllm LLM class so the
backend's lazy-init path can be exercised on CPU.
"""
from unittest.mock import patch, MagicMock
import pytest

from reliquary.miner.vllm_backend import VLLMBackend


def test_init_does_not_load_engine():
    """Construction is lazy — no LLM instance built until first generate()."""
    backend = VLLMBackend(model_path="/fake/path", gpu_id=0)
    assert backend._llm is None
    assert backend._model_path == "/fake/path"
    assert backend._gpu_id == 0


def test_default_config_values():
    backend = VLLMBackend(model_path="/fake/path", gpu_id=0)
    assert backend._gpu_memory_utilization == 0.85
    assert backend._max_model_len == 4096
    assert backend._dtype == "bfloat16"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/reliquary-miner-priv && source .venv/bin/activate
pytest tests/miner_priv/test_vllm_backend.py -v
```

Expected: ImportError (`reliquary.miner.vllm_backend` does not exist).

- [ ] **Step 3: Write the skeleton**

Create `reliquary/miner/vllm_backend.py`:

```python
"""vLLM generation backend for the private miner.

Wraps vllm.LLM with a thin synchronous API:
    backend = VLLMBackend(model_path, gpu_id=0)
    completions = backend.generate(prompt_tokens=[1,2,3], n=8, temperature=0.9, ...)
    backend.reload(new_model_path)   # kill+recreate on checkpoint advance

The engine calls `generate` from an asyncio coroutine via
`await asyncio.to_thread(backend.generate, ...)` to avoid blocking the loop.
"""
from __future__ import annotations

from typing import Optional


class VLLMBackend:
    """Lazy-initialized vLLM engine on a specific GPU.

    Instantiation is cheap; the actual `vllm.LLM` is built on the first
    `generate()` call. `reload(new_path)` deletes the engine and resets the
    lazy-init pointer so the next `generate()` rebuilds against new weights.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._llm: Optional[object] = None  # vllm.LLM instance, lazy
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_vllm_backend.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/vllm_backend.py tests/miner_priv/test_vllm_backend.py
git commit -m "feat(miner-priv): VLLMBackend skeleton with lazy init"
```

---

## Task 2: `VLLMBackend.generate()` — main API

**Files:**
- Modify: `reliquary/miner/vllm_backend.py`
- Modify: `tests/miner_priv/test_vllm_backend.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/miner_priv/test_vllm_backend.py`:

```python
@patch("reliquary.miner.vllm_backend._build_llm")
def test_generate_returns_n_token_id_lists(mock_build):
    """generate(n=8) returns 8 list-of-int completions."""
    fake_llm = MagicMock()
    fake_outputs_list = [
        MagicMock(token_ids=[10, 20, 30]),
        MagicMock(token_ids=[40, 50]),
        MagicMock(token_ids=[60]),
        MagicMock(token_ids=[70, 80, 90, 100]),
        MagicMock(token_ids=[110]),
        MagicMock(token_ids=[120, 130]),
        MagicMock(token_ids=[140]),
        MagicMock(token_ids=[150, 160, 170]),
    ]
    fake_request_output = MagicMock(outputs=fake_outputs_list)
    fake_llm.generate.return_value = [fake_request_output]
    mock_build.return_value = fake_llm

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    result = backend.generate(
        prompt_token_ids=[1, 2, 3],
        n=8,
        temperature=0.9,
        top_p=1.0,
        top_k=-1,
        max_tokens=1500,
    )

    assert len(result) == 8
    assert result[0] == [10, 20, 30]
    assert result[3] == [70, 80, 90, 100]


@patch("reliquary.miner.vllm_backend._build_llm")
def test_generate_lazy_inits_engine_once(mock_build):
    """Multiple generate() calls reuse the same LLM instance."""
    fake_llm = MagicMock()
    fake_request_output = MagicMock(outputs=[MagicMock(token_ids=[1]) for _ in range(8)])
    fake_llm.generate.return_value = [fake_request_output]
    mock_build.return_value = fake_llm

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    backend.generate(prompt_token_ids=[1], n=8)
    backend.generate(prompt_token_ids=[2], n=8)

    assert mock_build.call_count == 1


@patch("reliquary.miner.vllm_backend._build_sampling_params")
@patch("reliquary.miner.vllm_backend._build_llm")
def test_generate_passes_sampling_params(mock_build_llm, mock_build_sp):
    """The kwargs passed to _build_sampling_params reflect what the caller asked for.

    `_build_sampling_params` is the seam where the actual vLLM `SamplingParams`
    is constructed; mocking it lets us assert what would be passed without
    requiring vllm to be installed.
    """
    fake_llm = MagicMock()
    fake_request_output = MagicMock(outputs=[MagicMock(token_ids=[1]) for _ in range(4)])
    fake_llm.generate.return_value = [fake_request_output]
    mock_build_llm.return_value = fake_llm
    mock_build_sp.return_value = MagicMock(name="sampling_params_sentinel")

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    backend.generate(
        prompt_token_ids=[5, 6, 7],
        n=4,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        max_tokens=512,
    )

    mock_build_sp.assert_called_once_with(
        n=4, temperature=0.7, top_p=0.95, top_k=50, max_tokens=512,
    )
    # And the sentinel was forwarded into _llm.generate as sampling_params=
    fake_llm.generate.assert_called_once()
    call_kwargs = fake_llm.generate.call_args.kwargs
    assert call_kwargs["sampling_params"] is mock_build_sp.return_value
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/miner_priv/test_vllm_backend.py -v
```

Expected: 3 new failures (`generate` not defined, `_build_llm` not defined).

- [ ] **Step 3: Write the implementation**

Replace the contents of `reliquary/miner/vllm_backend.py` with:

```python
"""vLLM generation backend for the private miner.

Wraps vllm.LLM with a thin synchronous API:
    backend = VLLMBackend(model_path, gpu_id=0)
    completions = backend.generate(prompt_tokens=[1,2,3], n=8, temperature=0.9, ...)
    backend.reload(new_model_path)

The engine calls `generate` from an asyncio coroutine via
`await asyncio.to_thread(backend.generate, ...)` to avoid blocking the loop.
"""
from __future__ import annotations

import os
from typing import Optional


def _build_llm(
    model_path: str,
    gpu_id: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
):
    """Construct a vllm.LLM. Wrapped in a function for mock-ability in tests.

    The vllm import is local so the module imports cleanly on machines
    where vllm is not installed (CPU dev boxes). Tests patch this seam.
    """
    # Pin which CUDA device vLLM uses BEFORE importing it. vLLM picks up
    # CUDA_VISIBLE_DEVICES at import / engine init time.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM
    return LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
    )


def _build_sampling_params(
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
):
    """Construct a vllm.SamplingParams. Wrapped for mock-ability in tests.

    The vllm import is local so the module imports cleanly on machines
    where vllm is not installed.
    """
    from vllm import SamplingParams
    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )


class VLLMBackend:
    """Lazy-initialized vLLM engine on a specific GPU."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._llm: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._llm is None:
            self._llm = _build_llm(
                model_path=self._model_path,
                gpu_id=self._gpu_id,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                dtype=self._dtype,
            )

    def generate(
        self,
        prompt_token_ids: list[int],
        n: int = 8,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 1500,
    ) -> list[list[int]]:
        """Sample `n` completions for the given prompt; return token-id lists.

        Returns a list of length `n`; each element is the list of generated
        token IDs (NOT including the prompt). EOS handling is left to the
        caller — vLLM stops at its own EOS token id for the loaded model.
        """
        self._ensure_loaded()
        sampling_params = _build_sampling_params(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        outputs = self._llm.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
        )
        # outputs is a list of length 1 (we passed one prompt).
        # outputs[0].outputs is a list of n CompletionOutput objects.
        return [list(co.token_ids) for co in outputs[0].outputs]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_vllm_backend.py -v
```

Expected: 5 passed (2 from Task 1 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/vllm_backend.py tests/miner_priv/test_vllm_backend.py
git commit -m "feat(miner-priv): VLLMBackend.generate via vllm.LLM with SamplingParams"
```

---

## Task 3: `VLLMBackend.reload()` — kill + recreate

**Files:**
- Modify: `reliquary/miner/vllm_backend.py`
- Modify: `tests/miner_priv/test_vllm_backend.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/miner_priv/test_vllm_backend.py`:

```python
@patch("reliquary.miner.vllm_backend._build_llm")
def test_reload_swaps_model_path_and_clears_engine(mock_build):
    """After reload, the next generate() rebuilds with the new path."""
    fake_llm_a = MagicMock()
    fake_llm_b = MagicMock()
    fake_request_output = MagicMock(outputs=[MagicMock(token_ids=[1]) for _ in range(8)])
    fake_llm_a.generate.return_value = [fake_request_output]
    fake_llm_b.generate.return_value = [fake_request_output]
    mock_build.side_effect = [fake_llm_a, fake_llm_b]

    backend = VLLMBackend(model_path="/old", gpu_id=0)
    backend.generate(prompt_token_ids=[1], n=8)   # builds fake_llm_a

    backend.reload(new_model_path="/new")
    assert backend._model_path == "/new"
    assert backend._llm is None

    backend.generate(prompt_token_ids=[1], n=8)   # rebuilds → fake_llm_b
    assert mock_build.call_count == 2
    assert mock_build.call_args.kwargs["model_path"] == "/new"


@patch("reliquary.miner.vllm_backend._build_llm")
def test_reload_before_first_generate_is_safe(mock_build):
    """reload() before any generate() just updates the path."""
    backend = VLLMBackend(model_path="/old", gpu_id=0)
    backend.reload(new_model_path="/new")
    assert backend._model_path == "/new"
    assert backend._llm is None
    # mock_build NOT called yet
    assert mock_build.call_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/miner_priv/test_vllm_backend.py -v
```

Expected: 2 new failures (`reload` not defined).

- [ ] **Step 3: Write the implementation**

Append to `reliquary/miner/vllm_backend.py` (inside `VLLMBackend`):

```python
    def reload(self, new_model_path: str) -> None:
        """Swap to a new checkpoint. Deletes the current LLM instance.

        The next call to `generate()` will rebuild against `new_model_path`.
        Cost: ~20-30 s on a 4B model on H100 due to weight loading +
        KV cache preallocation. Acceptable since checkpoints publish on
        the order of every 5-10 minutes.
        """
        if self._llm is not None:
            del self._llm
            self._llm = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._model_path = new_model_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_vllm_backend.py -v
```

Expected: 7 passed (5 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/vllm_backend.py tests/miner_priv/test_vllm_backend.py
git commit -m "feat(miner-priv): VLLMBackend.reload — kill+recreate on checkpoint advance"
```

---

## Task 4: Wire `VLLMBackend` into `MiningEngine.__init__`

**Files:**
- Modify: `reliquary/miner/engine.py`
- Test: `tests/miner_priv/test_engine_phase2.py`

This task changes how `MiningEngine` holds its generation backend. Instead of taking a pre-loaded HF `vllm_model`, it now takes a `VLLMBackend` instance (or builds one from a `model_path`).

**Strategy:** keep the existing `__init__` signature backward-compatible by making `vllm_model` optional, and add a new `vllm_backend` parameter. If both are passed, prefer `vllm_backend`. This lets us migrate the CLI in Task 7 without breaking existing tests.

- [ ] **Step 1: Read the current `MiningEngine.__init__` signature**

```bash
cd ~/reliquary-miner-priv
grep -n "def __init__\|self.vllm_model\|self.hf_model" reliquary/miner/engine.py | head -10
```

Note the line numbers — you'll edit them in step 3.

- [ ] **Step 2: Write the failing test**

Create `tests/miner_priv/test_engine_phase2.py`:

```python
"""Phase 2 engine wiring tests."""
from unittest.mock import MagicMock
import pytest


def test_engine_accepts_vllm_backend_param():
    """MiningEngine.__init__ accepts a VLLMBackend via keyword arg."""
    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.vllm_backend import VLLMBackend

    backend = VLLMBackend(model_path="/fake", gpu_id=0)
    # Bypass full init by patching dependencies
    engine = MiningEngine.__new__(MiningEngine)
    engine._vllm_backend = backend
    assert engine._vllm_backend is backend


def test_engine_init_records_vllm_backend_attr():
    """When vllm_backend kwarg is passed, it's stored as self._vllm_backend."""
    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.vllm_backend import VLLMBackend

    fake_backend = VLLMBackend(model_path="/fake", gpu_id=0)
    fake_hf_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_wallet = MagicMock()
    fake_env = MagicMock()
    fake_env.__len__ = MagicMock(return_value=10)

    # Patch BucketIndex to avoid the real HF dataset load
    from unittest.mock import patch
    with patch("reliquary.miner.engine.BucketIndex") as mock_bucket:
        mock_bucket.return_value = MagicMock(__len__=MagicMock(return_value=10),
                                              bucket_of=MagicMock(return_value=("X", "Y")))
        engine = MiningEngine(
            vllm_model=None,
            hf_model=fake_hf_model,
            tokenizer=fake_tokenizer,
            wallet=fake_wallet,
            env=fake_env,
            vllm_backend=fake_backend,
        )
    assert engine._vllm_backend is fake_backend
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/miner_priv/test_engine_phase2.py -v
```

Expected: failures (no `vllm_backend` kwarg, no `_vllm_backend` attr).

- [ ] **Step 4: Edit `engine.py`**

Add the import near the top of `reliquary/miner/engine.py` (alongside the Phase 1 imports of `Selector` / `BucketIndex`):

```python
from reliquary.miner.vllm_backend import VLLMBackend
```

Modify `MiningEngine.__init__` signature: add `vllm_backend: VLLMBackend | None = None` as a keyword arg. After the existing attribute assignments, store it:

```python
self._vllm_backend = vllm_backend
```

Keep the existing `self.vllm_model = vllm_model` line for backward compatibility — Task 5 will route `_generate_m_rollouts` based on `self._vllm_backend` first.

Concretely, the diff to `__init__` is approximately:

```python
def __init__(
    self,
    vllm_model,
    hf_model,
    tokenizer,
    wallet,
    env,
    *,
    vllm_gpu: int = 0,
    proof_gpu: int = 1,
    validator_url_override: str | None = None,
    max_new_tokens: int = 1500,
    vllm_backend: VLLMBackend | None = None,   # NEW
) -> None:
    # ... existing assignments ...
    self.vllm_model = vllm_model
    self.hf_model = hf_model
    # ... rest unchanged ...
    self._vllm_backend = vllm_backend                  # NEW (last attribute set)
    # Phase 1 selector wiring (already present)
    self._selector = Selector(buckets=BucketIndex(), rng=_random.Random())
```

If the existing init signature differs in how kwargs are handled, ADAPT — the rule is: `vllm_backend` must be a keyword argument with default `None`, and `self._vllm_backend` must always be set (to `None` if not provided).

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_engine_phase2.py -v
pytest tests/miner_priv/ -v
```

Expected: 2 passed in `test_engine_phase2.py`. Full miner_priv suite: 53+ passed (46 prior + 7 vllm_backend + 2 new engine).

- [ ] **Step 6: Commit**

```bash
git add reliquary/miner/engine.py tests/miner_priv/test_engine_phase2.py
git commit -m "feat(miner-priv): MiningEngine accepts VLLMBackend via vllm_backend kwarg"
```

---

## Task 5: Route `_generate_m_rollouts` through `VLLMBackend`

**Files:**
- Modify: `reliquary/miner/engine.py`
- Modify: `tests/miner_priv/test_engine_phase2.py`

Currently `_generate_m_rollouts` calls `self.vllm_model.generate(...)` (HF). When `self._vllm_backend` is set, route through the backend instead. Output format must remain `list[dict]` with `{"tokens", "prompt_length"}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/miner_priv/test_engine_phase2.py`:

```python
def test_generate_m_rollouts_uses_vllm_backend():
    """When _vllm_backend is set, _generate_m_rollouts calls backend.generate."""
    from unittest.mock import patch, MagicMock
    from reliquary.miner.engine import MiningEngine

    engine = MiningEngine.__new__(MiningEngine)
    # Mock the tokenizer
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock(return_value=[100, 101, 102])
    fake_tokenizer.eos_token_id = 999
    engine.tokenizer = fake_tokenizer
    engine.max_new_tokens = 1500
    # Mock the backend
    fake_backend = MagicMock()
    fake_backend.generate = MagicMock(return_value=[
        [200, 201, 999, 700],   # first EOS at index 2 → keep up to and including 999
        [300, 999],
        [400, 401, 402],         # no EOS
        [500],
        [600, 601, 602, 603, 604],
        [700, 701],
        [800],
        [900, 901],
    ])
    engine._vllm_backend = fake_backend
    engine.vllm_model = None   # explicit: not used

    problem = {"prompt": "What is 2+2?"}
    result = engine._generate_m_rollouts(problem, randomness=b"x" * 32)

    # 8 rollouts back
    assert len(result) == 8
    # Each is a dict with tokens + prompt_length
    for r in result:
        assert "tokens" in r
        assert "prompt_length" in r
        assert r["prompt_length"] == 3   # len([100, 101, 102])
        # tokens = prompt + completion
        assert r["tokens"][:3] == [100, 101, 102]
    # First rollout: completion truncated at first EOS (index 2 → keeps [200,201,999])
    assert result[0]["tokens"] == [100, 101, 102, 200, 201, 999]
    # Third rollout has no EOS — completion kept whole
    assert result[2]["tokens"] == [100, 101, 102, 400, 401, 402]
    # Backend was called with the right token ids and n=8
    fake_backend.generate.assert_called_once()
    call_kwargs = fake_backend.generate.call_args.kwargs
    assert call_kwargs["prompt_token_ids"] == [100, 101, 102]
    assert call_kwargs["n"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_engine_phase2.py::test_generate_m_rollouts_uses_vllm_backend -v
```

Expected: failure (current `_generate_m_rollouts` uses HF, not backend).

- [ ] **Step 3: Edit `_generate_m_rollouts` in `engine.py`**

Find the method (around line 417 in your local copy). Replace its body with logic that branches on `self._vllm_backend`:

```python
def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
    """Generate M_ROLLOUTS completions at T_PROTO via vLLM backend (if set)
    or HF .generate() fallback.

    Returns list[dict] with {"tokens": prompt+completion, "prompt_length": int},
    same shape as the upstream HF path. Each completion is truncated at the
    first EOS so trailing batch-padding is not carried downstream — otherwise
    the validator's GRAIL forward pass would see extra EOS tokens the miner
    didn't "generate" in the usual sense.
    """
    from reliquary.constants import M_ROLLOUTS, T_PROTO, TOP_P_PROTO, TOP_K_PROTO

    prompt_tokens = self.tokenizer.encode(
        problem["prompt"], add_special_tokens=False
    )
    prompt_length = len(prompt_tokens)
    eos = self.tokenizer.eos_token_id

    # vLLM path
    if self._vllm_backend is not None:
        completions = self._vllm_backend.generate(
            prompt_token_ids=prompt_tokens,
            n=M_ROLLOUTS,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO if TOP_K_PROTO > 0 else -1,
            max_tokens=self.max_new_tokens,
        )
        rollouts = []
        for completion in completions:
            gen = list(completion)
            # Truncate at first post-prompt EOS (inclusive)
            try:
                first_eos = gen.index(eos)
                gen = gen[: first_eos + 1]
            except ValueError:
                pass
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    # HF path (Phase 1 fallback)
    import torch
    with torch.no_grad():
        input_tensor = torch.tensor(
            [prompt_tokens] * M_ROLLOUTS,
            device=getattr(self.vllm_model, "device", "cpu"),
        )
        outputs = self.vllm_model.generate(
            input_tensor,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            pad_token_id=self.tokenizer.pad_token_id,
        )
    rollouts = []
    for i in range(M_ROLLOUTS):
        seq = outputs[i].tolist()
        gen = seq[prompt_length:]
        try:
            first_eos = gen.index(eos)
            gen = gen[: first_eos + 1]
        except ValueError:
            pass
        rollouts.append({
            "tokens": prompt_tokens + gen,
            "prompt_length": prompt_length,
        })
    return rollouts
```

This preserves the HF path as a fallback if `_vllm_backend is None`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_engine_phase2.py -v
pytest tests/miner_priv/ -v
```

Expected: 3 passed in `test_engine_phase2.py`. Full suite still 54+ passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/engine.py tests/miner_priv/test_engine_phase2.py
git commit -m "feat(miner-priv): _generate_m_rollouts routes through VLLMBackend when set"
```

---

## Task 6: Drop HF reload on GPU 0; trigger `VLLMBackend.reload` instead

**Files:**
- Modify: `reliquary/miner/engine.py`
- Modify: `tests/miner_priv/test_engine_phase2.py`

When a checkpoint advances, `_load_checkpoint` currently rebuilds two HF models. With vLLM in place, we drop the HF GPU 0 rebuild and instead call `self._vllm_backend.reload(local_path)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/miner_priv/test_engine_phase2.py`:

```python
def test_load_checkpoint_calls_backend_reload_when_set(tmp_path):
    """When _vllm_backend is set, _load_checkpoint should call backend.reload
    and SKIP the HF GPU-0 rebuild path."""
    from unittest.mock import MagicMock, patch
    from reliquary.miner.engine import MiningEngine

    engine = MiningEngine.__new__(MiningEngine)
    engine._loaded_checkpoint_path = None
    engine.proof_gpu = 1
    engine.vllm_gpu = 0
    engine.hf_model = MagicMock()    # current proof model
    engine.vllm_model = None         # not used in vLLM path

    fake_backend = MagicMock()
    engine._vllm_backend = fake_backend

    # Stub torch.cuda.empty_cache to avoid hardware dep
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_hf:
        mock_hf.return_value = MagicMock(to=MagicMock(return_value=MagicMock(eval=MagicMock(return_value=MagicMock()))))
        result = engine._load_checkpoint(str(tmp_path))

    fake_backend.reload.assert_called_once_with(new_model_path=str(tmp_path))
    # Only ONE HF load (the proof model on GPU 1), NOT two
    assert mock_hf.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/miner_priv/test_engine_phase2.py::test_load_checkpoint_calls_backend_reload_when_set -v
```

Expected: failure (current `_load_checkpoint` always rebuilds 2 HF models).

- [ ] **Step 3: Edit `_load_checkpoint` in `engine.py`**

Find the method (around line 346 in your copy). Modify the second HF reload block (the one for `self.vllm_model` on `self.vllm_gpu`) to be conditional on `self._vllm_backend is None`. When the backend is set, replace that block with `self._vllm_backend.reload(new_model_path=local_path)`.

Concretely:

```python
def _load_checkpoint(self, local_path: str):
    """Reload both proof (HF on proof_gpu) and gen (vLLM via backend, or HF on vllm_gpu).

    historical ``vllm_model`` naming — when self._vllm_backend is set, we
    delegate gen-side reload to the backend (kill + recreate the vllm.LLM).
    Otherwise we fall back to reloading an HF model on self.vllm_gpu.
    """
    if self._loaded_checkpoint_path == local_path:
        logger.debug("_load_checkpoint: already loaded from %s", local_path)
        return self.hf_model

    logger.info("Loading checkpoint from %s", local_path)

    # 1. Reload hf_model (for GRAIL proofs) on the proof GPU. Always HF.
    try:
        new_hf = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(f"cuda:{self.proof_gpu}").eval()
    except Exception:
        logger.exception(
            "Failed to reload hf_model from %s; keeping old model",
            local_path,
        )
        return self.hf_model

    old_hf = self.hf_model
    self.hf_model = new_hf
    del old_hf
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 2. Reload generation side.
    if self._vllm_backend is not None:
        # vLLM path — backend manages its own LLM lifecycle.
        self._vllm_backend.reload(new_model_path=local_path)
        logger.info("vLLM backend reloaded for %s", local_path)
    else:
        # HF fallback path (Phase 1) — rebuild HF model on vllm_gpu.
        try:
            new_gen = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.vllm_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload vllm_model from %s; miner generation is "
                "BROKEN until the next successful pull.",
                local_path,
            )
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = new_gen
        del old_gen
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    self._loaded_checkpoint_path = local_path
    logger.info("Checkpoint %s loaded (proof=HF, gen=%s)",
                local_path,
                "vLLM" if self._vllm_backend is not None else "HF")
    return self.hf_model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/miner_priv/test_engine_phase2.py -v
pytest tests/miner_priv/ -v
```

Expected: 4 passed in `test_engine_phase2.py`. Full suite still passing.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/engine.py tests/miner_priv/test_engine_phase2.py
git commit -m "feat(miner-priv): _load_checkpoint delegates gen reload to VLLMBackend"
```

---

## Task 7: Update CLI to construct `VLLMBackend` and pass it to `MiningEngine`

**Files:**
- Modify: `reliquary/cli/main.py`

The CLI currently builds two HF models. With vLLM, we build ONE HF model (for proof on GPU 1) and ONE `VLLMBackend` (for gen on GPU 0).

- [ ] **Step 1: Read the current CLI miner setup**

```bash
cd ~/reliquary-miner-priv && sed -n '90,140p' reliquary/cli/main.py
```

Note the line where `vllm_model = AutoModelForCausalLM.from_pretrained(...)` is built — this gets replaced.

- [ ] **Step 2: Edit `reliquary/cli/main.py`**

Replace the GPU-0 HF model construction with a `VLLMBackend` instance, and pass it via the `vllm_backend` kwarg.

Find the block:
```python
vllm_model = AutoModelForCausalLM.from_pretrained(
    initial_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=ATTN_IMPLEMENTATION,
).to("cuda:0").eval()
```

Replace with:
```python
from reliquary.miner.vllm_backend import VLLMBackend
vllm_backend = VLLMBackend(
    model_path=initial_path,
    gpu_id=0,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    dtype="bfloat16",
)
vllm_model = None   # legacy attribute kept for fallback compatibility
```

Find the engine construction:
```python
engine = MiningEngine(
    vllm_model,
    hf_model,
    tokenizer,
    wallet,
    env,
    proof_gpu=0 if proof_device == "cuda:0" else 1,
    validator_url_override=validator_url or None,
)
```

Add `vllm_backend=vllm_backend` to the kwargs:
```python
engine = MiningEngine(
    vllm_model,
    hf_model,
    tokenizer,
    wallet,
    env,
    proof_gpu=0 if proof_device == "cuda:0" else 1,
    validator_url_override=validator_url or None,
    vllm_backend=vllm_backend,
)
```

- [ ] **Step 3: Smoke-test the import path** (no GPU run yet)

```bash
cd ~/reliquary-miner-priv && source .venv/bin/activate
python -c "from reliquary.cli.main import *; print('cli import OK')"
```

Expected: `cli import OK` (no NameError, no ImportError on `VLLMBackend`).

- [ ] **Step 4: Run the broader suite to confirm no regressions**

```bash
pytest tests/miner_priv/ -v
pytest tests/ -k "miner or engine" --ignore=tests/integration -v 2>&1 | tail -10
```

Expected: `tests/miner_priv/` all green. The broader sweep may have pre-existing failures (pytest-asyncio absent on `tests/unit/test_miner_checkpoint_pull.py`); those are unrelated.

- [ ] **Step 5: Commit**

```bash
git add reliquary/cli/main.py
git commit -m "feat(miner-priv): CLI constructs VLLMBackend on GPU 0 instead of HF model"
```

---

## Task 8: Hardware smoke test (manual, GPU box only)

**Files:** none — operational task. **DO NOT run this on a CPU dev box.**

This task runs vLLM end-to-end on the actual mining hardware. Tasks 1-7 are pure code + mocked tests and run on any machine; Task 8 is the first time vLLM is exercised. The implementer subagent MUST detect a CPU-only environment and STOP without running this task.

- [ ] **Step 1: Confirm hardware available**

```bash
cd ~/reliquary-miner-priv && source .venv/bin/activate
python -c "import torch; n = torch.cuda.device_count(); assert n >= 2, f'CPU-dev box (GPU count {n}) — Task 8 must run on the GPU mining box'; print('GPU count OK:', n)"
```

If this fails, **STOP** the task and report **DEFERRED_TO_GPU_BOX**. The user runs Task 8 themselves on the rented/owned 2× H100 box after pushing this branch there.

- [ ] **Step 2: Quick standalone vLLM smoke**

```bash
python -c "
from reliquary.miner.vllm_backend import VLLMBackend
backend = VLLMBackend(model_path='Qwen/Qwen3-4B-Instruct-2507', gpu_id=0)
out = backend.generate(prompt_token_ids=[1, 2, 3, 4, 5], n=2, max_tokens=32)
print('generated', len(out), 'completions, first =', out[0][:8], '...')
"
```

This should:
- Download Qwen3-4B if not cached (~8 GB)
- Build vLLM engine on GPU 0 (~30-60 s on first run)
- Sample 2 completions of up to 32 tokens
- Print something like `generated 2 completions, first = [123, 456, ...] ...`

If this fails, the failure mode is hardware/CUDA/vLLM compatibility — report the exact error verbatim.

- [ ] **Step 3: Spin up a local validator** (one terminal):

```bash
cd ~/reliquary-miner-priv && source .venv/bin/activate
reliquary validate --network local --netuid 1 --wallet-name test_validator --hotkey default 2>&1 | tee validator_smoke.log
```

If `--network local` is unsupported, refer to `docs/validating.md` and adapt.

- [ ] **Step 4: Spin up the private miner** (another terminal):

```bash
cd ~/reliquary-miner-priv && source .venv/bin/activate
reliquary mine --network local --netuid 1 \
    --wallet-name test_miner --hotkey default \
    --validator-url http://127.0.0.1:8888 \
    --log-level INFO 2>&1 | tee miner_smoke.log
```

- [ ] **Step 5: Run for ~10 minutes, then verify**

```bash
# Did vLLM build successfully?
grep -i "vllm\|building\|loading\|cuda" miner_smoke.log | head -20

# Time-per-window distribution
grep "submit_attempt" miner_smoke.log | jq -r '.time_gen_ms // .ts' | head -20

# Outcomes
grep "submit_attempt" miner_smoke.log | jq -r '.outcome' | sort | uniq -c

# σ distribution
grep -E "submit_attempt|local_reject" miner_smoke.log | jq -r '.sigma_local' | datamash min q1 median q3 max
```

Expected:
- vLLM engine started on GPU 0 (one or more `Building` / `Loading` messages around startup)
- Generation time markedly faster than Phase 1 baseline (sub-5s for 8× ~2k tokens vs ~10-12s with HF)
- σ distribution still clustered near 0.4-0.5 (selector unchanged)
- Outcomes mostly `accepted`

- [ ] **Step 6: Stop both processes (Ctrl-C). Optionally archive the log:**

```bash
gzip miner_smoke.log
git add miner_smoke.log.gz
git commit -m "chore: phase 2 smoke log on local validator"
```

Phase 2 is shippable when:
- vLLM builds without error
- Generation time is observably faster
- No GRAIL_FAIL or REWARD_MISMATCH from the validator (i.e., vLLM-generated tokens proof correctly under HF teacher-forcing)

---

## Self-review

**Spec coverage check:**
- §5.1 vLLM backend → Tasks 1, 2, 3 ✓
- §5.2 HF teacher-forcing on GPU 1 — kept unchanged in `_load_checkpoint` Task 6 ✓
- §6.2 vLLM hot-reload (kill+recreate fallback) → Task 3 + Task 6 ✓
- §8 Phase 2 deployment criteria → Task 8 (smoke), full canary follows in a Phase 3 plan or as Task 9 below if needed ✓
- CLI integration (engine construction) → Task 7 ✓

No gaps in scope.

**Placeholder scan:** searched for "TBD", "TODO", "implement later" in plan body — none. Tests have full code; impls have full code.

**Type/name consistency:**
- `VLLMBackend` constructor: `model_path`, `gpu_id`, `gpu_memory_utilization`, `max_model_len`, `dtype` — same field names across Tasks 1, 2, 3, 7 ✓
- `VLLMBackend.generate(prompt_token_ids, n, temperature, top_p, top_k, max_tokens)` — signature consistent across Tasks 2, 5 ✓
- `VLLMBackend.reload(new_model_path)` — name matches across Tasks 3, 6 ✓
- `MiningEngine.__init__` adds `vllm_backend` keyword arg, stored as `self._vllm_backend` — used in Tasks 4, 5, 6, 7 ✓
- `_build_llm` is the patchable seam in tests — used in Tasks 2, 3 ✓

Plan complete.
