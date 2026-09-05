"""GRAIL Protocol Constants.

Immutable values that all network participants must agree on.
No os.getenv() overrides. Changes require coordinated deployment.
"""

# ────────────────  GRAIL PROOF VERSION  ────────────────

GRAIL_PROOF_VERSION = "v7"

# ────────────────  CRYPTOGRAPHIC CONSTANTS  ────────────────

# Mersenne prime for modular sketch arithmetic.
PRIME_Q = 2_147_483_647

# Number of random challenge positions per completion.
CHALLENGE_K = 32

# PRF domain labels for different randomness derivations.
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# Transformer layer index for hidden state extraction (-1 = last layer).
LAYER_INDEX = -1

# Batch size for proof computation (log-softmax / GRAIL commitments).
# Fixed: changing causes numerical divergence between miner and validator.
PROOF_BATCH_SIZE = 16

# Top-K activation selection for sketch computation.
PROOF_TOPK = 16

# Logarithmic bucketing: buckets per sign (16 total = 8 positive + 8 negative).
PROOF_NUM_BUCKETS = 8

# Bounded coefficient range for sketch robustness: r in [-127, 127].
PROOF_COEFF_RANGE = 127

# Sketch tolerance at position 0. Calibrated against a 10×30 cheater-curve
# sweep (scripts/cheater_curve_threshold.py) where a frozen base miner
# faces a freshly-trained validator: 1000 catches a 1-step-stale cheater
# with 75 % probability and 95 %+ from step 10 onwards, with 0 % false
# positives same-GPU. Subnet currently in test phase — miners are advised
# to use the same card as the validator (H200) until cross-GPU honest
# noise is measured. See docs/mining.md.
PROOF_SKETCH_TOLERANCE_BASE = 5000

# Sketch tolerance sqrt growth factor per position.
# tolerance(P) = base + growth * sqrt(P).
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0

# Attention implementation forced across all model loading paths.
# Override with GRAIL_ATTN_IMPL for test envs without flash-attn compiled
# (e.g. "eager" or "sdpa"). Production runs must stay on flash_attention_2
# because sketch commitments are bit-sensitive to attention kernel variance.
import os as _os
ATTN_IMPLEMENTATION = _os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2")

# Per-window prompt range (#91). Size of the eligible [lo, hi) prompt slice
# both validator and miner derive from the window randomness. MUST match the
# validator's value (5000) or the derived slice diverges → PROMPT_OUT_OF_RANGE.
PROMPT_RANGE_SIZE = int(_os.environ.get("PROMPT_RANGE_SIZE", "5000"))

# ────────────────  TIMING (CONSENSUS)  ────────────────

# Blocks per window — 5 blocks × 12s ≈ 60s.
# All roles use this to determine window boundaries. With a typical tempo of
# 360 blocks, the EMA covers 72 windows of scoring history per on-chain
# weight submission, providing ~72× smoothing of miner scores over the epoch.
WINDOW_LENGTH = 5

# Bittensor block time target average (seconds).
BLOCK_TIME_SECONDS = 12

# Typical variance in block production time (seconds).
BLOCK_TIME_VARIANCE = 3

# Network latency allowance for file uploads (seconds).
NETWORK_UPLOAD_LATENCY = 30

# Grace period = block variance + upload latency.
UPLOAD_GRACE_PERIOD = BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# Buffer for future drand beacon (seconds).
DRAND_FUTURE_BUFFER = 30

# Buffer subtracted from per-window deadline to leave room for final submissions.
UPLOAD_BUFFER = NETWORK_UPLOAD_LATENCY

# ────────────────  ROLLOUT GENERATION  ────────────────


def protocol_version(env=None) -> int:
    """Wire protocol version. Live = 3 (4B / auction-v3 profile). Env-overridable
    (``RELIQUARY_PROTOCOL_VERSION``) so a rollback to v2 — or the v4 cutover —
    is a launch flag, not a code change. Malformed / non-positive → the live
    default (3)."""
    src = _os.environ if env is None else env
    raw = src.get("RELIQUARY_PROTOCOL_VERSION")
    if raw is None:
        return 3
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 3
    return value if value > 0 else 3


def forced_seed_domain(version: int) -> str:
    """u_at domain string — seeds every forced token; must match the validator's
    ``reliquary-forced-seed-v{PROTOCOL_VERSION}``."""
    return f"reliquary-forced-seed-v{version}"


PROTOCOL_VERSION = protocol_version()

# v4+ : protocole raw-completion — jamais de chat template (un repo "-Base"
# DÉCLARE le template famille sans l'avoir appris ; l'auto-détection ouvrirait
# un <think> sur un modèle choisi précisément parce qu'il n'en ouvre pas). Un
# seul switch pour encode_prompt ET le grader (parité upstream 8c38992).
RAW_COMPLETION_PROMPTS = PROTOCOL_VERSION >= 4

# Format de réponse math (upstream a6456b4, profil environments.answer_format) :
# v4 = "boxed" — seul le span \boxed{} est authentifiable par la preuve
# d'intégrité de réponse ; tout non-boxé vaut 0, SANS fallback (le canal
# "Answer:" du 8c38992 a été supprimé le 17/08, il contournait le tamper
# guard). v2/v3 = "boxed_or_trailing_number" (comportement payé historique).
MATH_ANSWER_FORMAT = (
    "boxed" if PROTOCOL_VERSION >= 4 else "boxed_or_trailing_number"
)

# v4+ : manifest OMI restreint aux shards canoniques train-* (les shards
# train_1M/2M/5M sont des sous-ensembles curés de train → 8M lignes
# dupliquées). len(env) = consensus prompt-range — cutover-only.
OMI_TRAIN_SHARDS_ONLY = PROTOCOL_VERSION >= 4

# Network-wide protocol cap on completion length. v3 (4B) uniform ceiling = 16384
# for BOTH envs (was 32768 on 2B/v2). v4 (4B-Base/DAPO) starts the length
# curriculum at 8192. Env-overridable for rollback.
MAX_NEW_TOKENS_PROTOCOL_CAP = int(
    _os.environ.get(
        "RELIQUARY_MAX_NEW_TOKENS_PROTOCOL_CAP",
        "8192" if PROTOCOL_VERSION >= 4 else "16384",
    )
)

# Budget-Forced Termination (cot-2b / v7). Thinking model reasons up to
# BFT_THINKING_BUDGET; if </think> isn't closed by then, the miner appends
# BFT_FORCE_TEMPLATE and samples the boxed answer in BFT_ANSWER_BUDGET more
# tokens. BFT applies to openmathinstruct only. Values verbatim from validator.
# v4 : AUCUN BFT (les 2 envs `bft=None` dans le profil upstream) — le modèle
# base n'émet pas de <think>, et toute claim `forced` est rejetée fail-closed
# par le validateur v4 (validate_force_span).
BFT_ENABLED = PROTOCOL_VERSION < 4


def bft_thinking_budget(env=None) -> int:
    """Phase-1 math thinking budget — a WIRE CONSTANT (miner and validator must
    agree; it pins the forced-seed span at ``prompt_len + budget``).

    Live 4B/v3 = **15616** (was 2048 on the dead 2B/v2). BFT is MATH-only — our
    code env has no BFT (``bft_applicable`` is False for opencodeinstruct), so this
    is irrelevant to code mining but kept live-correct for math. Env-overridable
    (``RELIQUARY_BFT_THINKING_BUDGET``) for rollback; a missing / malformed /
    non-positive value falls back to the live default (a zero budget would break
    every forced rollout).
    """
    src = _os.environ if env is None else env
    raw = src.get("RELIQUARY_BFT_THINKING_BUDGET")
    if raw is None:
        return 15616
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 15616
    return value if value > 0 else 15616


BFT_THINKING_BUDGET = bft_thinking_budget()
BFT_ANSWER_BUDGET = 512
BFT_FORCE_TEMPLATE = "</think>\n\nFinal Answer: \\boxed{"
# v3 forced answer must fit thinking + force-template + answer under the cap.
# v4 : BFT off → la contrainte n'a pas d'objet (cap 8192 < 15616+512 est
# normal, les budgets BFT sont morts).
if BFT_ENABLED:
    assert MAX_NEW_TOKENS_PROTOCOL_CAP >= BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET, (
        f"cap {MAX_NEW_TOKENS_PROTOCOL_CAP} < BFT {BFT_THINKING_BUDGET}+{BFT_ANSWER_BUDGET}"
    )

# Overlong / under-thinking reward shaping (validator training-side only; not a
# miner reward gate). SHAPE_PENALTY = 0 disables. Kept for parity/reference.
SHAPE_PENALTY = 0.5
SHAPE_LEN_FRAC = 0.5

# Per-token authenticity floor (validator gate; tightened 1e-10 → 1e-8 in v7).
# Honest sampling passes by construction; kept for parity.
TOKEN_AUTH_THRESHOLD = 1e-8

# Soft cap on per-hotkey entries persisted to ``archive["rejected"]`` per
# window. Beyond this, ``reject_counts`` still increments but no metadata is
# appended — protects the R2 payload size against a flood of garbage
# submissions from a single attacker.
REJECTED_LIST_CAP_PER_HOTKEY = 5

# ────────────────  GRPO BATCHING  ────────────────

# Default HTTP port the validator listens on for miner submissions.
VALIDATOR_HTTP_PORT = 8888

# Active environment name (resolved by reliquary.environment.load_environment).
ENVIRONMENT_NAME = "openmathinstruct"

# UID that receives unused slot emission budget (the burn address).
UID_BURN = 0

# ────────────────  VALIDATION RULES  ────────────────

# File size bounds for valid rollout window files.
MIN_ROLLOUT_FILE_SIZE_BYTES = 200
MAX_ROLLOUT_FILE_SIZE_BYTES = 350 * 1024 * 1024  # 350 MB

# ────────────────  CONTINUOUS VALIDATION  ────────────────

# How often the validator polls for new state (seconds).
POLL_INTERVAL_SECONDS = 10

# ────────────────  WEIGHT SUBMISSION  ────────────────

# Submit weights when blocks_until_next_epoch <= this value. Tuned so all
# validators of a netuid land in the same ~20-block window (≈4 min on
# 12s/block) and read near-identical R2 archive snapshots, then converge
# to identical weights via the deterministic EMA replay.
EPOCH_SUBMIT_LEAD_BLOCKS = 20

# ────────────────  STORAGE  ────────────────

CHECKPOINT_PREFIX = "reliquary/checkpoints/"

# ────────────────  HUGGING FACE CHECKPOINT PUBLISHING  ────────────────

# How often to publish the current in-memory model to Hugging Face.
# Training happens every window (stub in v2.1, real GRPO in follow-up),
# but HF uploads are slow for large models, so we publish only every
# N windows. Between publishes, miners stay on the last pushed revision.
CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10

# Default HF repo target for published checkpoints. Operator may
# override via --hf-repo-id CLI arg. Must be a writable repo id for
# the validator's HF token.
DEFAULT_HF_REPO_ID = "aivolutionedge/reliquary-sn"

# ────────────────  DEPRECATED (GRPO REFACTOR)  ────────────────
# Kept importable to avoid breaking transitive imports during the rollout.
# These knobs no longer participate in any runtime decision and will be
# removed in a follow-up cleanup once no consumer references them.

MINER_SAMPLING_ENABLED = True
MINER_SAMPLE_RATE = 0.25
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35

ROLLOUT_SAMPLE_RATE = 0.10
ROLLOUT_SAMPLE_MIN = 16

VERIFICATION_BATCH_SIZE = 16
BATCH_FAILURE_THRESHOLD = 0.30

FAILURE_LOOKBACK_WINDOWS = 14
USED_INDICES_MAX_AGE_WINDOWS = 100

MAX_ROLLOUTS_PER_FILE = 6000

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DATASET_SPLIT = "train"

# ────────────────  GRPO MARKET (v2)  ────────────────

# Minimum reward-std for a group to pass the zone filter.
# For binary Bernoulli rewards this is equivalent to the old
# k ∈ [2, 6] gate (σ of Bernoulli(p=2/8) ≈ 0.433). For continuous
# rewards it filters groups whose rollouts clustered too tight to
# carry meaningful GRPO signal.
# v4 : critère dynamic-sampling DAPO — 0.24 admet tout k∈[1,15] à M=16
# (σ(k=1)=0.2421), 0.22 bootstrap ; upstream 8c38992. Les valeurs v3 (0.33,
# désalignement cosmétique historique — le chemin mineur effectif est
# zone.py à 0.43) restent intouchées : byte-identité v3 d'abord.
SIGMA_MIN = 0.24 if PROTOCOL_VERSION >= 4 else 0.33
BOOTSTRAP_SIGMA_MIN = 0.22 if PROTOCOL_VERSION >= 4 else 0.33

# Number of rollouts per submission (= size of each GRPO group).
# v4 : G=16 (DAPO §4.1).
M_ROLLOUTS = 16 if PROTOCOL_VERSION >= 4 else 8

# Training batch size — the first B valid in-zone submissions (FIFO by
# TCP arrival, distinct prompts, not in cooldown) feed the GRPO step.
B_BATCH = 16 if PROTOCOL_VERSION >= 4 else 8

# Sampling temperature fixed at protocol level. Miners who use a different
# T would produce samples from a different distribution → biased GRPO
# gradient. Value chosen in the GRPO-friendly range (non-zero).
# v4 : sampling DAPO/verl = T 1.0, support COMPLET (top_p 1.0, top_k 0) →
# warp() devient la softmax identité (les gardes `top_k and top_k > 0` /
# `top_p and top_p < 1.0` court-circuitent, vérifié sur nos 4 chemins FS).
T_PROTO = 1.0 if PROTOCOL_VERSION >= 4 else 0.6

# Top-p and top-k for sampling (fixed alongside T_PROTO).
TOP_P_PROTO = 1.0 if PROTOCOL_VERSION >= 4 else 0.95
TOP_K_PROTO = 0 if PROTOCOL_VERSION >= 4 else 20

# A prompt that entered the training batch is ineligible for B_BATCH for
# the next N windows (= training steps). Forces curriculum rotation so
# the policy has time to shift between reuses.
# v2.3 + OpenMathInstruct (14M prompts): bumped from 200 to 1_000_000 so
# each prompt is effectively single-use across the lifetime of any
# realistic training run (1M windows ≈ 700 days at 5 blocks × 12s). The
# 14M-prompt env supplies enough fresh material without needing reuse.
BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000

# Validator startup: cap the number of R2 archives scanned to rebuild
# CooldownMap. Independent of BATCH_PROMPT_COOLDOWN_WINDOWS — that
# constant can be astronomically large for one-shot semantics, but R2
# rebuild must stay O(1) in elapsed wall time. 10_000 archives ≈ 8.3
# days of windows, which dominates any realistic restart gap. Older
# entries are still in cooldown (the in-memory map is replayed from R2
# and any miss is treated as ``no cooldown record``, which is safe: the
# validator's hash-blacklist still rejects re-submission of the same
# token sequence).
COOLDOWN_REBUILD_LOOKBACK = 10_000

# Per-rollout content dedup horizon. Independent of and strictly longer
# than the prompt cooldown: cooldown lets a prompt come back for fresh
# content, the hash set blacklists the specific (tokens) of every rollout
# already trained on. 10000 windows ≈ 3.5 days at 5 blocks/window. After
# that, natural model drift between training steps is large enough that
# stale generations fall through the distribution / logprob filters.
HASH_DEDUP_RETENTION_WINDOWS = 10000

# Max submissions any single hotkey can send per window. Counter resets at
# every new window (on batcher swap). Excess submissions are HTTP-rejected
# as RATE_LIMITED before touching the validation pipeline. v3 : 8 = B_BATCH
# — one slot per prompt a hotkey can credibly win in a window. v4 : 2·B_BATCH
# (marge retry/amélioration, upstream 8c38992).
MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = (
    2 * B_BATCH if PROTOCOL_VERSION >= 4 else 8
)

# Budget de troncature par soumission — parité upstream (constants.py
# main==v4, valeurs NON gatées là-bas) : un rollout « truncated » = cap
# atteint sans EOS, toléré jusqu'à 1 par groupe (3 en code, où rien ne force
# la terminaison). Consommé par la garde v4 du mineur (engine) ; la garde v3
# locale reste volontairement tout-ou-rien (fix 2026-08-05).
MAX_TRUNCATED_PER_SUBMISSION = 1
BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION = 1
MAX_TRUNCATED_PER_SUBMISSION_BY_ENV: dict[str, int] = {
    "opencodeinstruct": 3,
}


def max_truncated_for_environment(
    environment: str,
    *,
    bootstrap: bool = False,
) -> int:
    if bootstrap:
        return BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
    return MAX_TRUNCATED_PER_SUBMISSION_BY_ENV.get(
        environment,
        MAX_TRUNCATED_PER_SUBMISSION,
    )


# Per-case wall-clock budget for the opencode local grader subprocess.
#
# Le validateur utilise 5 s (parité de sigma). MAIS l'environnement code est
# ``validator_authoritative_reward=True`` : le validateur ÉCRASE notre reward
# (`batcher.py`, `rollout.reward = computed_reward`) — notre note ne sert donc
# QU'À NOTRE SÉLECTION, jamais à la conformité. Baisser ce budget ne peut pas
# provoquer de REWARD_MISMATCH ; le seul risque est de mal estimer la zone.
#
# Mesuré le 26/08 sur 3 718 groupes soumis : la durée de grading d'un groupe est
# BIMODALE — 76,3 % sous 0,2 s, 22,0 % à exactement 5,0 s (un rollout qui boucle),
# et seulement **1,26 %** entre 0,5 et 4,5 s. Le plafond coûte donc en moyenne
# 1,18 s d'ARRIVÉE par entrée (0,85 s sur la 1re entrée de la fenêtre, dont
# 15,2 % mangent les 5 s pleines et passent de 9,1 s à 16,7 s d'arrivée médiane,
# c'est-à-dire de la bande payée 54 % à la bande payée 6 %).
# Abaisser à 1,0 s récupère 0,90 s d'arrivée moyenne pour au plus 0,8 % de
# groupes mal notés.
# Défaut INCHANGÉ (5) : le réglage se fait par variable d'environnement.
# ⚠️ Surveiller le taux de verdicts ``out_of_zone`` (référence 26/08 : 5,5 %) —
# c'est l'indicateur d'un désaccord de notation avec le validateur.
GRADER_EVAL_TIMEOUT_SECONDS = float(
    _os.environ.get("RELIQUARY_GRADE_TIMEOUT_S", "5")
)

# Max GRAIL-validated submissions retained per prompt per window. Once this
# cap is reached for a prompt, further submissions for that prompt are
# rejected as PROMPT_FULL before the heavy verify. Bounds the validator's
# GPU cost when many miners attack the same prompt — combined with the
# per-hotkey cap above, worst-case GRAIL load per window is
# MAX_SUBMISSIONS_PER_PROMPT × min(|env|, MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW × n_hotkeys).
MAX_SUBMISSIONS_PER_PROMPT = 10

# Bootstrap phase: first BOOTSTRAP_WINDOWS of a new subnet/checkpoint use
# relaxed thresholds to keep the batch filling while miner pop + env
# coverage are thin.
BOOTSTRAP_WINDOWS = 100

# First on-chain block at which this subnet deployed v2. Used to
# determine bootstrap eligibility. Set at the coordinated cutover.
SUBNET_START_BLOCK = 0

# ────────────────  v2.1 BATCH-DRIVEN WINDOWS  ────────────────

# Safety-net timeout: a window auto-seals after this many seconds even
# if fewer than B valid submissions have landed. The unused slots burn.
# Set generously — this is a backstop, not the cadence.
WINDOW_TIMEOUT_SECONDS = 7200

# Local JSON path for validator state (window_n counter + checkpoint_n).
# Resolved relative to the CWD if not absolute.
CHECKPOINT_STATE_PATH_DEFAULT = "reliquary/state/checkpoint.json"

# Local directory for staged checkpoint files before R2 upload.
CHECKPOINT_STAGING_DIR_DEFAULT = "reliquary/state/checkpoints"

# ────────────────  SCORING  ────────────────

# EMA smoothing factor for miner score. 2/(N+1) with N=72 (the EMA history depth).
# gives a ~25-window half-life — a miner that stops contributing loses
# half their score in ~25 windows.
EMA_ALPHA = 2.0 / (72 + 1)  # ≈ 0.0274

# ────────────────  GRPO TRAINING (v2.1)  ────────────────

# Learning rate for AdamW. RL fine-tuning on pretrained LLMs is sensitive;
# too high = collapse. Empirical drift measurement (scripts/measure_sketch_drift.py)
# showed 5e-7 produced a sketch delta of ~600 (≈10 % of the 6000 sketch
# tolerance) over 50 training steps — effectively indistinguishable from the
# base model, which also means stale-model cheaters pass GRAIL. Matched
# DAPO / R1-Zero-scale literature (1e-6 to 5e-6) by bumping to 5e-6.
LEARNING_RATE = 5e-6

# PPO clip range. Standard in GRPO/RLHF literature.
PPO_CLIP_EPSILON = 0.2

# KL penalty weight (DeepSeek's GRPO default). Keeps π_new close to the
# frozen reference; too low → drift / mode collapse; too high → no learning.
KL_BETA = 0.04

# Max gradient norm before step — standard RL stability guard.
GRAD_CLIP_NORM = 1.0

# Linear LR warmup for the first N training steps (= N windows sealed).
LR_WARMUP_WINDOWS = 10

# Cosine schedule end target (in windows). Chosen large so LR never
# actually reaches zero at normal cadence — effectively a slow decay.
LR_COSINE_MAX_WINDOWS = 10_000

# Default base model (HF repo id). Served as the reference for KL and the
# cold-start checkpoint.
# v4 (audit item 7) : modèle de départ = Qwen3-4B-Base, révision ÉPINGLÉE
# (celle du profil upstream 8c38992) — sans pin, le fallback --checkpoint
# chargerait le HEAD HF et un commit poussé par Qwen divergerait le
# checkpoint_hash du validateur. v3 : valeurs historiques intouchées.
DEFAULT_BASE_MODEL = (
    "Qwen/Qwen3-4B-Base" if PROTOCOL_VERSION >= 4 else "Qwen/Qwen3.5-2B"
)
DEFAULT_BASE_MODEL_REVISION = (
    "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    if PROTOCOL_VERSION >= 4 else None
)

# ────────────────  WANDB TELEMETRY (opt-in, validator-only)  ────────────────

# Wandb project name used by validator-side telemetry. Operators can
# override with the WANDB_PROJECT env var.
WANDB_PROJECT = "reliquary-validator"

# Bumping this constant (or setting RELIQUARY_WANDB_VERSION) starts a
# fresh wandb run. Same value across restarts → wandb resumes the
# existing run (resume="allow").
WANDB_TRAINING_VERSION = "v1"

# ────────────────  BEHAVIOURAL VALIDATORS  ────────────────
# Thresholds calibrated in the original grail repo against ~430k honest
# cross-GPU / cross-attn / cross-batch trials with 0 % false-positive
# rate. Do not re-tune without the same empirical setup.

# Minimum probability the model must have assigned to EOS at the position
# that produced it. Below this threshold, the rollout is presumed to be
# artificially truncated (a miner truncating mid-reasoning to lock in a
# favourable partial output). Upstream grail uses 0.02; we lowered to 0.01
# after Qwen3-4B + T_PROTO=0.9 prod logs showed honest EOS clustering just
# below 0.02. Mid-reasoning forgery still fails (p_stop typically < 0.001).
# v4 : support complet T=1.0 → l'EOS honnête traîne plus bas ; valeur
# calibrée upstream (H100 2026-08-12, 40 rollouts honnêtes Qwen3-4B-Base).
MIN_EOS_PROBABILITY = 0.001 if PROTOCOL_VERSION >= 4 else 0.01

# LogprobValidator: max allowed median importance-sampling deviation
# across K=CHALLENGE_K positions. dev_i = exp(|model_lp - miner_lp|) - 1.
# Reverted to GRAIL upstream's 0.10 (calibrated at 0% FP, ~50% headroom
# over the worst honest case). The previous 0.01 was tightened against
# same-stack miners observed at ~0.00013 median dev, but cross-stack
# honest drift (transformers 4.x miner ↔ 5.x validator) sits around
# 0.03-0.04 and was getting falsely rejected. 0.10 still flags clearly
# stale or forged checkpoints (cheater drift grows quickly past 0.10),
# while making the network functional for the majority of honest setups.
LOGPROB_IS_EPS = 0.10

# DistributionValidator: chosen-token probability thresholds. A "chosen
# token" is the token the miner sampled at step t; its probability under
# the validator's model (at the protocol temperature) is
# p_t = softmax(logits_{t-1} / T)[token_t].
SAMPLING_MIN_STEPS = 30         # completion must be at least this long
SAMPLING_LOW_P = 0.10           # prob <= this → "low" chosen token
SAMPLING_HIGH_P = 0.90           # prob >= this → "high" chosen token
# v4 : le sampling full-support visite légitimement la queue de distribution —
# planchers calibrés upstream (minima honnêtes : médiane 0.121, q10 0.00064 ;
# masse d'un token forgé ~1/vocab ≈ 7e-6).
SAMPLING_MEDIAN_LOW_MAX = (
    0.05 if PROTOCOL_VERSION >= 4 else 0.30
)                               # median chosen prob must be above
SAMPLING_LOW_Q10_MAX = (
    0.0002 if PROTOCOL_VERSION >= 4 else 0.025
)                               # 10th-percentile must be above

# ────────────────  FORCED-SEED SAMPLING (validator b790e42) ────────────────
# (protocol_version() / forced_seed_domain() sont définis en tête de module —
# les valeurs v4 en dépendent.)

FORCED_SEED_STOCHASTIC_MAXPROB = 0.99
FORCED_SEED_CONSISTENCY_FLOOR = 0.80
FORCED_SEED_MIN_STOCH_POSITIONS = 30
# v4 : le match single-rollout honnête descend à 0.78 en full-support (16k) →
# plancher 0.70 (upstream 8c38992) ; le plancher GROUPE 0.80 ne bouge pas.
FORCED_SEED_ROLLOUT_FLOOR = 0.70 if PROTOCOL_VERSION >= 4 else 0.75
FORCED_SEED_ROLLOUT_MIN_STOCH = 20
FORCED_SEED_ENFORCE = _os.environ.get(
    "FORCED_SEED_ENFORCE", "true"
).strip().lower() in ("1", "true", "yes", "on")
FORCED_SEED_PROTOCOL_VERSION = PROTOCOL_VERSION
FORCED_SEED_DOMAIN = forced_seed_domain(FORCED_SEED_PROTOCOL_VERSION)


def generation_profile_id(env=None) -> str:
    """Generation-contract profile advertised on the wire and bound into the v3
    envelope signature. Live = ``qwen35-4b-auction-v3`` — the validator rejects
    any other value GENERATION_CONTRACT_MISMATCH (submission ET precommit,
    avant quota/grading). v4 = ``qwen3-4b-base-dapo-v4`` (upstream 8c38992
    profiles.py). Dérivé de ``protocol_version(env)`` — pas du snapshot module
    — pour qu'un env dict explicite suive SON protocole. Env-overridable pour
    un bump de profil sans changement de code."""
    src = _os.environ if env is None else env
    default = (
        # v5 (23/08, PR #190) : profil « reasoning », le prompt passe a un
        # template versionne. L'id est ANNONCE au validateur — s'il ne
        # correspond pas au sien, c'est 100 % de GENERATION_CONTRACT_MISMATCH.
        "qwen3-4b-base-dapo-reasoning-v5"
        if protocol_version(env) >= 5
        else "qwen3-4b-base-dapo-v4"
        if protocol_version(env) >= 4
        else "qwen35-4b-auction-v3"
    )
    return src.get("RELIQUARY_GENERATION_PROFILE_ID", default)


GENERATION_PROFILE_ID = generation_profile_id()
