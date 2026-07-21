# Reliquary

Decentralized GRPO training for large language models on Bittensor subnet 81.

Reliquary is a coordination protocol that turns a set of independent GPU operators into a single distributed RLHF pipeline. Miners generate cryptographically-proven rollouts; the validator aggregates them into a GRPO training batch, updates a live LLM checkpoint, and publishes the result to Hugging Face — all without trusting any single participant.

## The incentive shift, in one line

**Old subnets:** miners are paid per rollout. The competition is "do as many rollouts as you can."

**Reliquary:** miners are paid for the rollouts the trainer actually uses. The competition is "find the rollouts I need to train on" — i.e. predict which prompts sit at the policy's current learning frontier (group-σ in the trainable band, not yet in cooldown). A miner who picks well submits earlier, wins batch slots, and earns emission. A miner who picks poorly burns their own rollouts on `OUT_OF_ZONE` rejects.

This converts DAPO's reactive Dynamic Sampling filter into an ex-ante prediction market: the generate-then-discard cost is pushed out of the validator and onto the miner who guessed wrong. As the policy matures and the learning frontier narrows, selection intelligence becomes more valuable, not less. See [docs/concepts.md](docs/concepts.md#the-thesis) for the full argument.

## What it does

Each training window is one GRPO step. The cadence is event-driven: a window seals the instant eight valid, distinct-prompt rollout groups land. Miners race to submit; the first eight in (by validator-side TCP arrival) win the batch. The validator runs a PPO-clipped surrogate loss with a KL penalty against the frozen reference, then pushes the updated weights to a public HF repo. The whole cycle repeats immediately.

The network produces three artefacts: a continuously-trained model (published to HF every ten windows), a per-window rollout dataset (archived to R2), and a signed checkpoint manifest (served from `/checkpoint`) that lets anyone verify the chain of custody from a base model through every training step. The audit trail is cryptographic — each rollout carries a GRAIL sketch that lets the validator re-run the forward pass and confirm the generation came from the announced checkpoint.

Validators hold stake and run the training loop. Miners hold hotkeys, run GPU inference, and earn emission proportional to their share of batch slots over a rolling 72-window scoring interval; the main optimization surface for a miner is predicting which prompts sit at the policy's learning frontier — selection intelligence wins slots, and by construction also feeds the GRPO step gradient-rich groups. Downstream consumers — researchers, fine-tuning pipelines — pull the published HF checkpoint or the R2 rollout dataset directly.

## Quick deploy (Docker)

Recommended way to run a miner: pull the prebuilt image
`ghcr.io/reliquadotai/reliquary-miner:latest`. It pins torch 2.7.0+cu128,
flash-attn 2.8.3, vLLM 0.10.2, transformers 4.x, bittensor 10.2 and the
rest of the deps gauntlet, so you skip the version-pinning dance.

**Prerequisites on the host**

- NVIDIA GPU with a driver supporting CUDA 12.8 (H100/H200 confirmed)
- Docker 24+ and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  (`docker run --gpus all` must work)
- A Bittensor wallet with a registered hotkey on netuid 81

**One-time: log in to GHCR**

The repo and the image package are private. You need:

1. To be added as a collaborator on `reliquadotai/reliquary-miner` (ask
   an org admin); without that, `docker pull` returns `denied`.
2. A GitHub
   [Personal Access Token (classic)](https://github.com/settings/tokens)
   with the `read:packages` scope.

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u <your-github-user> --password-stdin
```

**Pull and run**

```bash
docker pull ghcr.io/reliquadotai/reliquary-miner:latest

docker run -d \
  --name reliquary-miner \
  --restart unless-stopped \
  --gpus all \
  -v /root/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -e BT_WALLET_NAME=<your-wallet-dir>       \
  -e BT_HOTKEY=<your-hotkey-name>           \
  -e RELIQUARY_VALIDATOR_URL=http://<validator-ip>:8080 \
  ghcr.io/reliquadotai/reliquary-miner:latest

docker logs -f reliquary-miner
```

**Environment variables**

| Variable | Required | Default | Notes |
|---|---|---|---|
| `BT_WALLET_NAME` | yes | — | Wallet dir name under `~/.bittensor/wallets` |
| `BT_HOTKEY` | yes | — | Hotkey file under `wallets/<name>/hotkeys/` |
| `BT_NETWORK` | no | `finney` | Bittensor network |
| `BT_NETUID` | no | `81` | Subnet id |
| `RELIQUARY_VALIDATOR_URL` | no | (metagraph discovery) | Override validator HTTP endpoint |
| `RELIQUARY_CHECKPOINT` | no | `Qwen/Qwen3-4B-Instruct-2507` | Fallback when validator has not published yet |
| `RELIQUARY_USE_DRAND` | no | `1` | Set to `0` to disable drand verification (test only) |
| `RELIQUARY_LOG_LEVEL` | no | `INFO` | Standard Python log levels |

The wallet mount needs the hotkey file and `coldkeypub.txt`. **Do not mount
the coldkey itself** — the miner does not need it. Read-only (`:ro`) is
enforced for safety.

## Quickstart

- To mine: see [docs/mining.md](docs/mining.md)
- To validate: see [docs/validating.md](docs/validating.md)
- To understand the mechanism: see [docs/concepts.md](docs/concepts.md)

## Architecture at a glance

```
┌─────────────┐    HTTP    ┌─────────────┐   HF push   ┌──────────────┐
│   Miners    │ ─────────▶ │  Validator  │ ──────────▶ │   HF Hub     │
│  (N nodes)  │ ◀───────── │  (1 node)   │             │ (model repo) │
└─────────────┘ /submit    └──────┬──────┘             └──────┬───────┘
     ▲         /state             │                            │
     │         /checkpoint        │ weights                    │ pull
     │                            ▼                            │
     │                   ┌──────────────┐                      │
     │                   │  Bittensor   │                      │
     │                   │  chain       │                      │
     │                   │  (set_weights│                      │
     │                   │   every 360  │                      │
     │                   │   blocks)    │                      │
     │                   └──────────────┘                      │
     │                                                         │
     │                   ┌──────────────┐                      │
     └───────────────────│     R2       │◀────── archive ──────┘
                         │ (rollouts +  │         (per window)
                         │  dataset)    │
                         └──────────────┘
```

Miners submit rollout groups to `/submit` and poll `/state` for checkpoint updates. The validator trains, publishes to HF, and broadcasts weights on-chain every `WEIGHT_SUBMISSION_INTERVAL = 360` blocks. Miners pull new weights via `/state` → HF `snapshot_download`. R2 stores the per-window rollout archive; the validator reads it at startup to rebuild the prompt cooldown map.

## Status

- **v1** — verifiable-inference dataset production (shipped, deprecated)
- **v2** — GRPO market with in-subnet training (shipped)
- **v2.1** — batch-driven windows, HF checkpoint distribution, EMA scoring (shipped)
- **v2.2** — manipulation-resistant batch ordering (current)
- **v2.3** — multi-validator consensus (planned)

## License

MIT — see `LICENSE`.
