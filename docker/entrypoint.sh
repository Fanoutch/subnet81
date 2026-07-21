#!/bin/bash
# Entrypoint for the Reliquary miner image.
#
# Reads environment variables to build the `reliquary mine` argv.
# Required: BT_WALLET_NAME, BT_HOTKEY.
set -euo pipefail

: "${BT_WALLET_NAME:?BT_WALLET_NAME is required (the wallet dir name under ~/.bittensor/wallets)}"
: "${BT_HOTKEY:?BT_HOTKEY is required (the hotkey file name under wallets/<name>/hotkeys/)}"

args=(
  --network      "${BT_NETWORK:-finney}"
  --netuid       "${BT_NETUID:-81}"
  --wallet-name  "${BT_WALLET_NAME}"
  --hotkey       "${BT_HOTKEY}"
  --checkpoint   "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3-4B-Instruct-2507}"
  --log-level    "${RELIQUARY_LOG_LEVEL:-INFO}"
)

if [[ "${RELIQUARY_USE_DRAND:-1}" == "1" ]]; then
  args+=(--use-drand)
else
  args+=(--no-use-drand)
fi

if [[ -n "${RELIQUARY_VALIDATOR_URL:-}" ]]; then
  args+=(--validator-url "${RELIQUARY_VALIDATOR_URL}")
fi

echo "Launching: reliquary mine ${args[*]}"
exec reliquary mine "${args[@]}"
