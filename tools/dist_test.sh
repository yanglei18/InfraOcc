#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((29500 + RANDOM % 500))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
VISIBLE_CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES" \
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    tools/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --launcher pytorch "${@:4}"
