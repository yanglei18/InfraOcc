#!/usr/bin/env bash
# Run the three camera-only baselines serially on the dedicated GPU set.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_IDS=${GPU_IDS:-0,1,2,3}
GPU_COUNT=${GPU_COUNT:-4}
ALOCC_SESSION=${ALOCC_SESSION:-alocc24_pretrained_dynamicflow}
ALOCC_WORK_DIR=${ALOCC_WORK_DIR:-work_dirs/alocc_c_2x4_24e_pretrained_dynamicflow}
LETOCCFLOW_WORK_DIR=${LETOCCFLOW_WORK_DIR:-work_dirs/stcroadocc/archive/external_projects/letoccflow_c_2x4_24e_pretrained_fixed}
LETOCCFLOW_LOG=${LETOCCFLOW_LOG:-${LETOCCFLOW_WORK_DIR}.log}
CRTFUSION_WORK_DIR=${CRTFUSION_WORK_DIR:-work_dirs/crtfusion_c_2x4_24e_pretrained_fixed}
CRTFUSION_LOG=${CRTFUSION_LOG:-${CRTFUSION_WORK_DIR}.log}

cd "$ROOT_DIR"

require_checkpoint() {
    local work_dir=$1
    if [[ ! -s "$work_dir/epoch_24.pth" ]]; then
        echo "Missing $work_dir/epoch_24.pth; aborting suite." >&2
        exit 1
    fi
}

echo "[$(date '+%F %T')] Waiting for ALOcc session $ALOCC_SESSION."
while tmux has-session -t "$ALOCC_SESSION" 2>/dev/null; do
    sleep 60
done
require_checkpoint "$ALOCC_WORK_DIR"
echo "[$(date '+%F %T')] ALOcc checkpoint verified."

run_training() {
    local config=$1
    local work_dir=$2
    local log_file=$3
    echo "[$(date '+%F %T')] Starting $config."
    CUDA_VISIBLE_DEVICES="$GPU_IDS" bash tools/dist_train.sh "$config" "$GPU_COUNT" \
        --work-dir "$work_dir" > "$log_file" 2>&1
    echo "[$(date '+%F %T')] Finished $config."
}

run_training \
    projects/LetOccFlow/configs/letoccflow_c_2x4_24e.py \
    "$LETOCCFLOW_WORK_DIR" \
    "$LETOCCFLOW_LOG"
require_checkpoint "$LETOCCFLOW_WORK_DIR"

run_training \
    projects/CRTFusion/configs/crtfusion_c_2x4_24e.py \
    "$CRTFUSION_WORK_DIR" \
    "$CRTFUSION_LOG"
require_checkpoint "$CRTFUSION_WORK_DIR"
