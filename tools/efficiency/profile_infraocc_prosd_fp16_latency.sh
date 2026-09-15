#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: python was not found in PATH. Activate the intended environment or set PYTHON_BIN." >&2
  exit 127
fi
OUT_DIR="${OUT_DIR:-tools/efficiency/latency_accuracy_efficiency_fp16_20260710}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
PROFILE_ITERS="${PROFILE_ITERS:-20}"
PROFILE_GPU_ID="${PROFILE_GPU_ID:-0}"
VISIBLE_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT_DIR" /tmp/matplotlib-infraocc-fp16
export CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-infraocc-fp16}"

run_profile() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local output_json="$OUT_DIR/${name}.json"

    "$PYTHON_BIN" tools/infraocc/profile_model.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --split test \
        --gpu-id "$PROFILE_GPU_ID" \
        --warmup-iters "$WARMUP_ITERS" \
        --profile-iters "$PROFILE_ITERS" \
        --output-json "$output_json"
}

run_profile \
    c_prosd_occ_fp16 \
    projects/InfraOcc/configs/main_table_update/infraocc_c_4x4_36e_prosd_fp16.py \
    projects/InfraOcc/checkpoints_update/c_prosd.pth

run_profile \
    l_prosd_occ_fp16 \
    projects/InfraOcc/configs/main_table_update/infraocc_l_4x4_36e_prosd_fp16.py \
    projects/InfraOcc/checkpoints_update/l_prosd.pth

run_profile \
    m_prosd_occ_fp16 \
    projects/InfraOcc/configs/main_table_update/infraocc_m_4x4_36e_prosd_fp16.py \
    projects/InfraOcc/checkpoints_update/m_prosd.pth

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
def fmt(value):
    return "None" if value is None else f"{value:.4f}"

for path in sorted(out_dir.glob("*_fp16.json")):
    data = json.loads(path.read_text())
    latency = data.get("latency_ms")
    fps = data.get("fps")
    params = data.get("total_params_m")
    print(f"{path.stem}: params={fmt(params)}M latency={fmt(latency)}ms fps={fmt(fps)}")
PY
