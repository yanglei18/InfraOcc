#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((29500 + RANDOM % 500))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
VISIBLE_CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

# camera-only fair comparison
# bash tools/dist_test.sh projects/V2XOcc/configs/v2xocc_c_4x4_24e.py projects/V2XOcc/checkpoints/v2xocc_c.pth 4
# bash tools/dist_test.sh projects/TPVFormer/configs/tpvformer_4x4_24e.py projects/TPVFormer/checkpoints/tpvformer.pth 4
# bash tools/dist_test.sh projects/BEVFormer/configs/bevformer_4x4_24e.py projects/BEVFormer/checkpoints/bevformer.pth 4
# bash tools/dist_test.sh projects/OccFormer/configs/occformer_4x4_24e.py projects/OccFormer/checkpoints/occformer.pth 4
# bash tools/dist_test.sh projects/SurroundOcc/configs/surroundocc_4x4_24e.py projects/SurroundOcc/checkpoints/surroundocc.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/bevdet_4x4_24e.py projects/OpenOcc/checkpoints/bevdet.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/bevdepth_c_4x4_24e.py projects/OpenOcc/checkpoints/bevdepth_c.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/conet_c_4x4_24e.py projects/OpenOcc/checkpoints/conet_c.pth 4

# lidar-only fair comparison
# bash tools/dist_test.sh projects/V2XOcc/configs/v2xocc_l_4x4_24e.py projects/V2XOcc/checkpoints/v2xocc_l.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/voxelnet_4x4_24e.py projects/OpenOcc/checkpoints/voxelnet.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/pointpillars_4x4_24e.py projects/OpenOcc/checkpoints/pointpillars.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/conet_l_4x4_24e.py projects/OpenOcc/checkpoints/conet_l.pth 4

# multimodal fair comparison
# bash tools/dist_test.sh projects/V2XOcc/configs/v2xocc_m_4x4_24e.py projects/V2XOcc/checkpoints/v2xocc_m.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/bevdepth_m_4x4_24e.py projects/OpenOcc/checkpoints/bevdepth_m.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/bevfusion_4x4_24e.py projects/OpenOcc/checkpoints/bevfusion.pth 4
# bash tools/dist_test.sh projects/OpenOcc/configs/conet_m_4x4_24e.py projects/OpenOcc/checkpoints/conet_m.pth 4

# ablation
# bash tools/dist_test.sh projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_whprior_woheads.py projects/V2XOcc/checkpoints/v2xocc_c_whprior_woheads.pth 4
# bash tools/dist_test.sh projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_woprior_whheads.py projects/V2XOcc/checkpoints/v2xocc_c_woprior_whheads.pth 4
# bash tools/dist_test.sh projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_woprior_woheads.py projects/V2XOcc/checkpoints/v2xocc_c_woprior_woheads.pth 4

# roadocc
# bash tools/dist_test.sh projects/RoadOcc/configs/roadocc_c_4x4_24e.py work_dirs/roadocc_c_4x4_24e/epoch_24.pth 4
# bash tools/dist_test.sh projects/RoadOcc/configs/roadocc_m_4x4_24e.py projects/RoadOcc/checkpoints/roadocc_m.pth 4
# bash tools/dist_test.sh projects/RoadOcc/configs/roadocc_l_4x4_24e.py projects/RoadOcc/checkpoints/roadocc_l.pth 4
