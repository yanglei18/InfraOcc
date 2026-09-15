# TORCH_DISTRIBUTED_DEBUG=DETAIL
# tmux new -s train3d
# conda activate roadocc
# Optional: force GLOO without editing config 
# — export OCC_V2X_DIST_BACKEND=gloo

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((29500 + RANDOM % 500))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
VISIBLE_CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun  --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}

# camera-only fair comparison
# nohup bash ./tools/dist_train.sh ./projects/InfraOcc/configs/infraocc_c_4x4_24e.py 4 > infraocc_c_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/v2xocc_c_4x4_24e.py 4 > v2xocc_c_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/BEVFormer/configs/bevformer_4x4_24e.py 4 > bevformer_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OccFormer/configs/occformer_4x4_24e.py 4 > occformer_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/SurroundOcc/configs/surroundocc_4x4_24e.py 4 > surroundocc_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/bevdet_4x4_24e.py 4 > bevdet_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/bevdepth_c_4x4_24e.py 4 > bevdepth_c_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/conet_c_4x4_24e.py 4 > conet_c_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/SparseOcc/configs/sparseocc_4x4_24e.py 4 > sparseocc_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/TPVFormer/configs/tpvformer_4x4_24e.py 4 > tpvformer_4x4_24e.log 2>&1 &

# nohup bash ./tools/dist_train.sh ./projects/InfraOcc/configs/infraocc_c_2x4_72e_progressive_sd.py 4 > infraocc_c_2x4_72e_progressive_sd.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/InfraOcc/configs/infraocc_c_2x4_24e_progressive_sd.py 4 > infraocc_c_2x4_24e_progressive_sd.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/InfraOcc/configs/infraocc_c_2x4_24e_progressive_sd_no_static_consistency.py 4 > infraocc_c_2x4_24e_progressive_sd_no_static_consistency.log 2>&1 &

# lidar-only fair comparison
# nohup bash ./tools/dist_train.sh ./projects/InfraOcc/configs/infraocc_l_4x4_24e.py 4 > infraocc_l_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/v2xocc_l_4x4_24e.py 4 > v2xocc_l_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/voxelnet_4x4_24e.py 4 > voxelnet_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/pointpillars_4x4_24e.py 4 > pointpillars_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/conet_l_4x4_24e.py 4 > conet_l_4x4_24e.log 2>&1 &

# multimodal fair comparison
# nohup bash ./tools/dist_train.sh ./projects/InfraOcc/configs/infraocc_m_4x4_24e.py 4 > infraocc_m_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/v2xocc_m_4x4_24e.py 4 > v2xocc_m_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/bevdepth_m_4x4_24e.py 4 > bevdepth_m_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/bevfusion_4x4_24e.py 4 > bevfusion_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/OpenOcc/configs/conet_m_4x4_24e.py 4 > conet_m_4x4_24e.log 2>&1 &

# ablation
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_whprior_woheads.py 4 > v2xocc_c_4x4_24e_whprior_woheads.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_woprior_whheads.py 4 > v2xocc_c_4x4_24e_woprior_whheads.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_woprior_woheads.py 4 > v2xocc_c_4x4_24e_woprior_woheads.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_woprior_woheads_worecursive.py 4 > v2xocc_c_4x4_24e_woprior_woheads_worecursive.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_whprior_whheads_worecursive.py 4 > v2xocc_c_4x4_24e_whprior_whheads_worecursive.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_whprior_woheads_worecursive.py 4 > v2xocc_c_4x4_24e_whprior_woheads_worecursive.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/V2XOcc/configs/ablation_study/v2xocc_c_4x4_24e_woprior_whheads_worecursive.py 4 > v2xocc_c_4x4_24e_woprior_whheads_worecursive.log 2>&1 &

# roadocc
# nohup bash ./tools/dist_train.sh ./projects/RoadOcc/configs/roadocc_c_4x4_72e_mf3.py 4 > roadocc_c_4x4_72e_mf3.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/RoadOcc/configs/roadocc_c_4x4_24e_mf3.py 4 > roadocc_c_4x4_24e_mf3.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/RoadOcc/configs/roadocc_c_4x4_24e.py 2 > roadocc_c_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/RoadOcc/configs/roadocc_m_4x4_24e.py 4 > roadocc_m_4x4_24e.log 2>&1 &
# nohup bash ./tools/dist_train.sh ./projects/RoadOcc/configs/roadocc_l_4x4_24e.py 4 > roadocc_l_4x4_24e.log 2>&1 &
