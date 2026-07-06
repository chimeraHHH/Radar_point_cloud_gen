#!/usr/bin/env bash
# G3 规模战恢复: 仅 Stage 3(双臂大模型)+ Stage 4(rollout). 用法: bash launch_fullscale_s34.sh [GPU1=2] [GPU2=6]
G1="${1:-2}"; G2="${2:-6}"
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
export TSROOT=/storage/data/metaiot_data/wangning_radar/truckscenes/man-truckscenes
export TSVER=v1.2-trainval
RES=~/Workspace/radar_gen/results
SC=~/Workspace/radar_gen/scripts
{
echo "[$(date)] ===== STAGE 3(恢复): 双臂大模型训练(dim512/depth8, L_temp, 60k 步) ====="
export VAL_N=12 STEPS=60000
CUDA_VISIBLE_DEVICES=$G1 python -u $SC/train_bridge.py dopp big_dopp temporal_full_k10 0 0.1 512 8 > $RES/bridge_big_dopp_log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 python -u $SC/train_bridge.py ego  big_ego  temporal_full_k10 0 0.1 512 8 > $RES/bridge_big_ego_log.txt 2>&1 &
wait
echo "[$(date)] stage3 done"
echo "[$(date)] ===== STAGE 4: rollout 三战(trainval val 场景) ====="
TAG=full VAL_N=12 SEGCAP=24 CUDA_VISIBLE_DEVICES=$G1 python -u $SC/eval_rollout.py big_dopp big_ego > $RES/rollout_full_log.txt 2>&1
echo "[$(date)] ===== ORCHESTRATOR_ALL_DONE ====="
} >> $RES/fullscale.log 2>&1
