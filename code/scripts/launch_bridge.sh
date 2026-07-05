#!/usr/bin/env bash
# 桥式扩散双臂: bridge_ego(GPU $1) vs bridge_dopp(GPU $2), K=10
G1="${1:-2}"; G2="${2:-6}"
pkill -f "python.*train_bridge" 2>/dev/null
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$G1 python ~/Workspace/radar_gen/scripts/train_bridge.py ego br_ego > ~/Workspace/radar_gen/results/bridge_br_ego_log.txt 2>&1" < /dev/null &
disown
setsid bash -c "CUDA_VISIBLE_DEVICES=$G2 python ~/Workspace/radar_gen/scripts/train_bridge.py dopp br_dopp > ~/Workspace/radar_gen/results/bridge_br_dopp_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED br_ego@GPU$G1 br_dopp@GPU$G2"
