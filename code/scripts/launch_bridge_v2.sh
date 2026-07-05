#!/usr/bin/env bash
# G3 再战双臂: v2aug(仅增广, GPU $1) vs v2full(增广+L_temp, GPU $2), dopp 草稿
G1="${1:-2}"; G2="${2:-6}"
pkill -f "python.*train_bridge" 2>/dev/null
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$G1 python ~/Workspace/radar_gen/scripts/train_bridge.py dopp v2aug temporal_mini_k10 1 0 > ~/Workspace/radar_gen/results/bridge_v2aug_log.txt 2>&1" < /dev/null &
disown
setsid bash -c "CUDA_VISIBLE_DEVICES=$G2 python ~/Workspace/radar_gen/scripts/train_bridge.py dopp v2full temporal_mini_k10 1 0.1 > ~/Workspace/radar_gen/results/bridge_v2full_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED v2aug@GPU$G1 v2full@GPU$G2"
