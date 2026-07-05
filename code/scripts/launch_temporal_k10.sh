#!/usr/bin/env bash
# K=10 时序消融: ego(GPU $1) vs dopp(GPU $2)
G1="${1:-2}"; G2="${2:-6}"
pkill -f "python.*train_temporal" 2>/dev/null
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$G1 python ~/Workspace/radar_gen/scripts/train_temporal.py ego ts_ego10 temporal_mini_k10 > ~/Workspace/radar_gen/results/temporal_ts_ego10_log.txt 2>&1" < /dev/null &
disown
setsid bash -c "CUDA_VISIBLE_DEVICES=$G2 python ~/Workspace/radar_gen/scripts/train_temporal.py dopp ts_dopp10 temporal_mini_k10 > ~/Workspace/radar_gen/results/temporal_ts_dopp10_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED ts_ego10@GPU$G1 ts_dopp10@GPU$G2"
