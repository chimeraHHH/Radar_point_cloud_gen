#!/usr/bin/env bash
# P3 时序消融: ego 条件(GPU $1) vs dopp 条件(GPU $2)
G1="${1:-2}"; G2="${2:-6}"
pkill -f "python.*train_temporal" 2>/dev/null
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$G1 python ~/Workspace/radar_gen/scripts/train_temporal.py ego ts_ego > ~/Workspace/radar_gen/results/temporal_ts_ego_log.txt 2>&1" < /dev/null &
disown
setsid bash -c "CUDA_VISIBLE_DEVICES=$G2 python ~/Workspace/radar_gen/scripts/train_temporal.py dopp ts_dopp > ~/Workspace/radar_gen/results/temporal_ts_dopp_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED ts_ego@GPU$G1 ts_dopp@GPU$G2"
