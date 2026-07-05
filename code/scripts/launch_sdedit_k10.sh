#!/usr/bin/env bash
# K=10 ckpt 的 SDEdit 扫描. 用法: bash launch_sdedit_k10.sh [GPU]
GPU="${1:-4}"
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU python ~/Workspace/radar_gen/scripts/eval_sdedit.py ego ts_ego10 temporal_mini_k10 > ~/Workspace/radar_gen/results/sdedit_ts_ego10_log.txt 2>&1 && CUDA_VISIBLE_DEVICES=$GPU python ~/Workspace/radar_gen/scripts/eval_sdedit.py dopp ts_dopp10 temporal_mini_k10 > ~/Workspace/radar_gen/results/sdedit_ts_dopp10_log.txt 2>&1" < /dev/null &
disown
echo "SDEDIT_K10_LAUNCHED GPU$GPU"
