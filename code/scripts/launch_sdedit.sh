#!/usr/bin/env bash
# SDEdit 扫描: 两个 K=5 ckpt 串行跑在一张卡上. 用法: bash launch_sdedit.sh [GPU]
GPU="${1:-4}"
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU python ~/Workspace/radar_gen/scripts/eval_sdedit.py ego ts_ego > ~/Workspace/radar_gen/results/sdedit_ts_ego_log.txt 2>&1 && CUDA_VISIBLE_DEVICES=$GPU python ~/Workspace/radar_gen/scripts/eval_sdedit.py dopp ts_dopp > ~/Workspace/radar_gen/results/sdedit_ts_dopp_log.txt 2>&1" < /dev/null &
disown
echo "SDEDIT_LAUNCHED GPU$GPU"
