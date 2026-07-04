#!/usr/bin/env bash
# 服务器端一键启动基线正式训练; 用法: bash launch_baseline.sh [GPU]
GPU="${1:-6}"
pkill -f "python.*train_baseline" 2>/dev/null
rm -f ~/Workspace/radar_gen/results/baseline_log.txt
setsid bash -c "source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh \
  && conda activate hym_radar \
  && CUDA_VISIBLE_DEVICES=$GPU python ~/Workspace/radar_gen/scripts/train_baseline.py \
     > ~/Workspace/radar_gen/results/baseline_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED_GPU$GPU"
