#!/usr/bin/env bash
# 服务器端一键启动过拟合冒烟(默认 GPU6), 幂等: 先杀旧进程
GPU="${1:-6}"
pkill -f "python.*train_overfit" 2>/dev/null
rm -f ~/Workspace/radar_gen/results/overfit_log.txt
setsid bash -c "source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh \
  && conda activate hym_radar \
  && CUDA_VISIBLE_DEVICES=$GPU python ~/Workspace/radar_gen/scripts/train_overfit.py \
     > ~/Workspace/radar_gen/results/overfit_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED_GPU$GPU"
