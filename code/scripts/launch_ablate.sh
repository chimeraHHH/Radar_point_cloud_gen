#!/usr/bin/env bash
# 双 GPU 并行消融: base(lam=0, GPU $1) vs phys(lam=$3, GPU $2). 默认 GPU 2/6, lam 0.1
G1="${1:-2}"; G2="${2:-6}"; LAM="${3:-0.1}"
pkill -f "python.*train_ablate" 2>/dev/null
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$G1 python ~/Workspace/radar_gen/scripts/train_ablate.py 0 base > ~/Workspace/radar_gen/results/ablate_base_log.txt 2>&1" < /dev/null &
disown
setsid bash -c "CUDA_VISIBLE_DEVICES=$G2 python ~/Workspace/radar_gen/scripts/train_ablate.py $LAM phys > ~/Workspace/radar_gen/results/ablate_phys_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED base@GPU$G1 phys(lam=$LAM)@GPU$G2"
