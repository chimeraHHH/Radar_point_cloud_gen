#!/usr/bin/env bash
# 双 GPU 消融轮3: ego_phys3(静0.1/动0, GPU $1) vs ego_dyn(静0.1/动0.1, GPU $2)
G1="${1:-2}"; G2="${2:-6}"
pkill -f "python.*train_ablate" 2>/dev/null
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
setsid bash -c "CUDA_VISIBLE_DEVICES=$G1 python ~/Workspace/radar_gen/scripts/train_ablate.py 0.1 0 ego_phys3 > ~/Workspace/radar_gen/results/ablate_ego_phys3_log.txt 2>&1" < /dev/null &
disown
setsid bash -c "CUDA_VISIBLE_DEVICES=$G2 python ~/Workspace/radar_gen/scripts/train_ablate.py 0.1 0.1 ego_dyn > ~/Workspace/radar_gen/results/ablate_ego_dyn_log.txt 2>&1" < /dev/null &
disown
echo "LAUNCHED ego_phys3@GPU$G1 ego_dyn@GPU$G2"
