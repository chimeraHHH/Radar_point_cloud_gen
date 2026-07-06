#!/usr/bin/env bash
# G3 规模战编排器(过夜级): 缓存构建 → manifest 合并 → OT 配对 → 双臂大模型训练 → rollout
# 用法: bash launch_fullscale.sh [GPU1=2] [GPU2=6]
G1="${1:-2}"; G2="${2:-6}"
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
export TSROOT=/storage/data/metaiot_data/wangning_radar/truckscenes/man-truckscenes
export TSVER=v1.2-trainval
RES=~/Workspace/radar_gen/results
SC=~/Workspace/radar_gen/scripts
LOG=$RES/fullscale.log
PAIRS=~/data/radar_gen/truckscenes/temporal_full_k10
{
echo "[$(date)] ===== STAGE 1: 全量时序缓存(6 分片, stride=4, K=10) ====="
for i in 0 1 2 3 4 5; do
  python $SC/build_temporal_full.py $i 6 4 10 > $RES/full_build_$i.log 2>&1 &
done
wait
echo "[$(date)] stage1 done; 合并 manifest"
python - <<PY
import json, glob, os
P = os.path.expanduser("$PAIRS")
pairs = []
for f in sorted(glob.glob(f"{P}/manifest_shard*.json")):
    pairs += json.load(open(f))
json.dump(dict(pairs=pairs, K=10, stride=4), open(f"{P}/manifest.json", "w"))
print(f"merged {len(pairs)} pairs")
PY

echo "[$(date)] ===== STAGE 2: OT 配对(6 分片) ====="
for i in 0 1 2 3 4 5; do
  python $SC/build_bridge_perm.py temporal_full_k10 $i 6 > $RES/full_perm_$i.log 2>&1 &
done
wait
echo "[$(date)] stage2 done"

echo "[$(date)] ===== STAGE 3: 双臂大模型训练(dim512/depth8, L_temp, 60k 步) ====="
export VAL_N=12 STEPS=60000
CUDA_VISIBLE_DEVICES=$G1 python $SC/train_bridge.py dopp big_dopp temporal_full_k10 0 0.1 512 8 > $RES/bridge_big_dopp_log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 python $SC/train_bridge.py ego  big_ego  temporal_full_k10 0 0.1 512 8 > $RES/bridge_big_ego_log.txt 2>&1 &
wait
echo "[$(date)] stage3 done"

echo "[$(date)] ===== STAGE 4: rollout 三战(trainval val 场景) ====="
TAG=full VAL_N=12 SEGCAP=24 CUDA_VISIBLE_DEVICES=$G1 python $SC/eval_rollout.py big_dopp big_ego > $RES/rollout_full_log.txt 2>&1
echo "[$(date)] ===== ORCHESTRATOR_ALL_DONE ====="
} >> $LOG 2>&1
