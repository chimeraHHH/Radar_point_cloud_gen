#!/usr/bin/env bash
# scheduled sampling 编排器: 三元组构建(6分片) → SS 训练(GPU $1) → rollout(ss vs big_dopp)
G1="${1:-2}"
source /home/metaiot_guest/miniconda3/etc/profile.d/conda.sh
conda activate hym_radar
export TSROOT=/storage/data/metaiot_data/wangning_radar/truckscenes/man-truckscenes
export TSVER=v1.2-trainval
RES=~/Workspace/radar_gen/results
SC=~/Workspace/radar_gen/scripts
P=~/data/radar_gen/truckscenes/triples_full_k10
{
echo "[$(date)] ===== SS-STAGE 1: 三元组缓存(6 分片) ====="
for i in 0 1 2 3 4 5; do
  python -u $SC/build_triples_full.py $i 6 4 10 > $RES/triples_build_$i.log 2>&1 &
done
wait
python - <<PY
import json, glob, os
P = os.path.expanduser("$P")
pairs = []
for f in sorted(glob.glob(f"{P}/manifest_shard*.json")):
    pairs += json.load(open(f))
json.dump(dict(pairs=pairs), open(f"{P}/manifest.json", "w"))
print(f"merged {len(pairs)} triples")
PY
echo "[$(date)] ===== SS-STAGE 2: scheduled sampling 训练 ====="
VAL_N=12 STEPS=60000 CUDA_VISIBLE_DEVICES=$G1 python -u $SC/train_bridge_ss.py ss_dopp > $RES/bridge_ss_dopp_log.txt 2>&1
echo "[$(date)] ===== SS-STAGE 3: rollout(ss_dopp vs big_dopp) ====="
TAG=ss VAL_N=12 SEGCAP=24 CUDA_VISIBLE_DEVICES=$G1 python -u $SC/eval_rollout.py ss_dopp big_dopp > $RES/rollout_ss_log.txt 2>&1
echo "[$(date)] ===== SS_ORCHESTRATOR_DONE ====="
} >> $RES/ss_orchestrator.log 2>&1
