#!/usr/bin/env bash
# Single-region ablation chain: 10 spots × (train + cross-cultural xeval)
# 외국 + 한국인 어떤 region이 가장 informative?
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

# 이전 chain들 끝나길 대기
echo "[$(date)] waiting for prior chains..."
while pgrep -f "run_patch_ablation_chain.sh\|run_au_subset_chain.sh" > /dev/null; do
  sleep 120
done
echo "[$(date)] prior chains DONE"

run_one() {
  local NAME=$1
  local CFG=$PHASE/configs/${NAME}.yaml
  local CKPT=/mnt/hdd/ajy_25/results/phase6_${NAME}/best.pth
  local LOG=$PHASE/logs/${NAME}.log
  if [ -f "$CKPT" ]; then
    echo "[$(date)] SKIP train $NAME"
  else
    echo "[$(date)] === TRAIN: $NAME ==="
    python scripts/train.py --config "$CFG" > "$LOG" 2>&1 || true
    echo "[$(date)] $NAME train done"
  fi
  for ds in affectnet ckplus sfew_train sfew_val afew_train afew_val; do
    local CSV=$PHASE/csvs_v3/${ds}_4c.csv
    [ -f "$CSV" ] || continue
    local OUT=$PHASE/results/xeval_v3_${ds}_phase6_${NAME}
    [ -f "$OUT/crossdataset_summary.json" ] && continue
    [ -f "$CKPT" ] || continue
    mkdir -p "$OUT"
    python $PHASE/scripts/cross_dataset_zeroshot.py \
      --config "$CFG" --ckpt "$CKPT" --val_csv "$CSV" --out_dir "$OUT" \
      > $PHASE/logs/xeval_v3_${ds}_${NAME}.log 2>&1 || true
  done
  echo "[$(date)] $NAME xeval done"
}

# 10 single-region (face 위→아래, 좌→우)
for R in forehead_69 forehead_299 forehead_9 \
         eyes_159 eyes_386 nose \
         cheeks_186 cheeks_410 mouth chin; do
  run_one stage6_v3_only_${R}
done

echo "[$(date)] === SINGLE REGION CHAIN DONE ==="
