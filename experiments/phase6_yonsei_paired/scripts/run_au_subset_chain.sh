#!/usr/bin/env bash
# AU subset ablation chain: 3 핵심 variants × (train + cross-cultural xeval)
# 외국 데이터셋에 어떤 region 조합이 가장 잘 transfer 되는지 검증
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

# patch ablation chain 끝나길 대기
echo "[$(date)] waiting for patch_ablation_chain to finish..."
while pgrep -f "run_patch_ablation_chain.sh" > /dev/null; do
  sleep 120
done
echo "[$(date)] patch_ablation_chain DONE"

run_variant() {
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

# 3 핵심 variants (시간 제약상 3개만)
run_variant stage6_v3_core6        # 핵심 6 (forehead_9 + eyes 2 + nose + mouth + chin)
run_variant stage6_v3_upper_face   # 위쪽 6 (forehead 3 + eyes 2 + nose)
run_variant stage6_v3_lower_face   # 아래쪽 4 (cheeks 2 + mouth + chin)

echo "[$(date)] === AU SUBSET CHAIN DONE ==="
