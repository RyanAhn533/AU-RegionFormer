#!/usr/bin/env bash
# Paper Table 1 controls: Korean prior vs scratch baseline on 3 Western datasets
# (AffectNet already done; this fills SFEW / AFEW / CK+ cells)
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_full/best.pth

run_finetune() {
  local TAG=$1   # finetune or scratch
  local DS=$2    # sfew_train / afew_train / ckplus
  local CFG=$PHASE/configs/finetune/${TAG}_${DS}.yaml
  local OUT=/mnt/hdd/ajy_25/results/phase6_${TAG}_${DS}/best.pth
  if [ -f "$OUT" ]; then
    echo "[$(date)] SKIP ${TAG}_${DS}"
    return
  fi
  if [ ! -f "$CFG" ]; then
    echo "[$(date)] no config: $CFG"
    return
  fi
  echo "[$(date)] === ${TAG}_${DS} ==="
  if [ "$TAG" = "finetune" ]; then
    python scripts/train.py --config "$CFG" --init_from "$INIT_CKPT" \
        > $PHASE/logs/${TAG}_${DS}.log 2>&1 || true
  else
    python scripts/train.py --config "$CFG" \
        > $PHASE/logs/${TAG}_${DS}.log 2>&1 || true
  fi
  echo "[$(date)] ${TAG}_${DS} done"
}

# Sequential pairs: Korean-prior + scratch on each small dataset
for DS in ckplus sfew_train afew_train; do
  run_finetune finetune $DS
  run_finetune scratch $DS
done

echo "[$(date)] === PAPER CONTROLS DONE ==="
