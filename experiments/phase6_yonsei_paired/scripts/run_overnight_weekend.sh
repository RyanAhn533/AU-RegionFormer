#!/usr/bin/env bash
# Overnight + weekend full queue
# Order:
#   1. Multi-seed Stage 6 (123, 777)
#   2. Stage 9 analysis on Stages 6, 7c, 7, 8, 6_seed123, 6_seed777
#   3. Stage 11 Human-KL sweep (λ=0.1, 0.3, 0.5)
#   4. Stage 13 no-Stage-A ablation
#   5. Stage 9 analysis on Stage 11 best + Stage 13
set -uo pipefail
cd /home/ajy/AU-RegionFormer
PHASE=experiments/phase6_yonsei_paired

run_train() {
  local STAGE=$1
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config $PHASE/configs/${STAGE}.yaml \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
  echo "[$(date)] $STAGE exit=$?"
}

run_stage9() {
  local NAME=$1   # ckpt name (e.g., phase6_stage6_full)
  local CFG=$2    # config file (e.g., stage6_full)
  local CKPT="/mnt/hdd/ajy_25/results/${NAME}/best.pth"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date)] SKIP stage9 for $NAME — no ckpt"
    return
  fi
  local OUT="$PHASE/results/stage9_${NAME}"
  mkdir -p "$OUT"
  echo "[$(date)] === Stage 9 analysis for $NAME ==="
  python $PHASE/scripts/stage9_human_perception_alignment.py \
      --config $PHASE/configs/${CFG}.yaml \
      --ckpt "$CKPT" \
      --out_dir "$OUT" \
      > $PHASE/logs/stage9_${NAME}.log 2>&1 || true
  echo "[$(date)] stage9 $NAME exit=$?"
}

# ─── 1. Multi-seed Stage 6 (4 extra seeds for solid stats) ───
run_train stage6_seed123
run_train stage6_seed777
run_train stage6_seed999
run_train stage6_seed2024

# ─── 2. Stage 9 analyses on existing ckpts ───
run_stage9 phase6_stage6_full          stage6_full
run_stage9 phase6_stage7c_patch_dropout stage7c_patch_dropout
run_stage9 phase6_stage7_jepa          stage7_jepa
run_stage9 phase6_stage8_dinov2        stage8_dinov2
run_stage9 phase6_stage6_seed123       stage6_seed123
run_stage9 phase6_stage6_seed777       stage6_seed777
run_stage9 phase6_stage6_seed999       stage6_seed999
run_stage9 phase6_stage6_seed2024      stage6_seed2024

# ─── 3. Stage 11 Human-KL sweep (5 lambdas) ───
run_train stage11_humankl_lam01
run_train stage11_humankl_lam03
run_train stage11_humankl_lam05
run_train stage11_humankl_lam07
run_train stage11_humankl_lam10

# ─── 4. Stage 13 no-Stage-A ablation ───
run_train stage13_no_stage_a

# ─── 5. Stage 9 on new Stage 11/13 ───
run_stage9 phase6_stage11_humankl_lam01 stage11_humankl_lam01
run_stage9 phase6_stage11_humankl_lam03 stage11_humankl_lam03
run_stage9 phase6_stage11_humankl_lam05 stage11_humankl_lam05
run_stage9 phase6_stage11_humankl_lam07 stage11_humankl_lam07
run_stage9 phase6_stage11_humankl_lam10 stage11_humankl_lam10
run_stage9 phase6_stage13_no_stage_a    stage13_no_stage_a


# ─── 6. Cross-dataset zero-shot eval (AffectNet + SFEW Train) on key ckpts ───
run_xeval() {
  local NAME=$1; local CFG=$2; local CSV=$3; local TAG=$4
  local CKPT="/mnt/hdd/ajy_25/results/${NAME}/best.pth"
  if [ ! -f "$CKPT" ]; then echo "[$(date)] SKIP xeval $NAME ($TAG)"; return; fi
  local OUT="$PHASE/results/xeval_${TAG}_${NAME}"
  mkdir -p "$OUT"
  echo "[$(date)] === xeval $NAME on $TAG ==="
  python $PHASE/scripts/cross_dataset_zeroshot.py \
      --config $PHASE/configs/${CFG}.yaml \
      --ckpt "$CKPT" \
      --val_csv "$CSV" \
      --out_dir "$OUT" \
      > $PHASE/logs/xeval_${TAG}_${NAME}.log 2>&1 || true
}

CSVS=(
  "affectnet:$PHASE/csvs/crossdataset/affectnet_4c.csv"
  "sfew_train:$PHASE/csvs/crossdataset/sfew_train_4c.csv"
  "sfew_val:$PHASE/csvs/crossdataset/sfew_val_4c.csv"
  "afew_train:$PHASE/csvs/crossdataset/afew_train_4c.csv"
  "afew_val:$PHASE/csvs/crossdataset/afew_val_4c.csv"
  "ckplus:$PHASE/csvs/crossdataset/ckplus_4c.csv"
)

for NAME in phase6_stage6_full phase6_stage6_seed123 phase6_stage6_seed777 \
            phase6_stage6_seed999 phase6_stage6_seed2024 \
            phase6_stage7c_patch_dropout phase6_stage7_jepa phase6_stage8_dinov2 \
            phase6_stage11_humankl_lam01 phase6_stage11_humankl_lam03 \
            phase6_stage11_humankl_lam05 phase6_stage11_humankl_lam07 \
            phase6_stage11_humankl_lam10 phase6_stage13_no_stage_a; do
  CFG=$(echo $NAME | sed 's/phase6_//')
  for entry in "${CSVS[@]}"; do
    TAG="${entry%%:*}"
    CSV="${entry#*:}"
    [ -f "$CSV" ] || continue
    run_xeval "$NAME" "$CFG" "$CSV" "$TAG"
  done
done

echo "[$(date)] === WEEKEND QUEUE DONE ==="


# ─── 7. Final aggregation ───
echo "[$(date)] === FINAL AGGREGATION ==="
python $PHASE/scripts/aggregate_results.py > $PHASE/logs/aggregate.log 2>&1
echo "[$(date)] aggregation done"

