#!/usr/bin/env bash
# Sunday queue: cross-dataset finetune from Stage 6 best.pth on each Western dataset.
# Run AFTER overnight queue completes.
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

mkdir -p $PHASE/configs/finetune
mkdir -p $PHASE/logs

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_full/best.pth
if [ ! -f "$INIT_CKPT" ]; then echo "FAIL: no Stage 6 ckpt"; exit 1; fi

# Generate finetune configs from stage6_full base
for DS in affectnet sfew_train afew_train ckplus; do
  CSV=$PHASE/csvs/crossdataset/${DS}_4c.csv
  [ -f "$CSV" ] || { echo "SKIP $DS: no CSV"; continue; }
  CFG=$PHASE/configs/finetune/finetune_${DS}.yaml
  python -c "
import yaml
y = yaml.safe_load(open('$PHASE/configs/stage6_full.yaml'))
y['paths']['train_csv'] = '$CSV'
y['paths']['val_csv']   = '$CSV'   # eval on same (no separate val for some); will overfit but signal
y['paths']['output_dir'] = '/mnt/hdd/ajy_25/results/phase6_finetune_${DS}'
y['training']['epochs'] = 8
y['training']['freeze_backbone_epochs'] = 1
y['training']['base_lr'] = 1e-4   # finetune-friendly (low)
y['training']['batch_size'] = 64
y['training']['early_stop_patience'] = 3
y['training']['ema'] = False
y['training']['distill_alpha'] = 0
y['training']['mixup_alpha'] = 0
y['training']['beta_anchor_weight'] = 0
y['training']['beta_var_weight'] = 0
yaml.safe_dump(y, open('$CFG','w'), sort_keys=False)
print('$CFG')
"
done

# Run each finetune sequentially with --resume (loads Stage 6 weights as init)
for DS in affectnet sfew_train afew_train ckplus; do
  CFG=$PHASE/configs/finetune/finetune_${DS}.yaml
  [ -f "$CFG" ] || { echo "SKIP $DS"; continue; }
  echo "[$(date)] === FINETUNE: $DS ==="
  python scripts/train.py --config "$CFG" --init_from "$INIT_CKPT" \
      > $PHASE/logs/finetune_${DS}.log 2>&1 || true
  echo "[$(date)] $DS exit=$?"
done

echo "[$(date)] === SUNDAY FINETUNE QUEUE DONE ==="

# Stage 14 + 15 (probabilistic Beta + backbone variants) + Stage 16 ablation matrix
for STAGE in stage14_prob_beta stage15_mambaout_femto stage15_convnextv2_tiny stage15_fastvit_t8 \
             stage16a_iso_only stage16b_rel_only stage16c_a_only stage16d_product; do
  CFG="$PHASE/configs/${STAGE}.yaml"
  [ -f "$CFG" ] || continue
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config "$CFG" \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
done

echo "[$(date)] === STAGE 14/15/16 DONE ==="

# Cross-dataset eval on every new ckpt (skip-cached)
for NAME in phase6_stage14_prob_beta phase6_stage15_mambaout_femto phase6_stage15_convnextv2_tiny \
            phase6_stage15_fastvit_t8 phase6_stage16a_iso_only phase6_stage16b_rel_only \
            phase6_stage16c_a_only phase6_stage16d_product phase6_stage13_no_stage_a; do
  CKPT="/mnt/hdd/ajy_25/results/${NAME}/best.pth"
  [ -f "$CKPT" ] || continue
  CFG_NAME=$(echo $NAME | sed 's/phase6_//')
  for csv_entry in "affectnet:$PHASE/csvs/crossdataset/affectnet_4c.csv" \
                   "sfew_train:$PHASE/csvs/crossdataset/sfew_train_4c.csv" \
                   "sfew_val:$PHASE/csvs/crossdataset/sfew_val_4c.csv" \
                   "afew_train:$PHASE/csvs/crossdataset/afew_train_4c.csv" \
                   "afew_val:$PHASE/csvs/crossdataset/afew_val_4c.csv" \
                   "ckplus:$PHASE/csvs/crossdataset/ckplus_4c.csv"; do
    TAG="${csv_entry%%:*}"; CSV="${csv_entry#*:}"
    [ -f "$CSV" ] || continue
    OUT="$PHASE/results/xeval_${TAG}_${NAME}"
    [ -f "$OUT/crossdataset_summary.json" ] && continue
    mkdir -p "$OUT"
    python $PHASE/scripts/cross_dataset_zeroshot.py \
        --config $PHASE/configs/${CFG_NAME}.yaml \
        --ckpt "$CKPT" --val_csv "$CSV" --out_dir "$OUT" \
        > $PHASE/logs/xeval_${TAG}_${NAME}.log 2>&1 || true
  done
done

python $PHASE/scripts/aggregate_results.py >> $PHASE/logs/aggregate.log 2>&1
