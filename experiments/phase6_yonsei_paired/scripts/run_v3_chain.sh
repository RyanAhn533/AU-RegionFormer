#!/usr/bin/env bash
# v3 full chain: wait for stage6_v3 training → cross-cultural xeval → finetune compare
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

# Wait for stage6_v3
echo "[$(date)] waiting for stage6_v3 training..."
while pgrep -f "stage6_v3.yaml$" > /dev/null; do
  sleep 60
done
echo "[$(date)] stage6_v3 DONE"

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_v3/best.pth
[ -f "$INIT_CKPT" ] || { echo "FAIL no v3 ckpt"; exit 1; }

# 1. cross-cultural zero-shot eval on 5 datasets
for ds in affectnet ckplus sfew_train sfew_val afew_train afew_val; do
  CSV=$PHASE/csvs_v3/${ds}_4c.csv
  [ -f "$CSV" ] || { echo "SKIP $ds"; continue; }
  OUT=$PHASE/results/xeval_v3_${ds}_phase6_stage6_v3
  [ -f "$OUT/crossdataset_summary.json" ] && { echo "SKIP $ds (done)"; continue; }
  mkdir -p "$OUT"
  echo "[$(date)] xeval v3 $ds"
  python $PHASE/scripts/cross_dataset_zeroshot.py \
    --config $PHASE/configs/stage6_v3.yaml \
    --ckpt "$INIT_CKPT" --val_csv "$CSV" --out_dir "$OUT" \
    > $PHASE/logs/xeval_v3_${ds}.log 2>&1 || true
done
echo "[$(date)] xeval v3 batch DONE"

# 2. AffectNet finetune (Korean prior) + scratch baseline (8 epoch each)
for tag in finetune scratch; do
  OUT_DIR=/mnt/hdd/ajy_25/results/phase6_v3_${tag}_affectnet
  [ -d "$OUT_DIR" ] && rm -rf "$OUT_DIR"
done

# Generate v3 finetune config from stage6_v3
python << 'PYEOF'
import yaml
y = yaml.safe_load(open("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/configs/stage6_v3.yaml"))
import copy
ft = copy.deepcopy(y)
ft["paths"]["train_csv"] = "/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs_v3/affectnet_4c.csv"
ft["paths"]["val_csv"]   = "/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs_v3/affectnet_4c.csv"
ft["paths"]["output_dir"] = "/mnt/hdd/ajy_25/results/phase6_v3_finetune_affectnet"
ft["training"]["epochs"] = 8
ft["training"]["freeze_backbone_epochs"] = 1
ft["training"]["base_lr"] = 1e-4
ft["training"]["batch_size"] = 64
ft["training"]["early_stop_patience"] = 3
ft["training"]["ema"] = False
ft["training"]["distill_alpha"] = 0
ft["training"]["mixup_alpha"] = 0
ft["training"]["beta_anchor_weight"] = 0
ft["training"]["beta_var_weight"] = 0
yaml.safe_dump(ft, open("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/configs/v3_finetune_affectnet.yaml", "w"), sort_keys=False)
sc = copy.deepcopy(ft)
sc["paths"]["output_dir"] = "/mnt/hdd/ajy_25/results/phase6_v3_scratch_affectnet"
yaml.safe_dump(sc, open("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/configs/v3_scratch_affectnet.yaml", "w"), sort_keys=False)
print("v3 finetune+scratch configs written")
PYEOF

echo "[$(date)] === v3 finetune affectnet (Korean prior init) ==="
python scripts/train.py --config $PHASE/configs/v3_finetune_affectnet.yaml --init_from "$INIT_CKPT" \
    > $PHASE/logs/v3_finetune_affectnet.log 2>&1 || true

echo "[$(date)] === v3 scratch affectnet (no Korean prior) ==="
python scripts/train.py --config $PHASE/configs/v3_scratch_affectnet.yaml \
    > $PHASE/logs/v3_scratch_affectnet.log 2>&1 || true

echo "[$(date)] === v3 chain DONE ==="
