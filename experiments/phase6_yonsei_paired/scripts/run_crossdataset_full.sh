#!/usr/bin/env bash
# Build all available Western/cross-cultural datasets to master CSV format.
# Run AFTER permissions are fully fixed for /mnt/ssd2.
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer
mkdir -p "$PHASE/csvs/crossdataset"
mkdir -p "$PHASE/logs"

# 1. AffectNet (already done partial; redo full when nested perms cleared)
if ls /mnt/ssd2/AffectNet/anger/ >/dev/null 2>&1; then
  python $PHASE/scripts/build_crossdataset_csv.py \
      --src /mnt/ssd2/AffectNet \
      --out $PHASE/csvs/crossdataset/affectnet_full_4c.csv \
      --label_map "anger:angry, happy:happy, neutral:neutral, sad:sad" \
      --workers 12 > $PHASE/logs/x_affectnet_full.log 2>&1 || true
fi

# 2. SFEW Train + Val (Val needs zip extraction first)
python $PHASE/scripts/build_crossdataset_csv.py \
    --src /mnt/ssd2/SFEW_2/Train --out $PHASE/csvs/crossdataset/sfew_train_4c.csv \
    --label_map "Angry:angry, Happy:happy, Neutral:neutral, Sad:sad" \
    --workers 12 > $PHASE/logs/x_sfew_train.log 2>&1 || true
if ls /mnt/ssd2/SFEW_2/Val/Angry/ >/dev/null 2>&1; then
  python $PHASE/scripts/build_crossdataset_csv.py \
      --src /mnt/ssd2/SFEW_2/Val --out $PHASE/csvs/crossdataset/sfew_val_4c.csv \
      --label_map "Angry:angry, Happy:happy, Neutral:neutral, Sad:sad" \
      --workers 12 > $PHASE/logs/x_sfew_val.log 2>&1 || true
fi

# 3. AFEW (videos → frames → CSV)
if ls "/mnt/ssd2/AFEW(이미지)/Train_AFEW/Angry/" >/dev/null 2>&1; then
  python $PHASE/scripts/extract_video_frames.py \
      --src "/mnt/ssd2/AFEW(이미지)/Train_AFEW" \
      --out /tmp/afew_train_frames \
      --classes "Angry,Happy,Neutral,Sad" --n_per_video 3 --workers 12 \
      > $PHASE/logs/x_afew_extract.log 2>&1 || true
  python $PHASE/scripts/build_crossdataset_csv.py \
      --src /tmp/afew_train_frames --out $PHASE/csvs/crossdataset/afew_train_4c.csv \
      --label_map "Angry:angry, Happy:happy, Neutral:neutral, Sad:sad" \
      --workers 12 > $PHASE/logs/x_afew_csv.log 2>&1 || true
fi

# 4. DFEW (videos → frames). Heavy.
if ls /mnt/ssd2/DFEW/DFEW-part1/ >/dev/null 2>&1; then
  echo "DFEW pipeline TODO (heavy, defer)"
fi

# 5. CK+ (lab-controlled, when nested perms fixed). Need to inspect structure first.
if ls "/mnt/ssd2/CK+&CK/OneDrive_2024-12-19/CK+/" >/dev/null 2>&1; then
  echo "CK+ pipeline TODO (depends on internal structure)"
fi

ls -lh $PHASE/csvs/crossdataset/
