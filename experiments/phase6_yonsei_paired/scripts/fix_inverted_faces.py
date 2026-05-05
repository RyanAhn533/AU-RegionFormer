#!/usr/bin/env python3
"""
v3 inverted face fix: 모든 CSV에서 face가 거꾸로 detected된 row 찾아서
  1. Image 180도 회전 후 in-place 저장
  2. CSV의 좌표 (cx, cy, wx1, wy1, wx2, wy2) 변환
     cx_new = work_w - cx_old
     cy_new = work_h - cy_old
     wx1/wx2 swap, wy1/wy2 swap (and reflected)

Detection criterion: forehead_9_cy > eyes_avg_cy OR mouth_cy > chin_cy
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

CSVS_V3 = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs_v3")
DATASETS = [
    "yonsei_train", "yonsei_val",
    "affectnet", "ckplus",
    "sfew_train", "sfew_val",
    "afew_train", "afew_val",
]

REGIONS = ["forehead_69","forehead_299","forehead_9","eyes_159","eyes_386",
           "nose","cheeks_186","cheeks_410","mouth","chin"]


def is_inverted(row):
    """forehead 위쪽 / chin 아래쪽 위반 = 거꾸로 face"""
    fh_y = float(row["forehead_9_cy"])
    eyes_y = (float(row["eyes_159_cy"]) + float(row["eyes_386_cy"])) / 2
    mouth_y = float(row["mouth_cy"])
    chin_y = float(row["chin_cy"])
    # 핵심 조건: forehead가 eyes 아래 OR mouth가 chin 아래
    return (fh_y > eyes_y) or (mouth_y > chin_y)


def flip_coords(row):
    """180도 회전 후 새 좌표 row 만듦. work_w/h 그대로."""
    W = float(row["work_w"]); H = float(row["work_h"])
    new = row.copy()
    for r in REGIONS:
        cx_old = float(row[f"{r}_cx"]); cy_old = float(row[f"{r}_cy"])
        x1_old = float(row[f"{r}_wx1"]); y1_old = float(row[f"{r}_wy1"])
        x2_old = float(row[f"{r}_wx2"]); y2_old = float(row[f"{r}_wy2"])
        # 180도 회전: x' = W-x, y' = H-y
        new[f"{r}_cx"] = W - cx_old
        new[f"{r}_cy"] = H - cy_old
        new[f"{r}_wx1"] = W - x2_old   # swap + reflect
        new[f"{r}_wy1"] = H - y2_old
        new[f"{r}_wx2"] = W - x1_old
        new[f"{r}_wy2"] = H - y1_old
    return new


def fix_dataset(name):
    csv_path = CSVS_V3 / f"{name}_4c.csv"
    if not csv_path.exists():
        print(f"  SKIP {name} (no csv)")
        return 0, 0
    df = pd.read_csv(csv_path, low_memory=False)
    n_total = len(df)

    inverted_idx = []
    for idx, row in df.iterrows():
        if is_inverted(row):
            inverted_idx.append(idx)

    n_inv = len(inverted_idx)
    if n_inv == 0:
        print(f"  {name}: 0 inverted (skip)")
        return n_total, 0

    print(f"  {name}: {n_inv} inverted of {n_total} ({100*n_inv/n_total:.3f}%)")

    n_fixed = 0
    for idx in tqdm(inverted_idx, desc=f"  fix {name}", file=sys.stderr):
        row = df.loc[idx]
        path = str(row["path"])
        # 1. Rotate image 180° in-place
        img = cv2.imread(path)
        if img is None:
            continue
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(path, rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # 2. Flip coords
        new_row = flip_coords(row)
        df.loc[idx] = new_row
        n_fixed += 1

    df.to_csv(csv_path, index=False)
    print(f"  {name}: {n_fixed} images rotated + CSV updated")
    return n_total, n_fixed


def main():
    total_imgs = 0
    total_fixed = 0
    for ds in DATASETS:
        n, f = fix_dataset(ds)
        total_imgs += n
        total_fixed += f
    print(f"\n=== TOTAL: {total_fixed}/{total_imgs} inverted fixed ({100*total_fixed/max(1,total_imgs):.4f}%) ===")


if __name__ == "__main__":
    main()
