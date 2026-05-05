#!/usr/bin/env python3
"""
v3 Yonsei merge: yonsei_{train,val}_4c.csv (v3 schema) → master_{train,val}_v3.csv
  - image_key, subject_hash from orig_path
  - Yonsei agreement merge from all_photos.csv
  - Subject-wise split (1083 subjects → 80/20)
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired")
SAD_MAP_PATH = Path("/home/ajy/AU-RegionFormer/data/label_quality/sad_to_orig_mapping.csv")
YONSEI_PATH  = Path("/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv")
TRAIN_IN  = ROOT / "csvs_v3" / "yonsei_train_4c.csv"
VAL_IN    = ROOT / "csvs_v3" / "yonsei_val_4c.csv"
TRAIN_OUT = ROOT / "csvs_v3" / "master_train_v3.csv"
VAL_OUT   = ROOT / "csvs_v3" / "master_val_v3.csv"

KO2EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}
KEY_RE = re.compile(r"([0-9a-f]{64}).*?(\d{14}-\d{3}-\d{3})\.jpg", re.IGNORECASE)


def image_key_from_filename(fname):
    m = KEY_RE.search(fname)
    return f"{m.group(1).lower()}__{m.group(2)}" if m else ""


def subject_from_key(k):
    return k.split("__", 1)[0] if "__" in k else ""


def quality_score(is_selected, n_raters):
    if n_raters >= 2 and is_selected == 0: return 1.0
    if n_raters >= 2 and is_selected == 1: return 0.1
    if n_raters == 1 and is_selected == 0: return 0.85
    if n_raters == 1 and is_selected == 1: return 0.5
    return 0.85


def attach_image_key(df, sad_lookup):
    keys, subj = [], []
    n_rec = 0; n_no = 0
    for p, lab in zip(df["orig_path"].astype(str), df["label"].astype(str)):
        fname = p.rsplit("/", 1)[-1]
        if lab == "sad":
            orig = sad_lookup.get(fname, "")
            k = image_key_from_filename(orig) if orig else ""
            if k: n_rec += 1
        else:
            k = image_key_from_filename(fname)
        if not k: n_no += 1
        keys.append(k); subj.append(subject_from_key(k))
    df = df.copy(); df["image_key"] = keys; df["subject_hash"] = subj
    print(f"  sad-recovered: {n_rec}, no-key: {n_no}/{len(df)}")
    return df


def load_yonsei_grp():
    yon = pd.read_csv(YONSEI_PATH)
    yon["label_en"] = yon["emotion"].map(KO2EN)
    yon = yon.dropna(subset=["label_en"]).copy()
    yon["img_key"] = yon["path"].astype(str).apply(
        lambda p: image_key_from_filename(p.rsplit("/", 1)[-1]))
    yon = yon[yon["img_key"] != ""].copy()
    yon["is_selected_img"] = (yon["n_selected"] > 0).astype(int)
    yon["mean_is_selected_img"] = (yon["n_selected"] / yon["n_evals"]).astype(float)
    grp = yon.groupby("img_key", as_index=False).agg(
        is_selected=("is_selected_img", "max"),
        n_raters=("n_evals", "max"),
        mean_is_selected=("mean_is_selected_img", "mean"),
    )
    print(f"  Yonsei rows: {len(grp)}, raters: {dict(grp['n_raters'].value_counts().sort_index())}")
    return grp


def merge_yon(df, grp):
    m = df.merge(grp, left_on="image_key", right_on="img_key", how="left")
    m["is_selected"] = m["is_selected"].fillna(0).astype(int)
    m["n_raters"]    = m["n_raters"].fillna(0).astype(int)
    m["mean_is_selected"] = m["mean_is_selected"].fillna(0.0).astype(float).round(4)
    m["quality_score"] = [quality_score(s, n) for s, n in zip(m["is_selected"], m["n_raters"])]
    if "img_key" in m.columns: m = m.drop(columns=["img_key"])
    n_judged = (m["n_raters"] > 0).sum()
    print(f"  judged: {n_judged}/{len(m)} ({100*n_judged/len(m):.1f}%)")
    return m


def subject_split(tr, vl, frac=0.8, seed=42):
    full = pd.concat([tr.assign(orig_split="train"), vl.assign(orig_split="val")], ignore_index=True)
    subj = sorted(full["subject_hash"].unique())
    np.random.seed(seed)
    np.random.shuffle(subj)
    n = int(len(subj) * frac)
    train_set, val_set = set(subj[:n]), set(subj[n:])
    new_tr = full[full["subject_hash"].isin(train_set)].drop(columns=["orig_split"]).reset_index(drop=True)
    new_vl = full[full["subject_hash"].isin(val_set)].drop(columns=["orig_split"]).reset_index(drop=True)
    overlap = set(new_tr["subject_hash"]) & set(new_vl["subject_hash"])
    print(f"  subjects: total={len(subj)} train={len(train_set)} val={len(val_set)} overlap={len(overlap)}")
    return new_tr, new_vl


def main():
    print("[1/4] Load")
    tr = pd.read_csv(TRAIN_IN); vl = pd.read_csv(VAL_IN)
    print(f"  train={len(tr)} val={len(vl)}")
    print("[2/4] image_key + subject_hash")
    sad_lookup = dict(zip(*[pd.read_csv(SAD_MAP_PATH)[c] for c in ("sad_fname","orig_fname")])) if SAD_MAP_PATH.exists() else {}
    tr = attach_image_key(tr, sad_lookup); vl = attach_image_key(vl, sad_lookup)
    print("[3/4] Yonsei merge")
    grp = load_yonsei_grp()
    tr = merge_yon(tr, grp); vl = merge_yon(vl, grp)
    print("[4/4] Subject-wise split")
    tr2, vl2 = subject_split(tr, vl)
    tr2.to_csv(TRAIN_OUT, index=False); vl2.to_csv(VAL_OUT, index=False)
    print(f"\n  saved {TRAIN_OUT} ({len(tr2)} rows)")
    print(f"  saved {VAL_OUT}   ({len(vl2)} rows)")


if __name__ == "__main__":
    main()
