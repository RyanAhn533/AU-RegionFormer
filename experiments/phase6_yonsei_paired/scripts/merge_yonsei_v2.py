#!/usr/bin/env python3
"""
Merge Yonsei agreement signal into v4-10region preprocessed CSVs.

Input: yonsei_train_4c.csv + yonsei_val_4c.csv from preprocess_full_v2.py.
Output: master_train_v2.csv + master_val_v2.csv (with subject_hash, image_key,
        is_selected, n_raters, mean_is_selected, quality_score)
        + subject-wise re-split.
"""
import re, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired")
SAD_MAP_PATH = Path("/home/ajy/AU-RegionFormer/data/label_quality/sad_to_orig_mapping.csv")
YONSEI_PATH  = Path("/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv")
TRAIN_IN  = ROOT / "csvs_v2" / "yonsei_train_4c.csv"
VAL_IN    = ROOT / "csvs_v2" / "yonsei_val_4c.csv"
TRAIN_OUT = ROOT / "csvs_v2" / "master_train_v2.csv"
VAL_OUT   = ROOT / "csvs_v2" / "master_val_v2.csv"

KO2EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}
KEY_RE = re.compile(r"([0-9a-f]{64}).*?(\d{14}-\d{3}-\d{3})\.jpg", re.IGNORECASE)


def image_key_from_filename(fname: str) -> str:
    m = KEY_RE.search(fname)
    return f"{m.group(1).lower()}__{m.group(2)}" if m else ""


def subject_from_key(k: str) -> str:
    return k.split("__", 1)[0] if "__" in k else ""


def quality_score(is_selected, n_raters):
    if n_raters >= 2 and is_selected == 0: return 1.0
    if n_raters >= 2 and is_selected == 1: return 0.1
    if n_raters == 1 and is_selected == 0: return 0.85
    if n_raters == 1 and is_selected == 1: return 0.5
    return 0.85


def attach_image_key(df, sad_lookup):
    keys, subj = [], []
    n_rec_sad = 0; n_no_key = 0
    for p, lab in zip(df["orig_path"].astype(str), df["label"].astype(str)):
        fname = p.rsplit("/", 1)[-1]
        if lab == "sad":
            orig = sad_lookup.get(fname, "")
            k = image_key_from_filename(orig) if orig else ""
            if k: n_rec_sad += 1
        else:
            k = image_key_from_filename(fname)
        if not k: n_no_key += 1
        keys.append(k); subj.append(subject_from_key(k))
    df = df.copy()
    df["image_key"] = keys
    df["subject_hash"] = subj
    print(f"  recovered sad keys: {n_rec_sad}, no-key: {n_no_key}/{len(df)}")
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
    print(f"  Yonsei image-level rows: {len(grp)}, rater dist: {dict(grp['n_raters'].value_counts().sort_index())}")
    return grp


def merge_yon(df, grp):
    m = df.merge(grp, left_on="image_key", right_on="img_key", how="left")
    m["is_selected"] = m["is_selected"].fillna(0).astype(int)
    m["n_raters"]    = m["n_raters"].fillna(0).astype(int)
    m["mean_is_selected"] = m["mean_is_selected"].fillna(0.0).astype(float).round(4)
    m["quality_score"] = [quality_score(s, n) for s, n in zip(m["is_selected"], m["n_raters"])]
    if "img_key" in m.columns: m = m.drop(columns=["img_key"])
    n_judged = (m["n_raters"] > 0).sum()
    print(f"  Yonsei-judged: {n_judged}/{len(m)} ({100*n_judged/len(m):.1f}%)")
    return m


def subject_wise_split(train_df, val_df, train_frac=0.8, seed=42):
    full = pd.concat([train_df.assign(orig_split="train"),
                      val_df.assign(orig_split="val")], ignore_index=True)
    print(f"  pooled: train={len(train_df)} val={len(val_df)} total={len(full)}")
    subjects = sorted(full["subject_hash"].unique())
    np.random.seed(seed)
    np.random.shuffle(subjects)
    n_train = int(len(subjects) * train_frac)
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:])
    print(f"  subjects: total={len(subjects)} train={len(train_subj)} val={len(val_subj)}")
    new_train = full[full["subject_hash"].isin(train_subj)].drop(columns=["orig_split"]).reset_index(drop=True)
    new_val   = full[full["subject_hash"].isin(val_subj)].drop(columns=["orig_split"]).reset_index(drop=True)
    overlap = set(new_train["subject_hash"]) & set(new_val["subject_hash"])
    print(f"  subject overlap: {len(overlap)} (must be 0)")
    return new_train, new_val


def main():
    print("[1/4] Load preprocessed CSVs")
    tr = pd.read_csv(TRAIN_IN)
    vl = pd.read_csv(VAL_IN)
    print(f"  train={len(tr)} val={len(vl)}")

    print("[2/4] Attach image_key + subject_hash")
    sad_lookup = {}
    if SAD_MAP_PATH.exists():
        sad_df = pd.read_csv(SAD_MAP_PATH)
        sad_lookup = dict(zip(sad_df["sad_fname"], sad_df["orig_fname"]))
    tr = attach_image_key(tr, sad_lookup)
    vl = attach_image_key(vl, sad_lookup)

    print("[3/4] Yonsei merge")
    grp = load_yonsei_grp()
    tr = merge_yon(tr, grp)
    vl = merge_yon(vl, grp)

    print("[4/4] Subject-wise re-split")
    tr2, vl2 = subject_wise_split(tr, vl)
    tr2.to_csv(TRAIN_OUT, index=False)
    vl2.to_csv(VAL_OUT,   index=False)
    print(f"\n  saved {TRAIN_OUT} ({len(tr2)} rows)")
    print(f"  saved {VAL_OUT} ({len(vl2)} rows)")
    print(f"  cols: {len(tr2.columns)} (path, orig_path, label, ..., quality_score)")


if __name__ == "__main__":
    main()
