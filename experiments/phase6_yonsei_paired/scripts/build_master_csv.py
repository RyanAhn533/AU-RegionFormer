#!/usr/bin/env python3
"""
Phase 6 — Stage 0: build master_train.csv + master_val.csv.

Input source: v4 schema CSVs (already MediaPipe-extracted)
  /home/ajy/AI_hub_250704/data2/Aihub_data_AU_Croped4/index_train.csv
  /home/ajy/AI_hub_250704/data2/Aihub_data_AU_Croped4/index_val.csv

Pipeline:
  1. Load v4 CSVs (107 cols, multi-landmark per region, 7-class)
  2. Drop anxious/hurt/surprised → 4-class subset (happy/angry/sad/neutral)
  3. Collapse multi-landmarks → 8-region clean schema:
       forehead = mean(forehead_69, forehead_299, forehead_9)
       eyes_left = eyes_159    eyes_right = eyes_386
       cheek_left = cheeks_186 cheek_right = cheeks_410
       nose, mouth, chin = single (already)
  4. Recover image_key for sad (sad_NNNNN.jpg) via sad_to_orig_mapping.csv
  5. Yonsei meta join on image_key (subject_hash + timestamp tail)
  6. Subject-wise re-split (1100 subjects → 80/20) — replaces v4's existing split
  7. Compute quality_score from (is_selected, n_raters)
  8. Write master_{train,val}.csv (8-region clean + Yonsei metadata)

Output cols (per row, 28 total):
  path, label, work_w, work_h,
  forehead_cx, forehead_cy, forehead_wx1, forehead_wy1, forehead_wx2, forehead_wy2,
  eyes_left_*, eyes_right_*, nose_*, cheek_left_*, cheek_right_*, mouth_*, chin_*  (6 each)
  ear_left, ear_right (computed from existing cx/cy if available, else 0),
  subject_hash, image_key,
  is_selected, n_raters, mean_is_selected, quality_score
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ─── Paths ───
V4_TRAIN = Path("/home/ajy/AI_hub_250704/data2/Aihub_data_AU_Croped4/index_train.csv")
V4_VAL   = Path("/home/ajy/AI_hub_250704/data2/Aihub_data_AU_Croped4/index_val.csv")
SAD_MAP  = Path("/home/ajy/AU-RegionFormer/data/label_quality/sad_to_orig_mapping.csv")
YONSEI   = Path("/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv")  # has n_evals + n_selected (true multi-rater signal)

PHASE_ROOT = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired")
CSV_DIR    = PHASE_ROOT / "csvs"
DOC_DIR    = PHASE_ROOT / "docs"
LOG_DIR    = PHASE_ROOT / "logs"
for d in (CSV_DIR, DOC_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Constants ───
KEEP_CLASSES = ("happy", "angry", "sad", "neutral")
KO2EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}

# v4 → 8-region collapse map
# (out_region, [list of v4 prefixes to merge])
REGION_COLLAPSE = [
    ("forehead",    ["forehead_69", "forehead_299", "forehead_9"]),
    ("eyes_left",   ["eyes_159"]),
    ("eyes_right",  ["eyes_386"]),
    ("nose",        ["nose"]),
    ("cheek_left",  ["cheeks_186"]),
    ("cheek_right", ["cheeks_410"]),
    ("mouth",       ["mouth"]),
    ("chin",        ["chin"]),
]

KEY_RE = re.compile(r"([0-9a-f]{64}).*?(\d{14}-\d{3}-\d{3})\.jpg", re.IGNORECASE)


def image_key_from_filename(fname: str) -> str:
    m = KEY_RE.search(fname)
    return f"{m.group(1).lower()}__{m.group(2)}" if m else ""


def subject_from_key(k: str) -> str:
    return k.split("__", 1)[0] if "__" in k else ""


def quality_score(is_selected: int, n_raters: int) -> float:
    """4-tier: high(consensus-agree)=1.0, default=0.85, single-reject=0.5, consensus-reject=0.1."""
    if n_raters >= 2 and is_selected == 0:
        return 1.0
    if n_raters >= 2 and is_selected == 1:
        return 0.1
    if n_raters == 1 and is_selected == 0:
        return 0.85
    if n_raters == 1 and is_selected == 1:
        return 0.5
    return 0.85   # unjudged: default trust


# ─── Step 1: Load + 4-class subset ───
def load_v4(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(KEEP_CLASSES)].copy()
    print(f"  {csv_path.name}: {len(df)} rows after 4-class filter")
    return df


# ─── Step 2: Collapse 8-region schema ───
def collapse_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-landmark columns to 8-region cx/cy + bbox."""
    out = df[["path", "label", "work_w", "work_h"]].copy()
    for out_region, srcs in REGION_COLLAPSE:
        cxs = np.zeros((len(df), len(srcs)), dtype=np.float32)
        cys = np.zeros_like(cxs)
        wx1s = np.full_like(cxs, np.inf)
        wy1s = np.full_like(cxs, np.inf)
        wx2s = np.full_like(cxs, -np.inf)
        wy2s = np.full_like(cxs, -np.inf)
        for i, s in enumerate(srcs):
            cxs[:, i]  = df[f"{s}_cx"].values
            cys[:, i]  = df[f"{s}_cy"].values
            wx1s[:, i] = df[f"{s}_wx1"].values
            wy1s[:, i] = df[f"{s}_wy1"].values
            wx2s[:, i] = df[f"{s}_wx2"].values
            wy2s[:, i] = df[f"{s}_wy2"].values
        out[f"{out_region}_cx"]  = cxs.mean(axis=1).round(2)
        out[f"{out_region}_cy"]  = cys.mean(axis=1).round(2)
        out[f"{out_region}_wx1"] = wx1s.min(axis=1).round(2)   # union bbox
        out[f"{out_region}_wy1"] = wy1s.min(axis=1).round(2)
        out[f"{out_region}_wx2"] = wx2s.max(axis=1).round(2)
        out[f"{out_region}_wy2"] = wy2s.max(axis=1).round(2)
    # ear (left/right) — v4 doesn't have it; placeholder zeros
    out["ear_left"]  = 0.0
    out["ear_right"] = 0.0
    return out


# ─── Step 3: Recover image_key (sad rename handling) ───
def attach_image_key(df: pd.DataFrame, sad_mapping: pd.DataFrame) -> pd.DataFrame:
    """Add image_key + subject_hash columns."""
    sad_lookup = dict(zip(sad_mapping["sad_fname"], sad_mapping["orig_fname"]))

    keys = []
    subj = []
    n_recovered_sad = 0
    n_no_key = 0
    for p, lab in zip(df["path"].astype(str), df["label"].astype(str)):
        fname = p.rsplit("/", 1)[-1]
        if lab == "sad":
            orig = sad_lookup.get(fname, "")
            k = image_key_from_filename(orig) if orig else ""
            if k:
                n_recovered_sad += 1
        else:
            k = image_key_from_filename(fname)
        if not k:
            n_no_key += 1
        keys.append(k)
        subj.append(subject_from_key(k))
    df = df.copy()
    df["image_key"] = keys
    df["subject_hash"] = subj
    print(f"  recovered sad keys: {n_recovered_sad}, no-key total: {n_no_key}")
    return df


# ─── Step 4: Yonsei merge ───
def load_yonsei_image_level() -> pd.DataFrame:
    yon = pd.read_csv(YONSEI)
    yon["label_en"] = yon["emotion"].map(KO2EN)
    yon = yon.dropna(subset=["label_en"]).copy()
    yon["img_key"] = yon["path"].astype(str).apply(
        lambda p: image_key_from_filename(p.rsplit("/", 1)[-1])
    )
    yon = yon[yon["img_key"] != ""].copy()
    # all_photos.csv schema: n_evals = num raters, n_selected = num saying "not this emotion"
    if "n_evals" in yon.columns and "n_selected" in yon.columns:
        # is_selected at image level: any rater clicked → 1
        yon["is_selected_img"] = (yon["n_selected"] > 0).astype(int)
        # mean is_selected at image level: n_selected / n_evals (graded)
        yon["mean_is_selected_img"] = (yon["n_selected"] / yon["n_evals"]).astype(float)
        grp = yon.groupby("img_key", as_index=False).agg(
            is_selected=("is_selected_img", "max"),
            n_raters=("n_evals", "max"),
            mean_is_selected=("mean_is_selected_img", "mean"),
        )
    else:
        # Fallback for embeddings/meta_*.csv (legacy single-rater collapsed)
        grp = yon.groupby("img_key", as_index=False).agg(
            is_selected=("is_selected", "max"),
            n_raters=("is_selected", "size"),
            mean_is_selected=("is_selected", "mean"),
        )
    print(f"  Yonsei image-level rows: {len(grp)}")
    print(f"  rater distribution: {dict(grp['n_raters'].value_counts().sort_index())}")
    return grp


def merge_yonsei(df: pd.DataFrame, yon_grp: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(yon_grp, left_on="image_key", right_on="img_key", how="left")
    merged["is_selected"] = merged["is_selected"].fillna(0).astype(int)
    merged["n_raters"]    = merged["n_raters"].fillna(0).astype(int)
    merged["mean_is_selected"] = merged["mean_is_selected"].fillna(0.0).astype(float).round(4)
    merged["quality_score"] = [
        quality_score(s, n) for s, n in zip(merged["is_selected"], merged["n_raters"])
    ]
    if "img_key" in merged.columns:
        merged = merged.drop(columns=["img_key"])
    n_judged = (merged["n_raters"] > 0).sum()
    print(f"  Yonsei-judged: {n_judged} / {len(merged)} ({100*n_judged/len(merged):.1f}%)")
    return merged


# ─── Step 5: Subject-wise split ───
def subject_wise_split(train_df, val_df, train_frac=0.8, seed=42):
    """Pool train+val, then re-split by subject_hash to ensure disjoint subjects."""
    print(f"\n[5/7] Subject-wise re-split (frac={train_frac}, seed={seed})")
    full = pd.concat([train_df.assign(orig_split="train"),
                      val_df.assign(orig_split="val")], ignore_index=True)
    full = full[full["subject_hash"] != ""].copy()
    rng = np.random.default_rng(seed)
    subjects = full["subject_hash"].unique().tolist()
    rng.shuffle(subjects)
    n_train_subj = int(len(subjects) * train_frac)
    train_subj = set(subjects[:n_train_subj])
    full["split"] = full["subject_hash"].apply(
        lambda h: "train" if h in train_subj else "val"
    )
    print(f"  subjects total={len(subjects)}, train={n_train_subj}, val={len(subjects)-n_train_subj}")
    print(f"  rows: train={(full['split']=='train').sum()}, val={(full['split']=='val').sum()}")
    return full


# ─── Step 6: Save + stats ───
def save_master(full_df: pd.DataFrame):
    print(f"\n[6/7] Writing master CSVs")
    cols = ["path", "label", "work_w", "work_h"]
    for r, _ in REGION_COLLAPSE:
        cols += [f"{r}_cx", f"{r}_cy", f"{r}_wx1", f"{r}_wy1", f"{r}_wx2", f"{r}_wy2"]
    cols += ["ear_left", "ear_right",
             "subject_hash", "image_key",
             "is_selected", "n_raters", "mean_is_selected", "quality_score"]
    train_df = full_df[full_df["split"] == "train"][cols]
    val_df   = full_df[full_df["split"] == "val"][cols]
    out_train = CSV_DIR / "master_train.csv"
    out_val   = CSV_DIR / "master_val.csv"
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    print(f"  → {out_train}  rows={len(train_df)}")
    print(f"  → {out_val}    rows={len(val_df)}")
    return out_train, out_val


def write_stats(full_df: pd.DataFrame):
    stats = {}
    for split in ("train", "val"):
        sub = full_df[full_df["split"] == split]
        cls_stats = {}
        for cls in sorted(KEEP_CLASSES):
            csub = sub[sub["label"] == cls]
            cls_stats[cls] = {
                "n": int(len(csub)),
                "n_judged": int((csub["n_raters"] > 0).sum()),
                "n_consensus_agree":  int(((csub["n_raters"] >= 2) & (csub["is_selected"] == 0)).sum()),
                "n_consensus_reject": int(((csub["n_raters"] >= 2) & (csub["is_selected"] == 1)).sum()),
                "n_single_reject":    int(((csub["n_raters"] == 1) & (csub["is_selected"] == 1)).sum()),
                "n_single_agree":     int(((csub["n_raters"] == 1) & (csub["is_selected"] == 0)).sum()),
                "reject_rate": round(float((csub["is_selected"] == 1).mean()), 4),
                "mean_quality": round(float(csub["quality_score"].mean()), 4),
            }
        stats[split] = {
            "total_n": int(len(sub)),
            "n_subjects": int(sub["subject_hash"].nunique()),
            "per_class": cls_stats,
        }
    stats["train_val_subject_overlap"] = int(len(
        set(full_df[full_df["split"]=="train"]["subject_hash"]) &
        set(full_df[full_df["split"]=="val"]["subject_hash"])
    ))
    out = DOC_DIR / "stage0_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {out}")
    return stats


# ─── Main ───
def main():
    print("[1/7] Loading v4 CSVs (107 cols → 4-class subset)")
    train = load_v4(V4_TRAIN)
    val   = load_v4(V4_VAL)

    print("\n[2/7] Collapsing multi-landmark → 8-region schema")
    train_c = collapse_regions(train)
    val_c   = collapse_regions(val)

    print("\n[3/7] Loading sad mapping")
    sad_map = pd.read_csv(SAD_MAP)
    print(f"  sad mapping rows: {len(sad_map)}")

    print("\n[3/7] Attaching image_key + subject_hash")
    train_c = attach_image_key(train_c, sad_map)
    val_c   = attach_image_key(val_c, sad_map)

    print("\n[4/7] Loading + aggregating Yonsei meta")
    yon_grp = load_yonsei_image_level()
    train_c = merge_yonsei(train_c, yon_grp)
    val_c   = merge_yonsei(val_c, yon_grp)

    full = subject_wise_split(train_c, val_c, train_frac=0.8, seed=42)

    save_master(full)
    print("\n[7/7] Stats")
    write_stats(full)
    print("\nStage 0 DONE.")


if __name__ == "__main__":
    main()
