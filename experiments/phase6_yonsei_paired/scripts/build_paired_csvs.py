"""
Build Yonsei-paired CSVs by joining our index_train.csv to Yonsei meta
on SHA-256 hash prefix in the filename.

Outputs (under phase6_yonsei_paired/csvs/):
  - train_yonsei_full.csv   : 4-class subset of index_train, Yonsei-matched (any is_selected)
  - train_yonsei_clean.csv  : 4-class subset, is_selected==0 only
  - val_yonsei_full.csv
  - val_yonsei_clean.csv

Plus:
  - join_stats.json — match coverage, per-class counts, class-wise reject rate
"""
import json
import re
from pathlib import Path

import pandas as pd

OUR_TRAIN = "/mnt/hdd/ajy_25/au_csv/index_train.csv"
OUR_VAL = "/mnt/hdd/ajy_25/au_csv/index_val.csv"
YONSEI = "/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings/meta_convnext_base.fb_in22k_ft_in1k.csv"

OUT_DIR = Path(__file__).resolve().parent.parent / "csvs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS = Path(__file__).resolve().parent.parent / "docs" / "join_stats.json"

# Korean → English label mapping
KO2EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}
KEEP_CLASSES = set(KO2EN.values())

# NOTE: the 64-char hex prefix is the SUBJECT id (one per person, ~1100),
# NOT a per-image id. The truly-unique image key is
#     {subject_hash}__{timestamp}-{group}-{seq}
# where the tail looks like '20210130220827-003-001'.
KEY_RE = re.compile(r"([0-9a-f]{64}).*?(\d{14}-\d{3}-\d{3})\.jpg",
                    re.IGNORECASE)


def image_key_from_path(path: str) -> str:
    """Extract the unique-per-image key (subject_hash + capture tail)."""
    base = path.rsplit("/", 1)[-1]
    m = KEY_RE.search(base)
    if not m:
        return ""
    return f"{m.group(1).lower()}__{m.group(2)}"


def build_one(our_csv: str, split: str):
    print(f"\n=== Build for {split} ===")
    print(f"Loading {our_csv}")
    ours = pd.read_csv(our_csv)
    print(f"  rows={len(ours)} cols={len(ours.columns)}")

    # Filter to 4-class
    ours = ours[ours["label"].isin(KEEP_CLASSES)].copy()
    print(f"  4-class subset: {len(ours)}")

    print(f"Loading {YONSEI}")
    yon = pd.read_csv(YONSEI)
    yon["label_en"] = yon["emotion"].map(KO2EN)
    yon = yon.dropna(subset=["label_en"]).copy()
    print(f"  Yonsei rows (4-class only): {len(yon)}")

    # Hash join
    ours["img_key"] = ours["path"].astype(str).apply(image_key_from_path)
    yon["img_key"] = yon["path"].astype(str).apply(image_key_from_path)

    n_no_key_ours = (ours["img_key"] == "").sum()
    n_no_key_yon = (yon["img_key"] == "").sum()
    print(f"  ours: {n_no_key_ours} rows without key; yonsei: {n_no_key_yon}")

    # Drop rows without key
    ours = ours[ours["img_key"] != ""].copy()
    yon = yon[yon["img_key"] != ""].copy()

    # If yonsei has multiple rows per hash, keep most-conservative is_selected
    # (any rejection -> mark as rejected: max is_selected)
    yon_grp = yon.groupby("img_key", as_index=False).agg({
        "is_selected": "max",
        "label_en": "first",
    })

    # Cross-check label consistency between ours and yonsei (should match within 4-class)
    merged = ours.merge(yon_grp[["img_key", "is_selected", "label_en"]],
                         on="img_key", how="left", suffixes=("", "_yon"))
    n_matched = merged["is_selected"].notna().sum()
    print(f"  matched: {n_matched} / {len(ours)} ({100*n_matched/len(ours):.1f}%)")

    label_mismatch = (merged["label"] != merged["label_en"]).fillna(False).sum()
    print(f"  label mismatches: {label_mismatch}")

    full = merged.dropna(subset=["is_selected"]).copy()
    full["is_selected"] = full["is_selected"].astype(int)
    clean = full[full["is_selected"] == 0].copy()

    # Save (drop helper cols `hash`, `label_en` to keep dataset compatible)
    keep_cols = [c for c in ours.columns if c not in ("img_key",)] + ["is_selected"]
    full[keep_cols].to_csv(OUT_DIR / f"{split}_yonsei_full.csv", index=False)
    clean[keep_cols].to_csv(OUT_DIR / f"{split}_yonsei_clean.csv", index=False)
    print(f"  → {split}_yonsei_full.csv  rows={len(full)}")
    print(f"  → {split}_yonsei_clean.csv rows={len(clean)}")

    # Class-wise stats
    cls_stats = {}
    for cls in sorted(KEEP_CLASSES):
        cls_full = full[full["label"] == cls]
        cls_clean = clean[clean["label"] == cls]
        cls_stats[cls] = {
            "full_n": int(len(cls_full)),
            "clean_n": int(len(cls_clean)),
            "rejected_n": int(len(cls_full) - len(cls_clean)),
            "reject_rate": round((len(cls_full) - len(cls_clean)) / max(1, len(cls_full)), 4),
        }
    print(f"  per-class stats: {json.dumps(cls_stats, indent=2)}")

    return {
        "split": split,
        "our_rows_total": int(len(ours)),
        "matched": int(n_matched),
        "label_mismatches": int(label_mismatch),
        "full_n": int(len(full)),
        "clean_n": int(len(clean)),
        "per_class": cls_stats,
    }


def main():
    stats = {
        "train": build_one(OUR_TRAIN, "train"),
        "val": build_one(OUR_VAL, "val"),
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {STATS}")


if __name__ == "__main__":
    main()
