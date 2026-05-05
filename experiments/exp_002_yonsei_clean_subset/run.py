"""
Exp 002 — Yonsei Clean Subset Linear Probe

4 variants:
  1. Region POOLED ConvNeXt + raw (baseline, from phase0_01c)
  2. Region POOLED ConvNeXt + clean (is_selected=0)
  3. AU 41d + raw (from phase0_03_v2)
  4. AU 41d + clean (is_selected=0)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)

EMB_DIR = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
META_CSV = EMB_DIR / "meta_convnext_base.fb_in22k_ft_in1k.csv"
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_002_yonsei_clean_subset")
OUT.mkdir(exist_ok=True, parents=True)

REGIONS = ["eyes_left", "eyes_right", "nose", "mouth",
           "forehead", "chin", "cheek_left", "cheek_right"]
AU_NAMES = [
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU11","AU12",
    "AU13","AU14","AU15","AU16","AU17","AU18","AU19","AU20","AU22","AU23",
    "AU24","AU25","AU26","AU27","AU32","AU38","AU39",
    "AUL1","AUR1","AUL2","AUR2","AUL4","AUR4","AUL6","AUR6",
    "AUL10","AUR10","AUL12","AUR12","AUL14","AUR14",
]


def balanced_subsample(indices_by_class, n_sub, seed=SEED):
    rng = np.random.default_rng(seed)
    per_class = n_sub // len(indices_by_class)
    out = []
    for c, idx in indices_by_class.items():
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        out.extend(idx.tolist())
    rng.shuffle(out)
    return np.array(out)


def linear_probe(X, y, pca_dim=256, n_folds=3):
    X = X.astype(np.float32)
    # L2 norm
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # PCA
    if pca_dim and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs, f1s, per_cls = [], [], []
    classes = sorted(np.unique(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        per_cls_fold = []
        for c in classes:
            mask = y[te] == c
            if mask.sum() > 0:
                per_cls_fold.append(f1_score(y[te][mask], yhat[mask], labels=[c], average=None)[0] if (yhat[mask] == c).any() else 0.0)
            else:
                per_cls_fold.append(np.nan)
        per_cls.append(per_cls_fold)
    per_cls = np.nanmean(per_cls, axis=0)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "per_class_f1": {str(c): float(v) for c, v in zip(classes, per_cls)},
    }


# ====== Data load ======
print("[info] loading meta + embeddings")
meta = pd.read_csv(META_CSV)
le = LabelEncoder()
y_all = le.fit_transform(meta["emotion"].values)
classes_ko = le.classes_.tolist()

# Region POOLED ConvNeXt (8 regions concat, 8192d)
print("[info] loading region embeddings...")
parts = []
for r in REGIONS:
    parts.append(np.load(EMB_DIR / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r"))
region_pooled = np.concatenate([np.asarray(p) for p in parts], axis=1)
print(f"[info] region_pooled shape: {region_pooled.shape}")

# AU parquet 41d — align to meta index
print("[info] aligning AU parquet to meta order...")
au_df = pd.read_parquet(AU_PARQUET)
# AU parquet has 'path' column matching meta
au_df_indexed = au_df.set_index("path")
# Get feature matrix row-aligned to meta (NaN where missing)
au_features = np.full((len(meta), len(AU_NAMES)), np.nan, dtype=np.float32)
paths = meta["path"].tolist()
for i, p in enumerate(paths):
    if p in au_df_indexed.index:
        au_features[i] = au_df_indexed.loc[p, AU_NAMES].values

valid_au_mask = ~np.isnan(au_features[:, 0])
print(f"[info] AU-feature coverage: {valid_au_mask.sum()}/{len(meta)} ({100*valid_au_mask.mean():.1f}%)")

# ====== Subset indices ======
# all = everyone (region은 237K 전부 있음, AU는 valid_au_mask)
# clean = is_selected==0
# rejected = is_selected==1
is_selected = meta["is_selected"].values

variants = {
    "region_raw":      {"idx": np.arange(len(meta)), "use_au": False},
    "region_clean":    {"idx": np.where(is_selected == 0)[0], "use_au": False},
    "au_raw":          {"idx": np.where(valid_au_mask)[0], "use_au": True},
    "au_clean":        {"idx": np.where(valid_au_mask & (is_selected == 0))[0], "use_au": True},
    "region_rejected": {"idx": np.where(is_selected == 1)[0], "use_au": False},
    "au_rejected":     {"idx": np.where(valid_au_mask & (is_selected == 1))[0], "use_au": True},
}

results = {}
for name, cfg in variants.items():
    idx = cfg["idx"]
    use_au = cfg["use_au"]

    # Stratified subsample (30K for large, or min available for small)
    idx_by_class = {}
    y_idx = y_all[idx]
    for c in np.unique(y_idx):
        idx_by_class[int(c)] = idx[y_idx == c]
    min_per_class = min(len(v) for v in idx_by_class.values())
    n_sub = min(30000, min_per_class * len(idx_by_class))
    sel = balanced_subsample(idx_by_class, n_sub)

    X = au_features[sel] if use_au else region_pooled[sel]
    y = y_all[sel]
    print(f"\n[{name}] n={len(sel)} (per-class min {min_per_class})  dim={X.shape[1]}")

    res = linear_probe(X, y, pca_dim=256 if not use_au else None, n_folds=3)
    res["n_used"] = int(len(sel))
    res["dim"] = int(X.shape[1])
    results[name] = res
    print(f"  acc={res['acc_mean']*100:.2f}% ± {res['acc_std']*100:.2f}  F1={res['f1_mean']:.3f}")
    pc = ", ".join(f"{classes_ko[int(c)]}={v:.3f}" for c, v in res["per_class_f1"].items())
    print(f"  per-class F1: {pc}")

# ====== Save ======
with open(OUT / "results.json", "w") as f:
    json.dump({"classes_ko": classes_ko, "results": results}, f, indent=2, ensure_ascii=False)

# Summary table
lines = ["# Exp 002 — Yonsei Clean Subset Linear Probe", "",
         "| Variant | N | Accuracy | Macro F1 | per-class F1 (기쁨/분노/슬픔/중립) |",
         "|---------|---|----------|----------|---------------------------------|"]
for name, r in results.items():
    pc_str = " / ".join(f"{r['per_class_f1'][str(i)]:.3f}" for i in range(4))
    lines.append(f"| {name} | {r['n_used']} | **{r['acc_mean']*100:.2f}%** ± {r['acc_std']*100:.2f} | {r['f1_mean']:.3f} | {pc_str} |")

# Delta comparison
lines.append("\n## Clean vs Raw Δ")
lines.append("| Feature | Raw acc | Clean acc | Δ (clean-raw) |")
lines.append("|---------|---------|-----------|--------------|")
for ft in ["region", "au"]:
    raw = results[f"{ft}_raw"]["acc_mean"] * 100
    clean = results[f"{ft}_clean"]["acc_mean"] * 100
    lines.append(f"| {ft.upper()} | {raw:.2f}% | {clean:.2f}% | **{clean-raw:+.2f}%p** |")

# Rejected (interpretation signal)
lines.append("\n## Rejected subset (연세대 298명이 '감정 아님' 판정한 샘플만)")
for ft in ["region", "au"]:
    if f"{ft}_rejected" in results:
        r = results[f"{ft}_rejected"]
        lines.append(f"- {ft.upper()} rejected: acc={r['acc_mean']*100:.2f}% (n={r['n_used']})")

with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines))

print(f"\n[done] {OUT}/summary.md")
