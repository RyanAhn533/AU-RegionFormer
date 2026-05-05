"""
Exp 004 — AU + Region Joint Linear Probe (clean subset)

Question: Region 8192d가 이미 AU 정보 흡수? 또는 AU가 orthogonal signal 추가?
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
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_004_au_region_joint")

REGIONS = ["eyes_left", "eyes_right", "nose", "mouth",
           "forehead", "chin", "cheek_left", "cheek_right"]
AU_NAMES = [
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU11","AU12",
    "AU13","AU14","AU15","AU16","AU17","AU18","AU19","AU20","AU22","AU23",
    "AU24","AU25","AU26","AU27","AU32","AU38","AU39",
    "AUL1","AUR1","AUL2","AUR2","AUL4","AUR4","AUL6","AUR6",
    "AUL10","AUR10","AUL12","AUR12","AUL14","AUR14",
]


def subsample(idx, y_all, n=30000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_idx = y_all[idx]
    per_class = n // len(np.unique(y_idx))
    out = []
    for c in np.unique(y_idx):
        c_idx = idx[y_idx == c]
        if len(c_idx) > per_class:
            c_idx = rng.choice(c_idx, size=per_class, replace=False)
        out.extend(c_idx.tolist())
    rng.shuffle(out)
    return np.array(out)


def probe(X, y, pca_dim=256):
    X = X.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if pca_dim and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


# === Load ===
meta = pd.read_csv(EMB_DIR / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder()
y_all = le.fit_transform(meta["emotion"].values)
is_selected = meta["is_selected"].values

print("[info] load region embeddings")
parts = [np.asarray(np.load(EMB_DIR / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")) for r in REGIONS]
region = np.concatenate(parts, axis=1)
del parts
print(f"[info] region shape: {region.shape}")

print("[info] align AU parquet")
au_df = pd.read_parquet(AU_PARQUET).set_index("path")
au_feats = np.full((len(meta), len(AU_NAMES)), np.nan, dtype=np.float32)
for i, p in enumerate(meta["path"].values):
    if p in au_df.index:
        au_feats[i] = au_df.loc[p, AU_NAMES].values
valid = ~np.isnan(au_feats[:, 0])
print(f"[info] AU coverage: {valid.sum()}/{len(meta)}")

# Use clean + valid-AU samples only (fair comparison)
idx_clean_valid = np.where(valid & (is_selected == 0))[0]
print(f"[info] clean + valid-AU: {len(idx_clean_valid)}")
sel = subsample(idx_clean_valid, y_all, n=30000)

# === 3 variants ===
variants = {
    "region_only":   region[sel],                            # 8192d
    "au_only":       au_feats[sel],                          # 41d
    "region+au":     np.concatenate([region[sel], au_feats[sel]], axis=1),  # 8233d
}
y = y_all[sel]

results = {}
for name, X in variants.items():
    # For small dim, no PCA
    pca_dim = 256 if X.shape[1] > 256 else None
    acc, std, f1 = probe(X, y, pca_dim=pca_dim)
    results[name] = {"acc": acc, "std": std, "f1": f1, "dim": X.shape[1]}
    print(f"  {name}: dim={X.shape[1]}, acc={acc*100:.2f}% ± {std*100:.2f}, F1={f1:.3f}")

# Delta summary
lines = ["# Exp 004 — AU + Region Joint (clean subset)", "",
         f"n = {len(sel)}, fold=3, same seed, clean (is_selected=0) + valid AU subset", "",
         "| Variant | Dim | Accuracy | Macro F1 |",
         "|---------|-----|----------|----------|"]
for name, r in results.items():
    lines.append(f"| {name} | {r['dim']} | **{r['acc']*100:.2f}%** ± {r['std']*100:.2f} | {r['f1']:.3f} |")

delta_ra = (results["region+au"]["acc"] - results["region_only"]["acc"]) * 100
delta_au_alone = (results["au_only"]["acc"] - 0.25) * 100  # vs random

lines.append(f"\n## Δ Joint vs Region only: **{delta_ra:+.2f}%p**")
if delta_ra > 0.5:
    lines.append("→ **AU가 region과 orthogonal signal 추가** (method novelty candidate)")
elif delta_ra > 0:
    lines.append("→ marginal 증가 — AU 정보가 대부분 region에 흡수됨")
else:
    lines.append("→ AU 추가가 도움 안 됨 — region이 AU signal 완전 흡수")

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines))
print(f"\n[done] {OUT}/summary.md")
