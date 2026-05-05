"""
Exp 008 — Landmark geometric + Region joint (clean subset)

17 geometric features (ear, mar, mouth_width, brow_height, ...) vs Region 8192d.
AU was redundant, but geometric could be orthogonal (different modality).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)
EMB = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
LAND = Path("/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv")
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_008_landmark_joint")
REGIONS = ["eyes_left","eyes_right","nose","mouth","forehead","chin","cheek_left","cheek_right"]
LAND_COLS = ["ear_left","ear_right","ear_avg","mar","mouth_width","brow_height_left","brow_height_right",
             "brow_height_avg","brow_furrow","nose_bridge","cheek_raise_left","cheek_raise_right",
             "cheek_raise_avg","lip_corner_angle","face_aspect_ratio","chin_length","forehead_height"]


def stratified(idx, y, n=30000, seed=SEED):
    rng = np.random.default_rng(seed)
    cls = np.unique(y[idx]); per = n // len(cls); out = []
    for c in cls:
        ci = idx[y[idx] == c]
        if len(ci) > per: ci = rng.choice(ci, size=per, replace=False)
        out.extend(ci.tolist())
    rng.shuffle(out); return np.array(out)


def probe(X, y, pca_dim=256):
    X = X.astype(np.float32)
    if pca_dim and X.shape[1] > pca_dim:
        X = normalize(X, axis=1)
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


# Load meta
meta = pd.read_csv(EMB / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder()
y_all = le.fit_transform(meta["emotion"].values)

# Load region
parts = [np.asarray(np.load(EMB / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")) for r in REGIONS]
region = np.concatenate(parts, axis=1)
del parts

# Load landmark, align by path
print("[info] load landmark")
land = pd.read_csv(LAND)
land_idx = land.set_index("path")[LAND_COLS]
land_feats = np.full((len(meta), len(LAND_COLS)), np.nan, dtype=np.float32)
for i, p in enumerate(meta["path"].values):
    if p in land_idx.index:
        land_feats[i] = land_idx.loc[p].values
valid = ~np.isnan(land_feats[:, 0])
print(f"[info] landmark coverage: {valid.sum()}/{len(meta)} ({100*valid.mean():.1f}%)")

# Clean + valid
clean_valid = np.where((meta["is_selected"].values == 0) & valid)[0]
sel = stratified(clean_valid, y_all, n=30000)

X_region = region[sel]
X_land = land_feats[sel]
y = y_all[sel]
print(f"[info] sample: {len(sel)}")

variants = {
    "landmark_only (17d)": (X_land, None),
    "region_only (8192d)": (X_region, 256),
    "region + landmark (8209d)": (np.concatenate([X_region, X_land], axis=1), 256),
    "region_PCA256 + landmark (273d)": (
        np.concatenate([PCA(256, random_state=SEED).fit_transform(normalize(X_region, axis=1)),
                        StandardScaler().fit_transform(X_land)], axis=1), None),
}

results = {}
for name, (X, pca_dim) in variants.items():
    acc, std, f1 = probe(X, y, pca_dim=pca_dim)
    results[name] = {"acc": acc, "std": std, "f1": f1, "dim": int(X.shape[1])}
    print(f"  {name}: acc={acc*100:.2f}% ± {std*100:.2f}, F1={f1:.3f}")

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
