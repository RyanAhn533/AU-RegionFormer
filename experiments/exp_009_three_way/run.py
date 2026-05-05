"""
Exp 009 — Region + Landmark + kNN graph smoothing (3-way)

Stack on exp_008 best (86.90%):
  - Region PCA256 (256d)
  - Landmark (17d) standardized
  - kNN mean of PCA-Region (256d) — graph prior
  Total: 529d
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

SEED = 42; np.random.seed(SEED)
EMB = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
LAND = Path("/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv")
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_009_three_way")
REGIONS = ["eyes_left","eyes_right","nose","mouth","forehead","chin","cheek_left","cheek_right"]
LAND_COLS = ["ear_left","ear_right","ear_avg","mar","mouth_width","brow_height_left","brow_height_right",
             "brow_height_avg","brow_furrow","nose_bridge","cheek_raise_left","cheek_raise_right",
             "cheek_raise_avg","lip_corner_angle","face_aspect_ratio","chin_length","forehead_height"]
K = 30

# Load
meta = pd.read_csv(EMB / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder(); y_all = le.fit_transform(meta["emotion"].values)
parts = [np.asarray(np.load(EMB / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")) for r in REGIONS]
region = np.concatenate(parts, axis=1); del parts

land = pd.read_csv(LAND).set_index("path")[LAND_COLS]
land_feats = np.full((len(meta), len(LAND_COLS)), np.nan, dtype=np.float32)
for i, p in enumerate(meta["path"].values):
    if p in land.index:
        land_feats[i] = land.loc[p].values
valid = ~np.isnan(land_feats[:, 0])
clean_valid = np.where((meta["is_selected"].values == 0) & valid)[0]

# stratified
rng = np.random.default_rng(SEED)
per_cls = 30000 // 4
sel = []
for c in range(4):
    ci = clean_valid[y_all[clean_valid] == c]
    sel.extend(rng.choice(ci, size=per_cls, replace=False).tolist())
rng.shuffle(sel); sel = np.array(sel)

y = y_all[sel]
X_reg = region[sel].astype(np.float32)
X_land_raw = land_feats[sel].astype(np.float32)
print(f"[info] sample: {len(sel)}")

# Preproc Region: normalize + PCA256
X_reg = normalize(X_reg, axis=1)
X_reg_pca = PCA(n_components=256, random_state=SEED).fit_transform(X_reg)
X_reg_pca = StandardScaler().fit_transform(X_reg_pca).astype(np.float32)

# Preproc Landmark: standardize
X_land = StandardScaler().fit_transform(X_land_raw).astype(np.float32)

# 3-fold CV
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
results = {
    "region_only_PCA256": [],
    "region+landmark (exp_008 best)": [],
    "region+graph (exp_006 style)": [],
    "region+landmark+graph (3-way)": [],
}
f1_results = {k: [] for k in results}

for fold, (tr, te) in enumerate(skf.split(X_reg_pca, y)):
    print(f"\n[fold {fold}]")

    # kNN in train
    nn = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=-1).fit(X_reg_pca[tr])
    tr_nbr = nn.kneighbors(X_reg_pca[tr], n_neighbors=K+1, return_distance=False)[:, 1:]
    te_nbr = nn.kneighbors(X_reg_pca[te], n_neighbors=K, return_distance=False)
    tr_sm = X_reg_pca[tr][tr_nbr].mean(axis=1)
    te_sm = X_reg_pca[tr][te_nbr].mean(axis=1)

    def fit_predict(Xtr, Xte):
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
        clf.fit(Xtr, y[tr]); yhat = clf.predict(Xte)
        return accuracy_score(y[te], yhat), f1_score(y[te], yhat, average="macro")

    # a) region only
    acc, f1 = fit_predict(X_reg_pca[tr], X_reg_pca[te])
    results["region_only_PCA256"].append(acc); f1_results["region_only_PCA256"].append(f1)

    # b) region + landmark (exp_008)
    Xtr_b = np.concatenate([X_reg_pca[tr], X_land[tr]], axis=1)
    Xte_b = np.concatenate([X_reg_pca[te], X_land[te]], axis=1)
    acc, f1 = fit_predict(Xtr_b, Xte_b)
    results["region+landmark (exp_008 best)"].append(acc); f1_results["region+landmark (exp_008 best)"].append(f1)

    # c) region + graph
    Xtr_c = np.concatenate([X_reg_pca[tr], tr_sm], axis=1)
    Xte_c = np.concatenate([X_reg_pca[te], te_sm], axis=1)
    acc, f1 = fit_predict(Xtr_c, Xte_c)
    results["region+graph (exp_006 style)"].append(acc); f1_results["region+graph (exp_006 style)"].append(f1)

    # d) 3-way
    Xtr_d = np.concatenate([X_reg_pca[tr], X_land[tr], tr_sm], axis=1)
    Xte_d = np.concatenate([X_reg_pca[te], X_land[te], te_sm], axis=1)
    acc, f1 = fit_predict(Xtr_d, Xte_d)
    results["region+landmark+graph (3-way)"].append(acc); f1_results["region+landmark+graph (3-way)"].append(f1)

    for k in results:
        v = results[k][-1]
        print(f"  {k}: {v*100:.2f}%")

print()
summary = {}
for k in results:
    m = np.mean(results[k]); s = np.std(results[k])
    summary[k] = {"acc_mean": float(m), "acc_std": float(s), "f1_mean": float(np.mean(f1_results[k]))}
    print(f"{k}: acc={m*100:.2f}% ± {s*100:.2f}  F1={np.mean(f1_results[k]):.3f}")

with open(OUT / "results.json", "w") as f: json.dump(summary, f, indent=2)
