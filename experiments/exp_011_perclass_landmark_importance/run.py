"""
Exp 011 — Per-class F1 detail + Landmark feature importance

1) Triplet (exp_009 best 87.56%)의 per-class F1 + confusion matrix
2) 17 landmark feature 각각 Fisher score + per-class F1
3) Landmark single-feature linear probe ranking
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

SEED = 42; np.random.seed(SEED)
EMB = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
LAND = Path("/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv")
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_011_perclass_landmark_importance")
REGIONS = ["eyes_left","eyes_right","nose","mouth","forehead","chin","cheek_left","cheek_right"]
LAND_COLS = ["ear_left","ear_right","ear_avg","mar","mouth_width","brow_height_left","brow_height_right",
             "brow_height_avg","brow_furrow","nose_bridge","cheek_raise_left","cheek_raise_right",
             "cheek_raise_avg","lip_corner_angle","face_aspect_ratio","chin_length","forehead_height"]
K = 30


def probe(X, y, pca_dim=None):
    X = X.astype(np.float32)
    if pca_dim and X.shape[1] > pca_dim:
        X = normalize(X, axis=1)
        X = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs, f1s, per_cls, cms = [], [], [], []
    classes = sorted(np.unique(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        per_cls.append(f1_score(y[te], yhat, average=None, labels=classes, zero_division=0))
        cms.append(confusion_matrix(y[te], yhat, labels=classes))
    return {
        "acc": float(np.mean(accs)), "std": float(np.std(accs)),
        "f1": float(np.mean(f1s)),
        "per_class_f1": {str(c): float(v) for c, v in zip(classes, np.mean(per_cls, axis=0))},
        "cm": (np.mean(cms, axis=0) / np.mean(cms, axis=0).sum(axis=1, keepdims=True)).tolist(),
    }


def fisher_score(X, y):
    """per-feature Fisher ratio (between var / within var)"""
    classes = np.unique(y)
    overall = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in classes:
        mask = y == c
        n_c = mask.sum()
        mu_c = X[mask].mean(axis=0)
        between += n_c * (mu_c - overall) ** 2
        within += ((X[mask] - mu_c) ** 2).sum(axis=0)
    return between / (within + 1e-12)


# === Load ===
meta = pd.read_csv(EMB / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder(); y_all = le.fit_transform(meta["emotion"].values)
classes_ko = le.classes_.tolist()
parts = [np.asarray(np.load(EMB / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")) for r in REGIONS]
region = np.concatenate(parts, axis=1); del parts
land = pd.read_csv(LAND).set_index("path")[LAND_COLS]
land_feats = np.full((len(meta), len(LAND_COLS)), np.nan, dtype=np.float32)
for i, p in enumerate(meta["path"].values):
    if p in land.index:
        land_feats[i] = land.loc[p].values
valid = ~np.isnan(land_feats[:, 0])

# Clean + valid + stratified
clean_valid = np.where((meta["is_selected"].values == 0) & valid)[0]
rng = np.random.default_rng(SEED)
sel = []
for c in range(4):
    ci = clean_valid[y_all[clean_valid] == c]
    sel.extend(rng.choice(ci, size=7500, replace=False).tolist())
sel = np.array(sel)
y = y_all[sel]
print(f"[info] sample: {len(sel)}")

# Preproc region (PCA256) + landmark (standardize)
X_reg = normalize(region[sel].astype(np.float32), axis=1)
X_reg = PCA(256, random_state=SEED).fit_transform(X_reg)
X_reg = StandardScaler().fit_transform(X_reg).astype(np.float32)
X_land = StandardScaler().fit_transform(land_feats[sel].astype(np.float32)).astype(np.float32)

# Region-only kNN graph (Phase 0.3 triplet setting)
# (3-fold per-fold rebuild)

# === 1. Triplet per-class F1 ===
print("\n=== Triplet (Region PCA + Landmark + kNN) — per-class F1 ===")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
triplet_accs, triplet_f1s, per_cls, cms = [], [], [], []
for tr, te in skf.split(X_reg, y):
    nn = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=-1).fit(X_reg[tr])
    tr_nbr = nn.kneighbors(X_reg[tr], n_neighbors=K+1, return_distance=False)[:, 1:]
    te_nbr = nn.kneighbors(X_reg[te], n_neighbors=K, return_distance=False)
    tr_sm = X_reg[tr][tr_nbr].mean(axis=1)
    te_sm = X_reg[tr][te_nbr].mean(axis=1)

    Xtr = np.concatenate([X_reg[tr], X_land[tr], tr_sm], axis=1)
    Xte = np.concatenate([X_reg[te], X_land[te], te_sm], axis=1)
    clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
    clf.fit(Xtr, y[tr]); yhat = clf.predict(Xte)
    triplet_accs.append(accuracy_score(y[te], yhat))
    triplet_f1s.append(f1_score(y[te], yhat, average="macro"))
    per_cls.append(f1_score(y[te], yhat, average=None, labels=sorted(np.unique(y)), zero_division=0))
    cms.append(confusion_matrix(y[te], yhat, labels=sorted(np.unique(y))))

triplet_res = {
    "acc": float(np.mean(triplet_accs)), "std": float(np.std(triplet_accs)),
    "f1": float(np.mean(triplet_f1s)),
    "per_class_f1": {classes_ko[c]: float(v) for c, v in zip(sorted(np.unique(y)), np.mean(per_cls, axis=0))},
    "cm_normalized": (np.mean(cms, axis=0) / np.mean(cms, axis=0).sum(axis=1, keepdims=True)).tolist(),
}
print(f"  acc={triplet_res['acc']*100:.2f}% F1={triplet_res['f1']:.3f}")
for cls, v in triplet_res["per_class_f1"].items():
    print(f"    {cls}: F1={v:.3f}")

# === 2. Landmark feature importance ===
print("\n=== 17 Landmark features — Fisher score + single-feature acc ===")
fisher = fisher_score(X_land, y)

single_probe_results = []
for i, col in enumerate(LAND_COLS):
    r = probe(X_land[:, [i]], y)
    single_probe_results.append({
        "feature": col, "fisher": float(fisher[i]),
        "acc": r["acc"], "f1": r["f1"], "per_class_f1": r["per_class_f1"],
    })

single_probe_results.sort(key=lambda d: -d["acc"])
print(f"{'feature':25s}  {'fisher':>8s}  {'acc':>8s}  {'F1':>8s}")
for d in single_probe_results:
    print(f"  {d['feature']:22s}  {d['fisher']:8.3f}  {d['acc']*100:7.2f}%  {d['f1']:7.3f}")

# === Save + figs ===
with open(OUT / "results.json", "w") as f:
    json.dump({"classes_ko": classes_ko, "triplet": triplet_res,
               "landmark_ranking": single_probe_results}, f, indent=2, ensure_ascii=False)

# Fig 1: Triplet confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(np.array(triplet_res["cm_normalized"]), annot=True, fmt=".3f",
            xticklabels=classes_ko, yticklabels=classes_ko, cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Triplet per-class confusion (acc={triplet_res['acc']*100:.2f}%)")
plt.tight_layout(); plt.savefig(OUT / "triplet_confusion.png", dpi=120); plt.close()

# Fig 2: Landmark ranking bar
fig, ax = plt.subplots(figsize=(9, 7))
names = [d["feature"] for d in single_probe_results]
accs = [d["acc"] * 100 for d in single_probe_results]
fishers = [d["fisher"] for d in single_probe_results]
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, accs, color="steelblue", alpha=0.8)
ax.axvline(25, ls="--", color="red", alpha=0.5, label="Random 25%")
ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Single-feature linear probe acc (%)")
ax.set_title("17 Landmark geometric features — class-discriminative power")
ax.legend()
for i, (a, fi) in enumerate(zip(accs, fishers)):
    ax.text(a + 0.3, i, f"{a:.1f}% (F={fi:.2f})", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "landmark_ranking.png", dpi=130); plt.close()

# Summary md
lines = ["# Exp 011 — Per-class F1 + Landmark importance", "",
         "## 1. Triplet (Region+Landmark+Graph) per-class F1", ""]
lines.append(f"- **Overall**: acc={triplet_res['acc']*100:.2f}% ± {triplet_res['std']*100:.2f}  Macro F1={triplet_res['f1']:.3f}")
lines.append("")
lines.append("| Emotion | F1 |")
lines.append("|---------|-----|")
for cls, v in triplet_res["per_class_f1"].items():
    lines.append(f"| {cls} | **{v:.3f}** |")

lines.append("\n## 2. Landmark single-feature importance (17 features, sorted by acc)")
lines.append("| Rank | Feature | Fisher | Single-probe acc | Macro F1 |")
lines.append("|-----|---------|--------|-----------------|----------|")
for i, d in enumerate(single_probe_results, 1):
    lines.append(f"| {i} | {d['feature']} | {d['fisher']:.3f} | {d['acc']*100:.2f}% | {d['f1']:.3f} |")

with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines))
print(f"\n[done] {OUT}/summary.md")
