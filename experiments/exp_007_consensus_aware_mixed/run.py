"""
Exp 006 — kNN graph feature smoothing (Phase 3 prototype)

Inductive setting (fair):
  - Train graph: kNN on train set only (no test leakage)
  - Test feature augment: concat[x_test, mean of its k-NN in train]
  - Linear probe on augmented

Compare:
  (a) baseline: LogReg on x
  (b) mean-concat: LogReg on [x, mean_kNN(x)]
  (c) consensus-aware-smooth: weight edges by label agreement (proxy: is_selected)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)
EMB = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_007_consensus_aware_mixed")
REGIONS = ["eyes_left","eyes_right","nose","mouth","forehead","chin","cheek_left","cheek_right"]

K = 30
PCA_DIM = 256


def stratified_sample(idx, y, n=30000, seed=SEED):
    rng = np.random.default_rng(seed)
    classes = np.unique(y[idx])
    per_cls = n // len(classes)
    out = []
    for c in classes:
        c_idx = idx[y[idx] == c]
        if len(c_idx) > per_cls:
            c_idx = rng.choice(c_idx, size=per_cls, replace=False)
        out.extend(c_idx.tolist())
    rng.shuffle(out)
    return np.array(out)


def probe_fold(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
    clf.fit(X_tr, y_tr)
    yhat = clf.predict(X_te)
    return accuracy_score(y_te, yhat), f1_score(y_te, yhat, average="macro")


def knn_smooth(X_base, nbr_idx, is_sel_bool=None):
    """Each row = mean of its k-neighbors in X_base.
    If is_sel_bool given, down-weight neighbors where is_selected=1 (noisy).
    """
    if is_sel_bool is None:
        return X_base[nbr_idx].mean(axis=1)  # (N, D)
    # weights: 1.0 for clean neighbors, 0.3 for rejected
    w = np.where(is_sel_bool[nbr_idx], 0.3, 1.0).astype(np.float32)  # (N, K)
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("nkd,nk->nd", X_base[nbr_idx], w)


# === Load ===
print("[info] load meta + region")
meta = pd.read_csv(EMB / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder()
y_all = le.fit_transform(meta["emotion"].values)
is_selected = meta["is_selected"].values
is_sel_bool = (is_selected == 1)

parts = [np.asarray(np.load(EMB / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")) for r in REGIONS]
region = np.concatenate(parts, axis=1)
del parts

# Use MIXED subset (clean + rejected) — consensus-aware weight should help here
# stratified by emotion (is_selected can be either)
all_idx = np.arange(len(meta))
sel = stratified_sample(all_idx, y_all, n=30000)
print(f"[info] mixed subset — rejected fraction: {is_sel_bool[sel].mean():.3f}")
X = region[sel].astype(np.float32)
y = y_all[sel]
is_sel_sub = is_sel_bool[sel]  # all False here (clean)
print(f"[info] sample: {len(sel)}, dim: {X.shape[1]}")

# Preproc: L2 + PCA + Standardize — same as baseline
X = normalize(X, axis=1)
X = PCA(n_components=PCA_DIM, random_state=SEED).fit_transform(X)
X = StandardScaler().fit_transform(X).astype(np.float32)
print(f"[info] preproc done, dim: {X.shape[1]}")

# === CV ===
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
results = {"baseline": [], "concat_knn_mean": [], "concat_knn_consensus_weighted": []}
f1_results = {k: [] for k in results}

for fold, (tr, te) in enumerate(skf.split(X, y)):
    print(f"\n[fold {fold}] train={len(tr)}, test={len(te)}")
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    is_sel_tr = is_sel_sub[tr]

    # (a) baseline
    acc, f1 = probe_fold(X_tr, y_tr, X_te, y_te)
    results["baseline"].append(acc)
    f1_results["baseline"].append(f1)
    print(f"  baseline: acc={acc*100:.2f}% F1={f1:.3f}")

    # Build kNN index on train only (inductive)
    nn = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=-1).fit(X_tr)
    # For train samples: use K+1, drop self
    tr_nbr = nn.kneighbors(X_tr, n_neighbors=K+1, return_distance=False)[:, 1:]
    # For test: top-K
    te_nbr = nn.kneighbors(X_te, n_neighbors=K, return_distance=False)

    # (b) concat plain kNN mean
    tr_sm = knn_smooth(X_tr, tr_nbr)
    te_sm = knn_smooth(X_tr, te_nbr)
    X_tr_b = np.concatenate([X_tr, tr_sm], axis=1)
    X_te_b = np.concatenate([X_te, te_sm], axis=1)
    acc, f1 = probe_fold(X_tr_b, y_tr, X_te_b, y_te)
    results["concat_knn_mean"].append(acc)
    f1_results["concat_knn_mean"].append(f1)
    print(f"  concat_knn_mean: acc={acc*100:.2f}% F1={f1:.3f}")

    # (c) consensus-weighted kNN (clean subset이라 변이 없을 수 있음)
    # For demo: use raw region similarity × inverse is_selected proxy
    tr_sm_w = knn_smooth(X_tr, tr_nbr, is_sel_bool=is_sel_tr)
    te_sm_w = knn_smooth(X_tr, te_nbr, is_sel_bool=is_sel_tr)
    X_tr_c = np.concatenate([X_tr, tr_sm_w], axis=1)
    X_te_c = np.concatenate([X_te, te_sm_w], axis=1)
    acc, f1 = probe_fold(X_tr_c, y_tr, X_te_c, y_te)
    results["concat_knn_consensus_weighted"].append(acc)
    f1_results["concat_knn_consensus_weighted"].append(f1)
    print(f"  concat_consensus_weighted: acc={acc*100:.2f}% F1={f1:.3f}")

# === Summary ===
summary = {}
for k in results:
    summary[k] = {
        "acc_mean": float(np.mean(results[k])),
        "acc_std": float(np.std(results[k])),
        "f1_mean": float(np.mean(f1_results[k])),
    }
    print(f"\n{k}: acc={summary[k]['acc_mean']*100:.2f}% ± {summary[k]['acc_std']*100:.2f}  F1={summary[k]['f1_mean']:.3f}")

delta = (summary["concat_knn_mean"]["acc_mean"] - summary["baseline"]["acc_mean"]) * 100
print(f"\nΔ concat vs baseline: {delta:+.2f}%p")

with open(OUT / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(OUT / "summary.md", "w") as f:
    f.write("# Exp 006 — kNN Graph Feature Smoothing\n\n")
    f.write("| Variant | Accuracy | Macro F1 |\n|---|---|---|\n")
    for k, v in summary.items():
        f.write(f"| {k} | **{v['acc_mean']*100:.2f}%** ± {v['acc_std']*100:.2f} | {v['f1_mean']:.3f} |\n")
    f.write(f"\n**Δ concat_knn_mean vs baseline: {delta:+.2f}%p**\n")

print(f"\n[done] {OUT}")
