"""
Exp 003 — Jack 2012 per-region linear probe on Clean subset

각 8 region × {raw, clean} × ConvNeXt (primary) → Jack reversal figure
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

SEED = 42
np.random.seed(SEED)
EMB_DIR = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_003_jack_clean_per_region")
REGIONS = ["eyes_left", "eyes_right", "nose", "mouth",
           "forehead", "chin", "cheek_left", "cheek_right"]


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


def subsample(idx, y_all, n_sub=30000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_idx = y_all[idx]
    per_class = n_sub // len(np.unique(y_idx))
    out = []
    for c in np.unique(y_idx):
        c_idx = idx[y_idx == c]
        if len(c_idx) > per_class:
            c_idx = rng.choice(c_idx, size=per_class, replace=False)
        out.extend(c_idx.tolist())
    rng.shuffle(out)
    return np.array(out)


# ====== Load ======
meta = pd.read_csv(EMB_DIR / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder()
y_all = le.fit_transform(meta["emotion"].values)
is_selected = meta["is_selected"].values

idx_all = np.arange(len(meta))
idx_clean = np.where(is_selected == 0)[0]

results = {}
for region in REGIONS + ["POOLED"]:
    print(f"\n== {region} ==")
    if region == "POOLED":
        parts = []
        for r in REGIONS:
            parts.append(np.asarray(np.load(EMB_DIR / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")))
        X_full = np.concatenate(parts, axis=1)
        del parts
    else:
        X_full = np.asarray(np.load(EMB_DIR / f"{region}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r"))

    results[region] = {}
    for subset_name, subset_idx in [("raw", idx_all), ("clean", idx_clean)]:
        sel = subsample(subset_idx, y_all, 30000)
        X = X_full[sel]
        y = y_all[sel]
        acc, std, f1 = probe(X, y, pca_dim=256)
        results[region][subset_name] = {"acc": acc, "std": std, "f1": f1, "n": int(len(sel))}
        print(f"  {subset_name}: acc={acc*100:.2f}% ± {std*100:.2f}, F1={f1:.3f}")
    del X_full

# ====== Save + Figure ======
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary table
lines = ["# Exp 003 — Jack 2012 per-region (ConvNeXt, clean vs raw)", "",
         "| Region | Raw | Clean | Δ (clean-raw) |",
         "|--------|-----|-------|--------------|"]
for r in REGIONS + ["POOLED"]:
    raw = results[r]["raw"]["acc"] * 100
    clean = results[r]["clean"]["acc"] * 100
    lines.append(f"| {r} | {raw:.2f}% ± {results[r]['raw']['std']*100:.2f} | **{clean:.2f}%** ± {results[r]['clean']['std']*100:.2f} | {clean-raw:+.2f}%p |")

# Jack 2012 gap
def eyes_avg(res, sub):
    return (res["eyes_left"][sub]["acc"] + res["eyes_right"][sub]["acc"]) / 2
mouth_raw = results["mouth"]["raw"]["acc"] * 100
mouth_clean = results["mouth"]["clean"]["acc"] * 100
eyes_raw = eyes_avg(results, "raw") * 100
eyes_clean = eyes_avg(results, "clean") * 100

lines.append("\n## Jack 2012 reversal effect size (Mouth vs Eyes avg)")
lines.append("| Condition | Mouth | Eyes (avg L+R) | Gap (M - E) |")
lines.append("|-----------|-------|--------------|-------------|")
lines.append(f"| Raw | {mouth_raw:.2f}% | {eyes_raw:.2f}% | **+{mouth_raw-eyes_raw:.2f}%p** |")
lines.append(f"| Clean (Yonsei consensus) | {mouth_clean:.2f}% | {eyes_clean:.2f}% | **+{mouth_clean-eyes_clean:.2f}%p** |")

with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines))

# Figure
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(REGIONS) + 1)
width = 0.38
raw_vals = [results[r]["raw"]["acc"] * 100 for r in REGIONS + ["POOLED"]]
clean_vals = [results[r]["clean"]["acc"] * 100 for r in REGIONS + ["POOLED"]]
raw_errs = [results[r]["raw"]["std"] * 100 for r in REGIONS + ["POOLED"]]
clean_errs = [results[r]["clean"]["std"] * 100 for r in REGIONS + ["POOLED"]]

ax.bar(x - width/2, raw_vals, width, yerr=raw_errs, label="Raw (237K)",
       color="steelblue", alpha=0.8, capsize=3)
ax.bar(x + width/2, clean_vals, width, yerr=clean_errs, label="Clean (Yonsei consensus, 213K)",
       color="darkgreen", alpha=0.8, capsize=3)
ax.axhline(25, ls="--", color="red", alpha=0.6, label="Random (25%)")
ax.set_xticks(x)
ax.set_xticklabels(REGIONS + ["POOLED"], rotation=30, ha="right")
ax.set_ylabel("Linear probe accuracy (%)")
ax.set_title(f"Per-region classification accuracy (ConvNeXt)\n"
             f"Mouth-Eyes gap: Raw=+{mouth_raw-eyes_raw:.1f}%p, Clean=+{mouth_clean-eyes_clean:.1f}%p")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(20, 90)
for i, (r, c) in enumerate(zip(raw_vals, clean_vals)):
    ax.text(i - width/2, r + 1, f"{r:.1f}", ha="center", fontsize=7)
    ax.text(i + width/2, c + 1, f"{c:.1f}", ha="center", fontsize=7, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "per_region_raw_vs_clean.png", dpi=140)
plt.close()

print(f"\n[done] {OUT}/summary.md + per_region_raw_vs_clean.png")
