"""
Exp 005 — Per-demographic fairness analysis

Primary question: 성별/나이별 linear probe accuracy에 유의미한 gap이 있는가?
  - Gender (M/F) × Age (20s/30s/40s/50+) = 8 groups
  - Per-group acc + per-class F1 breakdown
  - Fairness gap metric: max-min accuracy
"""
import json
import re
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

SEED = 42
np.random.seed(SEED)
EMB_DIR = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
OUT = Path("/home/ajy/AU-RegionFormer/experiments/exp_005_demographic_fairness")

REGIONS = ["eyes_left", "eyes_right", "nose", "mouth",
           "forehead", "chin", "cheek_left", "cheek_right"]

GENDER_PAT = re.compile(r'_(│▓|┐⌐)_(\d{2})_')


def parse_path(p):
    m = GENDER_PAT.search(p)
    if m:
        g = 'M' if m.group(1) == '│▓' else 'F'
        a = int(m.group(2))
        return g, a
    return None, None


def age_bin(a):
    if a is None or pd.isna(a):
        return None
    if a <= 25: return "20s"
    if a <= 35: return "30s"
    if a <= 45: return "40s"
    return "50+"


def probe(X, y, pca_dim=256):
    X = X.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if pca_dim and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs, f1s, per_cls = [], [], []
    classes = sorted(np.unique(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        per_cls.append(f1_score(y[te], yhat, average=None, labels=classes, zero_division=0).tolist())
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "per_class_f1": {str(c): float(v) for c, v in zip(classes, np.mean(per_cls, axis=0))},
    }


def subsample(idx, y_all, n_sub=None, seed=SEED):
    """Balanced stratified subsample. n_sub=None → use all per-class min × 4."""
    rng = np.random.default_rng(seed)
    y_idx = y_all[idx]
    classes = np.unique(y_idx)
    per_class_counts = {int(c): (y_idx == c).sum() for c in classes}
    min_per_class = min(per_class_counts.values())
    if n_sub:
        per_class = min(n_sub // len(classes), min_per_class)
    else:
        per_class = min_per_class
    out = []
    for c in classes:
        c_idx = idx[y_idx == c]
        if len(c_idx) > per_class:
            c_idx = rng.choice(c_idx, size=per_class, replace=False)
        out.extend(c_idx.tolist())
    rng.shuffle(out)
    return np.array(out), per_class


# === Load ===
print("[info] load meta + parse demographic")
meta = pd.read_csv(EMB_DIR / "meta_convnext_base.fb_in22k_ft_in1k.csv")
le = LabelEncoder()
y_all = le.fit_transform(meta["emotion"].values)
classes_ko = le.classes_.tolist()

meta["gender"] = meta["path"].apply(lambda p: parse_path(p)[0])
meta["age"] = meta["path"].apply(lambda p: parse_path(p)[1])
meta["age_bin"] = meta["age"].apply(age_bin)
is_selected = meta["is_selected"].values

print(f"[info] demographic coverage: {meta['gender'].notna().sum()}/{len(meta)}")
print("[info] load region POOLED")
parts = [np.asarray(np.load(EMB_DIR / f"{r}_convnext_base.fb_in22k_ft_in1k.npy", mmap_mode="r")) for r in REGIONS]
region = np.concatenate(parts, axis=1)
del parts
print(f"[info] region shape: {region.shape}")

# === Define groups (use clean subset for cleanest fairness signal) ===
clean_mask = (is_selected == 0) & meta["gender"].notna() & meta["age_bin"].notna()

groups = {}
for g in ['M', 'F']:
    for ab in ['20s', '30s', '40s', '50+']:
        m = clean_mask & (meta["gender"].values == g) & (meta["age_bin"].values == ab)
        idx = np.where(m)[0]
        if len(idx) > 2000:  # require minimum size
            groups[f"{g}_{ab}"] = idx

print(f"\n[info] groups: {list(groups.keys())}")
for k, v in groups.items():
    dist = pd.Series(y_all[v]).value_counts().sort_index().tolist()
    print(f"  {k}: n={len(v)}, class dist={dist}")

# Overall baseline on clean (for comparison)
all_clean = np.where(clean_mask)[0]

# === Run per-group probe ===
# Use same-size subsample for fair comparison across groups
# target per-class: smallest group's min per-class
min_per_class_list = []
for idx in groups.values():
    y_idx = y_all[idx]
    min_per_class_list.append(min((y_idx == c).sum() for c in np.unique(y_idx)))
common_per_class = min(min_per_class_list)
print(f"\n[info] common per-class sample size: {common_per_class}")

common_n = common_per_class * 4  # 4 classes
results = {}
for name, idx in groups.items():
    sel, pc = subsample(idx, y_all, n_sub=common_n)
    X = region[sel]
    y = y_all[sel]
    res = probe(X, y, pca_dim=256)
    res["n_used"] = int(len(sel))
    res["per_class_n"] = int(pc)
    results[name] = res
    print(f"  {name} (n={len(sel)}): acc={res['acc_mean']*100:.2f}% ± {res['acc_std']*100:.2f}, F1={res['f1_mean']:.3f}")

# Overall clean baseline (same subsample size for fairness)
sel_all, _ = subsample(all_clean, y_all, n_sub=common_n)
res_all = probe(region[sel_all], y_all[sel_all], pca_dim=256)
results["ALL_CLEAN"] = {**res_all, "n_used": int(len(sel_all)), "per_class_n": int(common_per_class)}

# === Fairness gap ===
accs = {k: v["acc_mean"] for k, v in results.items() if k != "ALL_CLEAN"}
max_acc = max(accs.values())
min_acc = min(accs.values())
max_grp = max(accs, key=accs.get)
min_grp = min(accs, key=accs.get)
fairness_gap = (max_acc - min_acc) * 100

# === Save ===
with open(OUT / "results.json", "w") as f:
    json.dump({"classes_ko": classes_ko, "results": results,
               "fairness_gap_pct": fairness_gap,
               "max_group": max_grp, "min_group": min_grp}, f, indent=2, ensure_ascii=False)

# === Summary md ===
lines = ["# Exp 005 — Per-demographic Fairness Analysis", "",
         f"**Feature**: ConvNeXt POOLED (8192d) + clean subset (is_selected=0)",
         f"**Subsample**: {common_n} per group (per-class {common_per_class}), stratified",
         "",
         "| Group | N | Accuracy | Macro F1 | 기쁨 F1 / 분노 F1 / 슬픔 F1 / 중립 F1 |",
         "|-------|---|----------|----------|-------------------------------------|"]
for name in sorted(results.keys()):
    r = results[name]
    pc = r["per_class_f1"]
    pc_str = " / ".join(f"{pc[str(i)]:.3f}" for i in range(4))
    lines.append(f"| {name} | {r['n_used']} | **{r['acc_mean']*100:.2f}%** ± {r['acc_std']*100:.2f} | {r['f1_mean']:.3f} | {pc_str} |")

lines.append("")
lines.append(f"## Fairness gap: **{fairness_gap:.2f}%p**")
lines.append(f"- Best group: **{max_grp}** ({max_acc*100:.2f}%)")
lines.append(f"- Worst group: **{min_grp}** ({min_acc*100:.2f}%)")
lines.append(f"- ALL_CLEAN baseline: {res_all['acc_mean']*100:.2f}%")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy bar
ax = axes[0]
keys = sorted([k for k in results.keys() if k != "ALL_CLEAN"])
vals = [results[k]["acc_mean"] * 100 for k in keys]
errs = [results[k]["acc_std"] * 100 for k in keys]
colors = ["steelblue" if k.startswith("M") else "indianred" for k in keys]
x = np.arange(len(keys))
ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85, capsize=3)
ax.axhline(results["ALL_CLEAN"]["acc_mean"] * 100, ls="--", color="green",
           label=f"ALL_CLEAN ({results['ALL_CLEAN']['acc_mean']*100:.1f}%)")
ax.axhline(25, ls="--", color="gray", alpha=0.5, label="Random 25%")
ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=0)
ax.set_ylabel("Linear probe accuracy (%)")
ax.set_title(f"Per-demographic fairness (gap = {fairness_gap:.1f}%p)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, (v, e) in enumerate(zip(vals, errs)):
    ax.text(i, v + e + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_ylim(min(vals) - 3, max(vals) + 3)

# Per-class F1 heatmap
ax = axes[1]
f1_matrix = np.zeros((len(keys), 4))
for i, k in enumerate(keys):
    pc = results[k]["per_class_f1"]
    for j in range(4):
        f1_matrix[i, j] = pc[str(j)]
import seaborn as sns
sns.heatmap(f1_matrix, annot=True, fmt=".3f",
            xticklabels=classes_ko, yticklabels=keys,
            cmap="RdYlGn", vmin=0.7, vmax=1.0, ax=ax, cbar_kws={"label": "F1"})
ax.set_title("Per-class F1 per demographic group")

plt.tight_layout()
plt.savefig(OUT / "fairness_gap.png", dpi=140)
plt.close()

with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines))
print(f"\n[done] {OUT}/summary.md, fairness_gap.png")
