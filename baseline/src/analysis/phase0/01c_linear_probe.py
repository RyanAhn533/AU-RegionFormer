"""
Phase 0.1c — Linear Probe (더 명확한 metric)

Silhouette보다 '납득 가능한' 수치로 재측정:
  - Linear probe accuracy per (backbone × region)
  - Random baseline 비교
  - Confusion matrix
  - K-means ARI
"""
import argparse
import json
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

# Korean font
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

REGIONS = ["eyes_left", "eyes_right", "nose", "mouth",
           "forehead", "chin", "cheek_left", "cheek_right"]
BACKBONES = {
    "MobileViT-v2-150": "mobilevitv2_150",
    "ConvNeXt-base-22k": "convnext_base.fb_in22k_ft_in1k",
}

SEED = 42
np.random.seed(SEED)


def load_embedding(emb_dir: Path, region: str, backbone_suffix: str) -> np.ndarray:
    return np.load(emb_dir / f"{region}_{backbone_suffix}.npy", mmap_mode="r")


def linear_probe(X: np.ndarray, y: np.ndarray, n_sub: int = 30000,
                 n_folds: int = 3, pca_dim: int = 256) -> dict:
    """L2-normalize → PCA → LogReg 3-fold CV.
    N=30K 정도면 충분 statistical power."""
    rng = np.random.default_rng(SEED)

    # Stratified subsample
    if len(X) > n_sub:
        idx = []
        per_class = n_sub // len(np.unique(y))
        for c in np.unique(y):
            c_idx = np.where(y == c)[0]
            c_idx = rng.choice(c_idx, size=min(per_class, len(c_idx)),
                               replace=False)
            idx.append(c_idx)
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X_s = np.asarray(X[idx], dtype=np.float32)
        y_s = y[idx]
    else:
        X_s = np.asarray(X, dtype=np.float32)
        y_s = y

    # L2 normalize (cosine-space equivalent)
    X_s = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-12)

    # PCA 전처리 (고차원 저주 완화)
    if pca_dim and X_s.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=SEED)
        X_s = pca.fit_transform(X_s)
        explained_var = float(pca.explained_variance_ratio_.sum())
    else:
        explained_var = 1.0

    # Standardize for logistic
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s)

    # 3-fold CV linear probe
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs, f1s, cms = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X_s, y_s)):
        clf = LogisticRegression(
            max_iter=500, C=1.0, n_jobs=-1,
            solver="lbfgs", multi_class="multinomial", random_state=SEED,
        )
        clf.fit(X_s[tr], y_s[tr])
        yhat = clf.predict(X_s[te])
        accs.append(accuracy_score(y_s[te], yhat))
        f1s.append(f1_score(y_s[te], yhat, average="macro"))
        cms.append(confusion_matrix(y_s[te], yhat, labels=np.unique(y_s)))

    cm_mean = np.mean(cms, axis=0)
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)

    # K-means ARI (secondary metric)
    km = KMeans(n_clusters=len(np.unique(y_s)), random_state=SEED, n_init=5)
    km_labels = km.fit_predict(X_s)
    ari = float(adjusted_rand_score(y_s, km_labels))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "cm_normalized": cm_norm.tolist(),
        "kmeans_ari": ari,
        "pca_explained_var": explained_var,
        "n_samples_used": int(len(X_s)),
        "pca_dim": pca_dim if pca_dim and X_s.shape[1] >= pca_dim else X_s.shape[1],
    }


def plot_acc_bar(results: dict, classes: list, out_path: Path):
    """Bar chart — clear visual for '납득'"""
    rows = []
    for config, res in results.items():
        rows.append({
            "config": config,
            "acc": res["acc_mean"] * 100,
            "acc_err": res["acc_std"] * 100,
            "f1": res["macro_f1_mean"] * 100,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("acc", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.3)))
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["acc"], xerr=df["acc_err"],
                   color="steelblue", alpha=0.8, capsize=3)
    ax.axvline(25, color="red", linestyle="--", alpha=0.7,
               label="Random baseline (25%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["config"], fontsize=8)
    ax.set_xlabel("Linear probe accuracy (%) — 3-fold CV, 30K samples")
    ax.set_title("Phase 0.1c — AU Embedding Class-Discriminative Power")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for i, (bar, v) in enumerate(zip(bars, df["acc"])):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_confusion(cm: list, classes: list, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt=".2f",
                xticklabels=classes, yticklabels=classes,
                cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main(args):
    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    classes_ko = None

    for bb_name, bb_suffix in BACKBONES.items():
        print(f"\n{'='*60}\n  {bb_name}\n{'='*60}")
        meta = pd.read_csv(emb_dir / f"meta_{bb_suffix}.csv")
        le = LabelEncoder()
        y_all = le.fit_transform(meta["emotion"].values)
        classes_ko = le.classes_.tolist()

        # Per-region probe
        pooled_parts = []
        for region in REGIONS:
            print(f"  [{region}]", end=" ", flush=True)
            X = load_embedding(emb_dir, region, bb_suffix)
            res = linear_probe(X, y_all, n_sub=30000, n_folds=3, pca_dim=256)
            key = f"{bb_name} | {region}"
            all_results[key] = res
            print(f"acc={res['acc_mean']*100:.1f}±{res['acc_std']*100:.1f}% "
                  f"F1={res['macro_f1_mean']:.3f} ARI={res['kmeans_ari']:.3f}")
            pooled_parts.append(np.asarray(X, dtype=np.float32))

        # Pooled
        print(f"  [POOLED]", end=" ", flush=True)
        X_pool = np.concatenate(pooled_parts, axis=1)
        del pooled_parts
        res = linear_probe(X_pool, y_all, n_sub=30000, n_folds=3, pca_dim=256)
        key = f"{bb_name} | POOLED-8region"
        all_results[key] = res
        print(f"acc={res['acc_mean']*100:.1f}±{res['acc_std']*100:.1f}% "
              f"F1={res['macro_f1_mean']:.3f} ARI={res['kmeans_ari']:.3f}")

        # Confusion for pooled
        plot_confusion(
            res["cm_normalized"], classes_ko,
            f"Confusion (normalized) — {bb_name} POOLED, acc={res['acc_mean']*100:.1f}%",
            out_dir / f"cm_pooled_{bb_suffix}.png"
        )

        del X_pool

    # Random baseline (simulated: predict majority class)
    rand_acc = 1.0 / 4  # 4-class uniform
    all_results["[Baseline] Random uniform"] = {
        "acc_mean": rand_acc, "acc_std": 0.0,
        "macro_f1_mean": rand_acc, "macro_f1_std": 0.0,
        "kmeans_ari": 0.0,
    }

    # Save
    with open(out_dir / "linear_probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Bar chart
    plot_acc_bar(all_results, classes_ko, out_dir / "linear_probe_acc.png")

    # Markdown summary
    lines = ["# Phase 0.1c — Linear Probe Results (명확한 수치)",
             "", "**Setup**: L2 norm → PCA(256) → Standardize → LogReg 3-fold CV, 30K stratified samples",
             "",
             "| Config | Linear probe acc | Macro F1 | K-means ARI |",
             "|--------|-----------------|----------|-------------|"]
    # sort by acc desc
    sorted_keys = sorted(
        all_results.keys(),
        key=lambda k: -all_results[k]["acc_mean"]
    )
    for key in sorted_keys:
        r = all_results[key]
        acc_str = f"**{r['acc_mean']*100:.1f}%** ± {r['acc_std']*100:.1f}"
        f1_str = f"{r['macro_f1_mean']:.3f}"
        ari_str = f"{r['kmeans_ari']:.3f}"
        lines.append(f"| {key} | {acc_str} | {f1_str} | {ari_str} |")

    lines += ["",
              "## 해석",
              "- **Random baseline = 25%** (4-class uniform)",
              "- Acc > 35% → embedding에 class 정보 있음",
              "- Acc > 50% → 충분히 class-discriminative",
              "- Acc ~ 25% → 진짜 class 분리 안 됨"]
    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"Saved to: {out_dir}")
    print("Top 5 configs by acc:")
    for key in sorted_keys[:5]:
        r = all_results[key]
        print(f"  {key:50s}  acc={r['acc_mean']*100:.1f}%  F1={r['macro_f1_mean']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir",
                        default="/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
    parser.add_argument("--output-dir",
                        default="/home/ajy/AU-RegionFormer/outputs/phase0/01c_linear_probe")
    args = parser.parse_args()
    main(args)
