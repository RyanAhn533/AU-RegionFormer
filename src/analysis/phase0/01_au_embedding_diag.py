"""
Phase 0.1: AU Embedding Diagnostic
==================================
Goal: AU embedding이 class-discriminative한가?
Judgment: silhouette >= 0.1 → GO, < 0.05 → backbone 교체 검토

Input:
  - au_embeddings/{region}_{backbone}.npy (237K × d)
  - au_embeddings/meta_{backbone}.csv (emotion label)

Output:
  - outputs/phase0/01_au_embedding_diag/
      metrics.json           — all numeric results
      summary.md             — human readable
      umap_{region}_{bb}.png — per-region UMAP
      umap_pooled_{bb}.png   — pooled UMAP
      pca_variance.png       — PCA explained variance
      centroid_dist.png      — inter-class centroid distance matrix
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import umap


REGIONS = ["eyes_left", "eyes_right", "nose", "mouth",
           "forehead", "chin", "cheek_left", "cheek_right"]
BACKBONES = ["mobilevitv2_150", "convnext_base.fb_in22k_ft_in1k"]

# Reproducibility
SEED = 42
np.random.seed(SEED)


def load_embedding(emb_dir: Path, region: str, backbone: str) -> np.ndarray:
    path = emb_dir / f"{region}_{backbone}.npy"
    return np.load(path, mmap_mode="r")


def load_meta(emb_dir: Path, backbone: str) -> pd.DataFrame:
    path = emb_dir / f"meta_{backbone}.csv"
    return pd.read_csv(path)


def subsample_idx(n_total: int, n_sub: int, labels: np.ndarray = None,
                  stratify: bool = True, seed: int = SEED) -> np.ndarray:
    """Stratified sampling per class if labels given."""
    rng = np.random.default_rng(seed)
    if stratify and labels is not None:
        idx = []
        classes = np.unique(labels)
        per_class = n_sub // len(classes)
        for c in classes:
            c_idx = np.where(labels == c)[0]
            if len(c_idx) > per_class:
                c_idx = rng.choice(c_idx, size=per_class, replace=False)
            idx.append(c_idx)
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        return idx
    else:
        return rng.choice(n_total, size=min(n_sub, n_total), replace=False)


def pca_variance(X: np.ndarray, top_k: int = 20) -> dict:
    pca = PCA(n_components=min(top_k, X.shape[1]), random_state=SEED)
    pca.fit(X)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "top1": float(pca.explained_variance_ratio_[0]),
        "top10_cum": float(np.sum(pca.explained_variance_ratio_[:10])),
    }


def class_centroid_distance(X: np.ndarray, y: np.ndarray) -> dict:
    """Inter-class centroid cosine/euclidean distance matrix."""
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])
    # Cosine sim
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cos_sim = (centroids @ centroids.T) / (norms @ norms.T)
    # Euclidean
    diff = centroids[:, None, :] - centroids[None, :, :]
    eucl = np.linalg.norm(diff, axis=-1)
    # Intra-class std (mean of per-class std norm)
    intra_std = []
    for c in classes:
        Xc = X[y == c]
        intra_std.append(float(Xc.std(axis=0).mean()))
    return {
        "classes": classes.tolist(),
        "cosine_sim_matrix": cos_sim.tolist(),
        "euclidean_matrix": eucl.tolist(),
        "cos_sim_offdiag_mean": float(
            cos_sim[np.triu_indices_from(cos_sim, k=1)].mean()
        ),
        "cos_sim_offdiag_min": float(
            cos_sim[np.triu_indices_from(cos_sim, k=1)].min()
        ),
        "intra_class_std_mean": intra_std,
    }


def compute_silhouette(X: np.ndarray, y: np.ndarray, n_sub: int = 10000,
                       metric: str = "cosine") -> float:
    """Silhouette on subsample (O(n^2) memory)."""
    if len(X) > n_sub:
        idx = subsample_idx(len(X), n_sub, labels=y, stratify=True)
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y
    try:
        return float(silhouette_score(X_s, y_s, metric=metric, random_state=SEED))
    except Exception as e:
        print(f"  silhouette fail: {e}")
        return float("nan")


def run_umap(X: np.ndarray, n_sub: int = 20000, labels: np.ndarray = None) -> tuple:
    """Return (embedding_2d, indices_used)."""
    if len(X) > n_sub:
        idx = subsample_idx(len(X), n_sub, labels=labels, stratify=True)
        X_s = X[idx]
    else:
        idx = np.arange(len(X))
        X_s = X
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                        n_components=2, random_state=SEED, n_jobs=8)
    emb = reducer.fit_transform(X_s)
    return emb, idx


def plot_umap(emb_2d: np.ndarray, labels: np.ndarray, title: str,
              out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    classes = np.unique(labels)
    palette = sns.color_palette("Set1", n_colors=len(classes))
    for c, color in zip(classes, palette):
        mask = labels == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=1, alpha=0.4,
                   color=color, label=str(c))
    ax.set_title(title)
    ax.legend(markerscale=5, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_centroid_matrix(matrix: list, classes: list, title: str,
                         out_path: Path, fmt: str = ".3f"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(matrix), annot=True, fmt=fmt,
                xticklabels=classes, yticklabels=classes,
                cmap="coolwarm", ax=ax, cbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_pca_variance(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, val in results.items():
        ax.plot(val["cumulative"], label=key, alpha=0.7)
    ax.axhline(0.8, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — all regions × backbones")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def judgment(silhouette: float) -> str:
    if silhouette >= 0.1:
        return "GO — Phase 1 진입"
    elif silhouette >= 0.05:
        return "MARGINAL — Phase 0.2 원인 규명 필요"
    else:
        return "NO-GO — Backbone 교체 검토"


def main(args):
    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {"experiment_id": "phase0_01",
                   "date": datetime.now().isoformat(),
                   "seed": SEED,
                   "per_region": {},
                   "pooled": {}}
    pca_results = {}

    for backbone in BACKBONES:
        print(f"\n{'='*60}\nBackbone: {backbone}\n{'='*60}")
        meta = load_meta(emb_dir, backbone)
        le = LabelEncoder()
        y_all = le.fit_transform(meta["emotion"].values)
        classes_ko = le.classes_.tolist()
        print(f"  Samples: {len(meta)}, Classes: {classes_ko}")

        # ==== Per-region analysis ====
        pooled_parts = []
        for region in REGIONS:
            print(f"\n  [{region}]")
            X = np.asarray(load_embedding(emb_dir, region, backbone))
            assert len(X) == len(y_all), f"Shape mismatch: {len(X)} vs {len(y_all)}"

            # PCA
            pca_s = pca_variance(X, top_k=20)
            # Centroid
            cent = class_centroid_distance(X, y_all)
            # Silhouette (subsample 10K)
            sil = compute_silhouette(X, y_all, n_sub=10000, metric="cosine")
            print(f"    silhouette(cos)={sil:.4f}, "
                  f"inter-cos-sim mean={cent['cos_sim_offdiag_mean']:.4f}, "
                  f"PCA top10 cum={pca_s['top10_cum']:.3f}")

            key = f"{region}|{backbone}"
            all_metrics["per_region"][key] = {
                "pca": pca_s,
                "centroid": cent,
                "silhouette_cos": sil,
                "classes_ko": classes_ko,
            }
            pca_results[key] = pca_s

            pooled_parts.append(X)

            # UMAP per-region
            print(f"    UMAP...")
            emb_2d, idx = run_umap(X, n_sub=20000, labels=y_all)
            plot_umap(emb_2d, y_all[idx], f"{region} / {backbone}",
                      out_dir / f"umap_{region}_{backbone}.png")

            # Centroid heatmap per-region
            plot_centroid_matrix(
                cent["cosine_sim_matrix"], classes_ko,
                f"Cosine Sim — {region} / {backbone}",
                out_dir / f"cossim_{region}_{backbone}.png"
            )

        # ==== Pooled (concat 8 regions) ====
        print(f"\n  [POOLED: concat 8 regions]")
        X_pool = np.concatenate(pooled_parts, axis=1)  # (N, 8*d)
        del pooled_parts
        print(f"    shape={X_pool.shape}")

        pca_s = pca_variance(X_pool, top_k=30)
        cent = class_centroid_distance(X_pool, y_all)
        sil = compute_silhouette(X_pool, y_all, n_sub=10000, metric="cosine")
        print(f"    silhouette(cos)={sil:.4f}, "
              f"inter-cos-sim mean={cent['cos_sim_offdiag_mean']:.4f}")

        all_metrics["pooled"][backbone] = {
            "pca": pca_s,
            "centroid": cent,
            "silhouette_cos": sil,
            "classes_ko": classes_ko,
            "dim": int(X_pool.shape[1]),
        }
        pca_results[f"POOLED|{backbone}"] = pca_s

        # Pooled UMAP
        print(f"    UMAP...")
        emb_2d, idx = run_umap(X_pool, n_sub=20000, labels=y_all)
        plot_umap(emb_2d, y_all[idx], f"POOLED 8-region / {backbone}",
                  out_dir / f"umap_pooled_{backbone}.png")

        # Pooled centroid heatmap
        plot_centroid_matrix(
            cent["cosine_sim_matrix"], classes_ko,
            f"Cosine Sim — POOLED / {backbone}",
            out_dir / f"cossim_pooled_{backbone}.png"
        )

        del X_pool

    # ==== PCA global plot ====
    plot_pca_variance(pca_results, out_dir / "pca_variance.png")

    # ==== Save metrics ====
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # ==== Build summary ====
    lines = ["# Phase 0.1 — AU Embedding Diagnostic Summary",
             f"Generated: {all_metrics['date']}", ""]
    lines.append("## Judgment (pooled silhouette)")
    lines.append("| Backbone | Silhouette (cos) | Inter-class cos sim | Judgment |")
    lines.append("|---|---|---|---|")
    for backbone in BACKBONES:
        m = all_metrics["pooled"][backbone]
        lines.append(
            f"| {backbone} | {m['silhouette_cos']:.4f} | "
            f"{m['centroid']['cos_sim_offdiag_mean']:.4f} | "
            f"{judgment(m['silhouette_cos'])} |"
        )

    lines.append("\n## Per-region silhouette")
    lines.append("| Region | " + " | ".join(BACKBONES) + " |")
    lines.append("|---|" + "|".join(["---"] * len(BACKBONES)) + "|")
    for region in REGIONS:
        row = [region]
        for bb in BACKBONES:
            key = f"{region}|{bb}"
            row.append(f"{all_metrics['per_region'][key]['silhouette_cos']:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Inter-class cosine similarity (pooled)")
    lines.append("목표 수치 확인: '0.97-0.99'가 실제로 맞는지")
    for backbone in BACKBONES:
        m = all_metrics["pooled"][backbone]
        lines.append(f"\n### {backbone}")
        lines.append(f"- Mean off-diagonal cos sim: **{m['centroid']['cos_sim_offdiag_mean']:.4f}**")
        lines.append(f"- Min off-diagonal cos sim: **{m['centroid']['cos_sim_offdiag_min']:.4f}**")
        lines.append(f"- Classes: {m['classes_ko']}")

    lines.append("\n## Files")
    lines.append("- `metrics.json` — all numeric results")
    lines.append("- `umap_*.png` — 2D visualization per region/pooled")
    lines.append("- `cossim_*.png` — inter-class cosine similarity heatmap")
    lines.append("- `pca_variance.png` — explained variance curves")

    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}\nDone. Output: {out_dir}\n{'='*60}")
    for backbone in BACKBONES:
        m = all_metrics["pooled"][backbone]
        print(f"  [{backbone}] silhouette={m['silhouette_cos']:.4f} "
              f"→ {judgment(m['silhouette_cos'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir",
                        default="/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings")
    parser.add_argument("--output-dir",
                        default="/home/ajy/AU-RegionFormer/outputs/phase0/01_au_embedding_diag")
    args = parser.parse_args()
    main(args)
