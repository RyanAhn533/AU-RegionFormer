"""
Stage 2: Feature-based Noise Detection
========================================
모델 feature space에서 noise를 탐지.

1. Prototype Distance: 각 샘플 ↔ 클래스 centroid 거리
   → 멀면 noise 가능성 ↑

2. kNN Consistency: k개 이웃의 라벨과 불일치하면 noise 의심

3. 결합: prototype_score + knn_score → feature_noise_score

입력: AU-RegionFormer의 global feature ([B, d_emb])
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import os, sys

sys.path.insert(0, "/home/ajy/AU-RegionFormer/src")


@torch.no_grad()
def extract_features(
    model_path: str = "/home/ajy/AU-RegionFormer/outputs/v2/best.pth",
    train_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train.csv",
    output_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/train_features.npy",
    batch_size: int = 128,
    num_workers: int = 8,
):
    """AU-RegionFormer에서 global feature 추출."""
    from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
    from models.core.fer_model import AUFERModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})

    label2id = build_label_mapping(train_csv)
    region_prefixes = find_region_prefixes(train_csv)

    model = AUFERModel(
        backbone_name=str(mcfg.get("backbone", "mobilevitv2_150")),
        pretrained=False,
        num_au=len(region_prefixes),
        num_classes=len(label2id),
        d_emb=int(mcfg.get("d_emb", 384)),
        n_heads=int(mcfg.get("n_heads", 8)),
        n_fusion_layers=int(mcfg.get("n_fusion_layers", 2)),
        roi_mode=str(mcfg.get("roi_mode", "bilinear")),
        roi_spatial=int(mcfg.get("roi_spatial", 1)),
        dropout=0.0,
        head_dropout=0.0,
        gate_init=float(mcfg.get("gate_init", 0.0)),
        drop_path=0.0,
        img_size=224,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dataset = AUFERDataset(
        train_csv, label2id, region_prefixes,
        img_size=224, mean=model.backbone.norm_mean, std=model.backbone.norm_std,
        is_train=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    all_feats = []
    all_labels = []
    for batch in tqdm(loader, desc="Extracting features"):
        images = batch["image"].to(device, non_blocking=True)
        au_coords = batch["au_coords"].to(device, non_blocking=True)

        # return_features=True → (logits, global_feat)
        _, global_feat = model(images, au_coords, return_features=True)
        all_feats.append(global_feat.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

    features = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)
    print(f"Features saved: {output_path}, shape={features.shape}")
    return features, labels


def compute_prototype_distance(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """각 샘플과 자기 클래스 centroid 간 L2 거리. 멀수록 noise 가능성 ↑."""
    num_classes = labels.max() + 1
    centroids = np.zeros((num_classes, features.shape[1]))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(axis=0)

    distances = np.zeros(len(features))
    for i in range(len(features)):
        distances[i] = np.linalg.norm(features[i] - centroids[labels[i]])

    # Normalize per class (z-score)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 1:
            m, s = distances[mask].mean(), distances[mask].std() + 1e-8
            distances[mask] = (distances[mask] - m) / s

    return distances


def compute_knn_consistency(
    features: np.ndarray, labels: np.ndarray, k: int = 50
) -> np.ndarray:
    """k-NN 이웃의 라벨 불일치율. 높으면 noise 가능성 ↑."""
    print(f"Computing {k}-NN consistency for {len(features):,} samples...")

    # L2 normalize for cosine-like distance
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_norm = features / norms

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(features_norm)
    _, indices = nn.kneighbors(features_norm)

    inconsistency = np.zeros(len(features))
    for i in range(len(features)):
        neighbor_labels = labels[indices[i, 1:]]  # exclude self
        same_label = (neighbor_labels == labels[i]).sum()
        inconsistency[i] = 1.0 - (same_label / k)

    return inconsistency


def compute_feature_noise_scores(
    features_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/train_features.npy",
    labels_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/train_labels.npy",
    quality_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train_final_quality.csv",
    output_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train_full_quality.csv",
    k: int = 50,
):
    """Feature-based scores 계산 + 기존 quality scores와 결합."""

    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"Features: {features.shape}, Labels: {labels.shape}")

    # 1. Prototype distance
    print("\n[1] Computing prototype distances...")
    proto_dist = compute_prototype_distance(features, labels)

    # 2. kNN consistency
    print(f"\n[2] Computing {k}-NN consistency...")
    knn_incon = compute_knn_consistency(features, labels, k=k)

    # 3. Feature noise score: combine prototype + kNN
    # Higher = more likely noise
    proto_score = 1.0 / (1.0 + np.exp(-proto_dist))  # sigmoid normalize
    feature_noise = 0.5 * proto_score + 0.5 * knn_incon

    # 4. Load existing quality scores and merge
    df = pd.read_csv(quality_csv)
    df["proto_distance"] = proto_dist
    df["knn_inconsistency"] = knn_incon
    df["feature_noise_score"] = feature_noise
    df["feature_quality"] = 1.0 - feature_noise

    # 5. Final combined quality: metadata + cleanlab + feature
    # Three independent signals → geometric mean
    df["final_quality"] = np.cbrt(
        df["sample_weight"] *          # Phase 1: metadata
        df["cleanlab_quality"] *       # Phase 2: cleanlab
        df["feature_quality"]          # Stage 2: feature-based
    )

    # 6. Noise tier assignment
    df["noise_tier"] = "clean"
    df.loc[df["final_quality"] < 0.7, "noise_tier"] = "suspect"
    df.loc[df["final_quality"] < 0.5, "noise_tier"] = "noisy"
    df.loc[df["final_quality"] < 0.3, "noise_tier"] = "critical"

    df.to_csv(output_csv, index=False)

    # Report
    print(f"\n{'='*60}")
    print("Full Quality Assessment (3-stage)")
    print(f"{'='*60}")
    print(f"Total: {len(df):,}")
    print(f"\nNoise tiers:")
    for tier in ["clean", "suspect", "noisy", "critical"]:
        n = (df["noise_tier"] == tier).sum()
        print(f"  {tier}: {n:,} ({n/len(df)*100:.1f}%)")

    print(f"\nPer-emotion final quality:")
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        print(f"  {label}: quality={sub['final_quality'].mean():.4f}, "
              f"noisy={(sub['noise_tier'].isin(['noisy','critical'])).sum():,}/{len(sub):,}")

    print(f"\nFeature-based insights:")
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        print(f"  {label}: proto_dist={sub['proto_distance'].mean():.3f}, "
              f"knn_incon={sub['knn_inconsistency'].mean():.3f}")

    print(f"\nSaved: {output_csv}")
    return df


if __name__ == "__main__":
    import pandas as pd
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    if args.extract:
        extract_features()

    compute_feature_noise_scores()
