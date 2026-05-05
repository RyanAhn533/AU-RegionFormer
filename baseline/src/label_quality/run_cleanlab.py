"""
Phase 2: cleanlab — AU-RegionFormer 모델 기반 noise 탐지
=========================================================
AU-RegionFormer v2 best 모델로 413K 전체 데이터에 대해
predicted probability를 추출하고, cleanlab으로 label issue 탐지.

Phase 1 (metadata prior)과 교차 검증하여 최종 noise score 생성.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/ajy/AU-RegionFormer/src")

from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores


def extract_predictions(
    model_path: str = "/home/ajy/AU-RegionFormer/outputs/v2/best.pth",
    train_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train.csv",
    output_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/train_pred_probs.npy",
    batch_size: int = 128,
    num_workers: int = 8,
):
    """AU-RegionFormer v2 best로 413K 전체 데이터 predicted probability 추출."""
    from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
    from models.core.fer_model import AUFERModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})

    # Build label mapping
    label2id = build_label_mapping(train_csv)
    region_prefixes = find_region_prefixes(train_csv)
    num_classes = len(label2id)
    num_au = len(region_prefixes)

    print(f"Classes: {label2id}")
    print(f"AU regions: {num_au}")

    # Build model
    model = AUFERModel(
        backbone_name=str(mcfg.get("backbone", "mobilevitv2_150")),
        pretrained=False,
        num_au=num_au,
        num_classes=num_classes,
        d_emb=int(mcfg.get("d_emb", 384)),
        n_heads=int(mcfg.get("n_heads", 8)),
        n_fusion_layers=int(mcfg.get("n_fusion_layers", 2)),
        roi_mode=str(mcfg.get("roi_mode", "bilinear")),
        roi_spatial=int(mcfg.get("roi_spatial", 1)),
        dropout=float(mcfg.get("dropout", 0.1)),
        head_dropout=float(mcfg.get("head_dropout", 0.2)),
        gate_init=float(mcfg.get("gate_init", 0.0)),
        drop_path=float(mcfg.get("drop_path", 0.0)),
        img_size=224,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    mean = model.backbone.norm_mean
    std = model.backbone.norm_std

    # Dataset (no augmentation)
    dataset = AUFERDataset(
        train_csv, label2id, region_prefixes,
        img_size=224, mean=mean, std=std, is_train=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    print(f"Extracting predictions for {len(dataset):,} samples...")

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            images = batch["image"].to(device, non_blocking=True)
            au_coords = batch["au_coords"].to(device, non_blocking=True)
            labels = batch["label"]

            logits = model(images, au_coords)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    pred_probs = np.concatenate(all_probs, axis=0)
    labels_arr = np.array(all_labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, pred_probs)
    np.save(output_path.replace("pred_probs", "labels"), labels_arr)

    print(f"Saved: {output_path} shape={pred_probs.shape}")
    return pred_probs, labels_arr


def run_cleanlab(
    pred_probs_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/train_pred_probs.npy",
    labels_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/train_labels.npy",
    prior_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train_with_prior.csv",
    output_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train_final_quality.csv",
):
    """cleanlab으로 label issue 탐지 + Phase 1 prior와 결합."""

    pred_probs = np.load(pred_probs_path)
    labels = np.load(labels_path)
    print(f"Loaded: {pred_probs.shape}, labels: {labels.shape}")

    # cleanlab label quality scores
    print("Computing cleanlab label quality scores...")
    quality_scores = get_label_quality_scores(labels, pred_probs, method="self_confidence")

    # Find label issues
    print("Finding label issues...")
    issue_mask = find_label_issues(labels, pred_probs, return_indices_ranked_by="self_confidence")

    n_issues = len(issue_mask)
    print(f"Label issues found: {n_issues:,} / {len(labels):,} ({n_issues/len(labels)*100:.1f}%)")

    # Load Phase 1 prior
    df = pd.read_csv(prior_csv)
    df["cleanlab_quality"] = quality_scores
    df["cleanlab_issue"] = False
    df.loc[issue_mask, "cleanlab_issue"] = True

    # Final combined score
    # Phase 1: metadata prior (1 - noise_prior)
    # Phase 2: cleanlab quality score (0-1, higher = cleaner)
    # Combined: geometric mean for balanced fusion
    df["combined_quality"] = np.sqrt(df["sample_weight"] * df["cleanlab_quality"])

    # Save
    df.to_csv(output_csv, index=False)

    # Report
    print(f"\n{'='*60}")
    print("Final Quality Assessment")
    print(f"{'='*60}")
    print(f"Total samples: {len(df):,}")
    print(f"cleanlab issues: {df['cleanlab_issue'].sum():,} ({df['cleanlab_issue'].mean()*100:.1f}%)")
    print(f"\nCombined quality score:")
    print(f"  Mean: {df['combined_quality'].mean():.4f}")
    print(f"  High quality (>0.8): {(df['combined_quality']>0.8).sum():,}")
    print(f"  Medium (0.5-0.8): {((df['combined_quality']>0.5) & (df['combined_quality']<=0.8)).sum():,}")
    print(f"  Low (<0.5): {(df['combined_quality']<0.5).sum():,}")

    print(f"\nPer-emotion:")
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        print(f"  {label}: quality={sub['combined_quality'].mean():.4f}, "
              f"issues={sub['cleanlab_issue'].sum():,}/{len(sub):,} "
              f"({sub['cleanlab_issue'].mean()*100:.1f}%)")

    # Cross-validation: Phase 1 flagged vs Phase 2 flagged
    p1_flag = df["noise_prior"] > 0.15  # high metadata noise
    p2_flag = df["cleanlab_issue"]
    both = p1_flag & p2_flag
    either = p1_flag | p2_flag

    print(f"\nCross-validation:")
    print(f"  Phase 1 only (metadata): {(p1_flag & ~p2_flag).sum():,}")
    print(f"  Phase 2 only (cleanlab): {(~p1_flag & p2_flag).sum():,}")
    print(f"  Both phases flag: {both.sum():,} (high confidence noise)")
    print(f"  Either phase: {either.sum():,}")

    print(f"\nSaved: {output_csv}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="Extract predictions first")
    args = ap.parse_args()

    if args.extract:
        extract_predictions()

    run_cleanlab()
