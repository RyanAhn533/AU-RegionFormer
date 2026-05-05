"""
Apply Label Quality Scores to AU-RegionFormer Data
====================================================
학습된 label quality model로 413K 학습 데이터에 confidence score 부여.

Output:
  - index_train_scored.csv: 원본 + quality_score 컬럼 추가
  - score_distribution.png: 감정별 score 분포 시각화
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from label_quality.train_quality_model import LabelQualityModel


class ScoringDataset(Dataset):
    """AU-RegionFormer 학습 데이터를 scoring하기 위한 dataset."""

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        # AU label → 연세대 감정명 매핑
        self.label_map = {
            "happy": "기쁨", "angry": "분노", "sad": "슬픔", "neutral": "중립",
            # 검증 데이터에 없는 감정 → 가장 가까운 감정으로 간접 점수
            "anxious": "분노", "hurt": "슬픔", "surprised": "기쁨",
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        emotion_kr = self.label_map.get(row["label"], "중립")
        return {
            "image": img,
            "label": row["label"],
            "emotion_kr": emotion_kr,
            "idx": idx,
        }


def apply_scores(
    model_path: str = "/home/ajy/AU-RegionFormer/outputs/label_quality/best.pth",
    train_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train.csv",
    output_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train_scored.csv",
    batch_size: int = 128,
    num_workers: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    emotion2id = ckpt["emotion2id"]

    model = LabelQualityModel(num_emotions=len(emotion2id))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"Loaded model from epoch {ckpt['epoch']}, agreement={ckpt['best_metric']:.4f}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ScoringDataset(train_csv, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    all_scores = []
    with torch.no_grad():
        for batch_items in tqdm(loader, desc="Scoring"):
            images = batch_items["image"].to(device, non_blocking=True)
            emotion_ids = torch.tensor(
                [emotion2id.get(e, 0) for e in batch_items["emotion_kr"]],
                dtype=torch.long, device=device,
            )

            logits = model(images, emotion_ids)
            # P(is_selected=1) = confidence that this is clearly the labeled emotion
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.extend(probs)

    # Save scored CSV
    df = pd.read_csv(train_csv)
    df["quality_score"] = all_scores
    df.to_csv(output_csv, index=False)
    print(f"\nSaved scored CSV: {output_csv}")
    print(f"Score distribution: mean={np.mean(all_scores):.3f}, "
          f"std={np.std(all_scores):.3f}, "
          f"median={np.median(all_scores):.3f}")

    # Per-emotion stats
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        scores = sub["quality_score"]
        print(f"  {label}: mean={scores.mean():.3f} std={scores.std():.3f} "
              f"high(>0.5)={(scores>0.5).sum():,}/{len(sub):,}")

    # Visualization
    _save_score_distribution(df, os.path.dirname(output_csv))


def _save_score_distribution(df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        emotions = sorted(df["label"].unique())

        for i, emo in enumerate(emotions):
            ax = axes[i // 4][i % 4]
            scores = df[df["label"] == emo]["quality_score"]
            ax.hist(scores, bins=50, alpha=0.7, color=f"C{i}", edgecolor="white")
            ax.axvline(scores.mean(), color="red", linestyle="--", label=f"mean={scores.mean():.3f}")
            ax.set_title(f"{emo} (n={len(scores):,})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Quality Score")
            ax.legend(fontsize=8)

        # Hide unused subplot
        if len(emotions) < 8:
            axes[1][3].axis("off")

        plt.suptitle("Label Quality Score Distribution per Emotion", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(f"{out_dir}/quality_score_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir}/quality_score_distribution.png")
    except Exception as e:
        print(f"Viz error: {e}")


if __name__ == "__main__":
    apply_scores()
