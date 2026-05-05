"""
Label Quality Classifier
=========================
"이 사진이 진짜 해당 감정인지" 판별하는 binary classifier.

Architecture: MobileNetV3-Small (pretrained) + binary head
- 경량 모델로 빠르게 학습
- 감정별 독립 모델 vs 감정 조건부 단일 모델

Validation: 2명 이상 합의한 사진 4,658장으로 검증
→ 모델 예측과 인간 합의 일치율 90%+ 목표
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import timm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)


class LabelQualityDataset(Dataset):
    """Binary classification: is this photo clearly the labeled emotion?"""

    def __init__(self, csv_path: str, transform=None, emotion_filter: str = None):
        self.df = pd.read_csv(csv_path)
        if emotion_filter:
            self.df = self.df[self.df["emotion"] == emotion_filter].reset_index(drop=True)
        self.transform = transform

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

        return {
            "image": img,
            "label": torch.tensor(int(row["is_selected"]), dtype=torch.long),
            "emotion": row["emotion"],
            "path": row["path"],
        }


class LabelQualityModel(nn.Module):
    """Emotion-conditioned binary classifier."""

    def __init__(self, num_emotions: int = 4, backbone: str = "convnext_base.fb_in22k_ft_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        # Get actual output dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[-1]

        # Emotion embedding (조건부)
        self.emotion_embed = nn.Embedding(num_emotions, 128)

        # Binary head (deeper for stronger backbone)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 128, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, images, emotion_ids):
        feat = self.backbone(images)  # [B, feat_dim]
        emo_feat = self.emotion_embed(emotion_ids)  # [B, 64]
        combined = torch.cat([feat, emo_feat], dim=-1)
        return self.head(combined)


def train_quality_model(
    data_dir: str = "/home/ajy/AU-RegionFormer/data/label_quality",
    output_dir: str = "/home/ajy/AU-RegionFormer/outputs/label_quality",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-5,
    num_workers: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Emotion mapping
    emotion2id = {"기쁨": 0, "분노": 1, "슬픔": 2, "중립": 3}
    id2emotion = {v: k for k, v in emotion2id.items()}

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset
    train_ds = LabelQualityDataset(f"{data_dir}/train.csv", transform=train_transform)
    val_ds = LabelQualityDataset(f"{data_dir}/val_consensus.csv", transform=val_transform)

    # Class imbalance: is_selected=1 is ~12%, use weighted sampler
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    n_pos = train_df["is_selected"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, min(pos_weight, 10.0)], dtype=torch.float32, device=device)
    print(f"Class weights: neg=1.0, pos={class_weights[1]:.1f} (pos={n_pos:,}, neg={n_neg:,})")

    def collate_fn(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "emotion_id": torch.tensor(
                [emotion2id.get(b["emotion"], 0) for b in batch], dtype=torch.long
            ),
        }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # Model
    model = LabelQualityModel(num_emotions=len(emotion2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nTrain: {len(train_ds):,}, Val: {len(val_ds):,}")
    print(f"Device: {device}, Epochs: {epochs}")

    best_metric = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0
        n = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            emotion_ids = batch["emotion_id"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(images, emotion_ids)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * len(labels)
            n += len(labels)

        scheduler.step()
        train_loss = running_loss / max(n, 1)

        # Validate
        val_result = evaluate_quality_model(model, val_loader, device, emotion2id)

        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_acc": round(val_result["accuracy"], 4),
            "val_f1": round(val_result["f1"], 4),
            "val_auc": round(val_result["auc"], 4),
            "val_agreement": round(val_result["human_agreement"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time": round(elapsed, 1),
        }
        history.append(log)

        print(f"[E{epoch:02d}/{epochs}] loss={train_loss:.4f} | "
              f"val_acc={val_result['accuracy']:.4f} f1={val_result['f1']:.4f} "
              f"auc={val_result['auc']:.4f} | "
              f"human_agree={val_result['human_agreement']:.4f} | {elapsed:.1f}s")

        # Save best (by human agreement — 핵심 지표)
        if val_result["human_agreement"] > best_metric:
            best_metric = val_result["human_agreement"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "emotion2id": emotion2id,
            }, f"{output_dir}/best.pth")
            print(f"  ★ New best human_agreement: {best_metric:.4f}")

    # Save history
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final report
    print(f"\n{'='*50}")
    print(f"Best human agreement: {best_metric:.4f}")
    if best_metric >= 0.90:
        print("✓ 90%+ — 전체 데이터 적용 정당화 가능")
    elif best_metric >= 0.80:
        print("△ 80-90% — 보수적 적용 가능 (high confidence만)")
    else:
        print("✗ 80% 미만 — 신뢰도 부족, 모델 폐기 고려")

    return model, history


@torch.no_grad()
def evaluate_quality_model(model, loader, device, emotion2id):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        emotion_ids = batch["emotion_id"].to(device, non_blocking=True)

        logits = model(images, emotion_ids)
        probs = F.softmax(logits, dim=-1)[:, 1]

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # Human agreement: 모델 예측이 인간 합의와 일치하는 비율
    # val_consensus에는 2명+ 전원 동의한 사진만 있으므로, 이 라벨이 ground truth
    human_agreement = acc  # consensus label과의 일치율

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "human_agreement": human_agreement,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


if __name__ == "__main__":
    train_quality_model()
