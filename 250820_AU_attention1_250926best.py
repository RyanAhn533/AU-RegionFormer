#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from torchvision import transforms as T
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ======================== Config ========================
OUTPUT_DIR   = Path("/home/jy/AIhub/0_result_251205~/epoch300")
TRAIN_CSV    = Path("/home/jy/AIhub/AI_hub_250704/data2/Aihub_data_AU_Croped7_Full_data/index_train.csv")
VAL_CSV      = Path("/home/jy/AIhub/AI_hub_250704/data2/Aihub_data_AU_Croped4/index_val.csv")

# 학습 하이퍼파라미터
SEED                    = 42
EPOCHS                  = 300
BATCH_SIZE              = 32
NUM_WORKERS             = 8
BASE_LR                 = 2e-4
WEIGHT_DECAY            = 0.05
LABEL_SMOOTHING         = 0.05
FREEZE_BACKBONE_EPOCHS  = 5        # 초반 안정화용
GRAD_CLIP_NORM          = 5.0

# 입력 크기
GLOBAL_IMG_SIZE         = 224
PATCH_OUT_SIZE          = 128      # CSV의 patch_out과 일치시키거나, 강제 리사이즈
APPLY_ROTATION          = False    # 좌표가 이미 정렬되어 있다면 False 권장

# 모델 크기/어텐션
D_EMB                   = 512
N_HEAD                  = 8
N_LAYERS                = 2

# ========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 재현성 / 성능 균형
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_region_prefixes(columns: List[str]) -> List[str]:
    # *_wx1 컬럼을 찾아 prefix를 구성
    prefixes = []
    for c in columns:
        if c.endswith("_wx1"):
            pref = c[:-4]  # remove _wx1
            prefixes.append(pref)
    # 정렬: 고정된 순서 보장
    prefixes = sorted(prefixes)
    return prefixes


def timm_norm_from_model(model_name='mobilenetv3_small_100.lamb_in1k'):
    # timm data_config 기반 mean/std
    dummy = timm.create_model(model_name, pretrained=True)
    dc = timm.data.resolve_model_data_config(dummy)
    mean = dc.get('mean', (0.485,0.456,0.406))
    std  = dc.get('std' , (0.229,0.224,0.225))
    return mean, std


class AUCSVSet(Dataset):
    """
    CSV 스키마:
    path,label,rot_deg,work_w,work_h,patch_in,patch_out,<region>_wx1, ... _wy2, ...
    좌표는 work_w/work_h 기준으로 뽑혀있다고 가정.
    """
    def __init__(self,
                 csv_path: Path,
                 label2id: Dict[str,int],
                 region_prefixes: List[str],
                 mean: Tuple[float,...],
                 std:  Tuple[float,...],
                 is_train: bool = True):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.region_prefixes = region_prefixes
        self.is_train = is_train
        self.mean = mean
        self.std = std

        # 전역 변환
        aug = []
        if is_train:
            aug += [
                T.Resize((GLOBAL_IMG_SIZE, GLOBAL_IMG_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            ]
        else:
            aug += [T.Resize((GLOBAL_IMG_SIZE, GLOBAL_IMG_SIZE))]
        aug += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        self.global_tf = T.Compose(aug)

        # 패치 변환: 작은 플립/노이즈 정도
        patch_aug = []
        if is_train:
            patch_aug += [
                T.Resize((PATCH_OUT_SIZE, PATCH_OUT_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
            ]
        else:
            patch_aug += [T.Resize((PATCH_OUT_SIZE, PATCH_OUT_SIZE))]
        patch_aug += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        self.patch_tf = T.Compose(patch_aug)

        # 좌표 키 확인 (wx1/wy1/wx2/wy2)
        self.coord_map = {}
        for pref in self.region_prefixes:
            keys = [f"{pref}_wx1", f"{pref}_wy1", f"{pref}_wx2", f"{pref}_wy2"]
            for k in keys:
                if k not in self.df.columns:
                    raise KeyError(f"CSV에 {k} 컬럼이 없습니다.")
            self.coord_map[pref] = keys

        # 경로 존재 여부 체크(경고만)
        self.missing_paths = 0

    def __len__(self):
        return len(self.df)

    def _crop_box(self, img: Image.Image, box: Tuple[int,int,int,int]) -> Image.Image:
        w, h = img.size
        x1, y1, x2, y2 = box
        # 경계 보정
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(1, min(x2, w))
        y2 = max(1, min(y2, h))
        if x2 <= x1: x2 = min(w, x1+1)
        if y2 <= y1: y2 = min(h, y1+1)
        return img.crop((int(x1), int(y1), int(x2), int(y2)))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = str(row['path'])
        label_name = str(row['label'])
        label_id = self.label2id[label_name]
        rot_deg = int(row['rot_deg'])
        work_w = int(row['work_w']); work_h = int(row['work_h'])

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # 경로 문제 시 dummy
            self.missing_paths += 1
            img = Image.fromarray(np.zeros((work_h, work_w, 3), dtype=np.uint8))

        # 좌표가 work_w/h 기준이므로 먼저 리사이즈
        if img.size != (work_w, work_h):
            img = img.resize((work_w, work_h), resample=Image.BILINEAR)

        # 필요시 회전 보정 (좌표가 회전 반영 안됐을 때만)
        if APPLY_ROTATION and rot_deg in (90, 180, 270):
            img = img.rotate(rot_deg, expand=False)

        # 전역 이미지 변환
        global_tensor = self.global_tf(img)

        # AU 패치들 crop
        patches = []
        au_ids = []
        for i, pref in enumerate(self.region_prefixes):
            wx1, wy1, wx2, wy2 = [int(row[k]) for k in self.coord_map[pref]]
            crop = self._crop_box(img, (wx1, wy1, wx2, wy2))
            patches.append(self.patch_tf(crop))
            au_ids.append(i)

        patches = torch.stack(patches, dim=0)  # [K,3,H,W]
        au_ids = torch.tensor(au_ids, dtype=torch.long)

        return {
            "global": global_tensor,        # [3, H, W]
            "patches": patches,             # [K, 3, Hp, Wp]
            "au_ids": au_ids,               # [K]
            "label": torch.tensor(label_id, dtype=torch.long),
            "path": img_path,
        }


# ===================== Model: AU+Global Fusion =====================

class PatchEncoder(nn.Module):
    def __init__(self, model_name='mobilenetv3_small_100.lamb_in1k', d=512):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.proj = nn.Linear(576, d)

    def forward(self, patches):  # [B*K,3,H,W]
        feat = self.backbone.forward_features(patches)   # [B*K, 576, 7, 7]
        feat = feat.mean([-2,-1])                        # [B*K, 576]
        return self.proj(feat)                           # [B*K, d]


class GlobalEncoder(nn.Module):
    def __init__(self, model_name='mobilenetv3_small_100.lamb_in1k', d=512, gexp=1024):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.tail = nn.Sequential(
            nn.Conv2d(576, gexp, 1, bias=False),
            nn.BatchNorm2d(gexp), nn.SiLU(),
            nn.Conv2d(gexp, gexp, 7, groups=gexp, padding=3, bias=False),
            nn.BatchNorm2d(gexp), nn.SiLU(),
            nn.Conv2d(gexp, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.SiLU(),
        )

    def forward(self, img):      # [B,3,H,W]
        f = self.backbone.forward_features(img)   # [B,576,7,7]
        f = self.tail(f).mean([-2,-1])            # [B,d]
        return f


class FusionBlock(nn.Module):
    def __init__(self, d=512, nhead=8, nlayers=2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, batch_first=True)
        self.self_attn = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.cross_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, cls, g, au_tokens):  # cls:[B,d], g:[B,d], au_tokens:[B,K,d]
        B, K, d = au_tokens.shape
        tokens = torch.cat([cls.unsqueeze(1), g.unsqueeze(1), au_tokens], dim=1)  # [B,K+2,d]
        out = self.self_attn(tokens)
        # cross: Q=[CLS,g], KV=AU
        q = out[:, :2, :]
        k = v = out[:, 2:, :]
        cross, _ = self.cross_attn(q, k, v)
        out[:, :2, :] = out[:, :2, :] + torch.sigmoid(self.gamma) * cross
        return self.norm(out)


class FERHead(nn.Module):
    def __init__(self, d=512, num_classes=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d, num_classes)
        )

    def forward(self, tokens):   # [B, K+2, d]
        cls = tokens[:, 0, :]
        return self.mlp(cls)


class AUFusionModel(nn.Module):
    def __init__(self, d=512, nhead=8, nlayers=2, num_classes=7, K=16):
        super().__init__()
        self.K = K
        self.patch_enc = PatchEncoder(d=d)
        self.global_enc = GlobalEncoder(d=d)
        self.au_embed = nn.Embedding(K, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.fusion = FusionBlock(d=d, nhead=nhead, nlayers=nlayers)
        self.head = FERHead(d=d, num_classes=num_classes)

    def forward(self, global_img, patch_batch, au_ids):
        # global_img: [B,3,H,W], patch_batch: [B,K,3,Hp,Wp], au_ids: [B,K]
        B,K,_,_,_ = patch_batch.shape
        patches = patch_batch.view(B*K, 3, patch_batch.size(-2), patch_batch.size(-1))
        au = self.patch_enc(patches).view(B, K, -1)
        au = au + self.au_embed(au_ids)  # AU id embedding
        g  = self.global_enc(global_img) # [B,d]
        cls = self.cls_token.expand(B, -1, -1).squeeze(1)  # [B,d]
        tokens = self.fusion(cls, g, au) # [B,K+2,d]
        logits = self.head(tokens)
        return logits


# ===================== Utils: Train/Eval =====================

def build_label_mapping(train_csv: Path) -> Dict[str,int]:
    df = pd.read_csv(train_csv)
    labels = sorted(df['label'].unique().tolist())
    label2id = {lb:i for i,lb in enumerate(labels)}
    return label2id


def compute_class_weights(train_csv: Path, label2id: Dict[str,int]) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df['label'].value_counts()
    weights = []
    maxn = counts.max()
    for lb, idx in label2id.items():
        n = counts.get(lb, 1)
        weights.append(maxn / n)
    w = torch.tensor(weights, dtype=torch.float32)
    # normalize to mean=1
    w = w * (len(w) / w.sum())
    return w


def collate_fn(batch):
    global_imgs = torch.stack([b["global"] for b in batch], dim=0)
    patches = torch.stack([b["patches"] for b in batch], dim=0)
    au_ids = torch.stack([b["au_ids"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    paths = [b["path"] for b in batch]
    return {"global":global_imgs, "patches":patches, "au_ids":au_ids, "label":labels, "path":paths}


def make_schedulers(optimizer, total_steps, warmup_ratio=0.05, min_lr=1e-6):
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr/BASE_LR, 0.5*(1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    all_loss = []
    with torch.no_grad():
        for batch in loader:
            xg = batch["global"].to(device, non_blocking=True)
            xp = batch["patches"].to(device, non_blocking=True)
            au = batch["au_ids"].to(device, non_blocking=True)
            y  = batch["label"].to(device, non_blocking=True)

            logits = model(xg, xp, au)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            all_loss.append(loss.item())

            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            true = y.cpu().numpy().tolist()
            preds += pred
            trues += true

    f1 = f1_score(trues, preds, average='macro')
    acc = accuracy_score(trues, preds)
    return np.mean(all_loss), f1, acc, preds, trues


def save_confusion(cm, class_names, out_png: Path):
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(7,6))
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Pred')
        fig.savefig(out_png, bbox_inches='tight', dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] save_confusion failed: {e}")


def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device: {device}")

    # 라벨 매핑 및 클래스 가중치
    label2id = build_label_mapping(TRAIN_CSV)
    id2label = {v:k for k,v in label2id.items()}
    num_classes = len(label2id)
    class_weights = compute_class_weights(TRAIN_CSV, label2id).to(device)
    print(f"[INFO] labels: {label2id}")

    # region prefix 추출
    tmp_df = pd.read_csv(TRAIN_CSV, nrows=2)
    region_prefixes = find_region_prefixes(tmp_df.columns.tolist())
    K = len(region_prefixes)
    print(f"[INFO] AU regions ({K}): {region_prefixes}")

    # timm mean/std
    mean, std = timm_norm_from_model('mobilenetv3_small_100.lamb_in1k')

    # Dataset / Loader
    train_set = AUCSVSet(TRAIN_CSV, label2id, region_prefixes, mean, std, is_train=True)
    val_set   = AUCSVSet(VAL_CSV,   label2id, region_prefixes, mean, std, is_train=False)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )

    # Model
    model = AUFusionModel(d=D_EMB, nhead=N_HEAD, nlayers=N_LAYERS, num_classes=num_classes, K=K)
    model.to(device)
    model = model.train()

    # Optim / Sched
    # 초기에 백본 동결
    def set_backbone_requires_grad(req: bool):
        for p in model.patch_enc.backbone.parameters():
            p.requires_grad = req
        for p in model.global_enc.backbone.parameters():
            p.requires_grad = req

    set_backbone_requires_grad(False)  # freeze
    # trainable params만 옵티마이저에
    def trainable_params():
        return [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = make_schedulers(optimizer, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_f1 = -1.0
    history = []

    step = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        running_acc  = 0.0
        n_samples = 0

        # 특정 epoch 이후 백본 언프리즈 + 옵티마이저 재구성(한 번만)
        if epoch == FREEZE_BACKBONE_EPOCHS+1:
            set_backbone_requires_grad(True)
            optimizer = torch.optim.AdamW(trainable_params(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
            scheduler = make_schedulers(optimizer, total_steps)

        for batch in train_loader:
            xg = batch["global"].to(device, non_blocking=True)
            xp = batch["patches"].to(device, non_blocking=True)
            au = batch["au_ids"].to(device, non_blocking=True)
            y  = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(xg, xp, au)
                # 클래스 가중치 + 라벨 스무딩
                loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=LABEL_SMOOTHING)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            step += 1

            bs = y.size(0)
            running_loss += loss.item() * bs
            pred = logits.argmax(dim=1)
            running_acc += (pred == y).sum().item()
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        train_acc  = running_acc  / max(1, n_samples)

        val_loss, val_f1, val_acc, preds, trues = evaluate(model, val_loader, device)

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        }
        history.append(log)
        print(f"[E{epoch:03d}] train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} f1={val_f1:.4f} acc={val_acc:.4f} | lr={log['lr']:.2e}")

        # save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "label2id": label2id,
            "regions": region_prefixes,
            "config": {
                "D_EMB": D_EMB, "N_HEAD": N_HEAD, "N_LAYERS": N_LAYERS,
                "GLOBAL_IMG_SIZE": GLOBAL_IMG_SIZE, "PATCH_OUT_SIZE": PATCH_OUT_SIZE
            }
        }, OUTPUT_DIR / "last.pth")

        # best by macro-F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "label2id": label2id,
                "regions": region_prefixes,
                "config": {
                    "D_EMB": D_EMB, "N_HEAD": N_HEAD, "N_LAYERS": N_LAYERS,
                    "GLOBAL_IMG_SIZE": GLOBAL_IMG_SIZE, "PATCH_OUT_SIZE": PATCH_OUT_SIZE
                }
            }, OUTPUT_DIR / "best.pth")

            # 분석 산출물
            cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
            save_confusion(cm, [id2label[i] for i in range(num_classes)], OUTPUT_DIR / "confusion_best.png")
            report = classification_report(trues, preds, target_names=[id2label[i] for i in range(num_classes)], digits=4)
            with open(OUTPUT_DIR / "report_best.txt", "w", encoding="utf-8") as f:
                f.write(report)

        # 로그 저장
        with open(OUTPUT_DIR / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[DONE] best macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
