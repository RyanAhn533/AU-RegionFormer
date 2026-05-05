"""
Noise-Aware Trainer for AU-RegionFormer
========================================
기존 trainer.py에서 확장:
  1. Soft label 학습 (KL divergence loss)
  2. Sample-wise weight 적용
  3. Quality score 기반 curriculum learning

입력: index_train_full_quality.csv
  - soft_{emotion} columns → soft label
  - final_quality → sample weight
  - noise_tier → curriculum stage
"""

import time
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import build_label_mapping, find_region_prefixes, collate_fn
from models.core.fer_model import AUFERModel
from training.losses import FocalLoss, compute_class_weights
from training.scheduler import build_scheduler
from training.evaluator import evaluate, save_confusion_matrix, save_report
from utils.seed import set_seed, ensure_dir
from utils.logging import setup_logger, MetricLogger
from utils.checkpoint import save_checkpoint, load_checkpoint

from data.noise_aware_dataset import NoiseAwareDataset

try:
    from timm.utils import ModelEmaV3
    HAS_EMA = True
except ImportError:
    HAS_EMA = False

_torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
if _torch_version >= (2, 1):
    def _make_scaler(enabled):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    def _autocast(device_type, enabled):
        return torch.amp.autocast(device_type, enabled=enabled)
else:
    def _make_scaler(enabled):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    def _autocast(device_type, enabled):
        return torch.cuda.amp.autocast(enabled=enabled)


def soft_label_loss(logits, soft_targets, sample_weights=None, temperature=1.0):
    """
    KL divergence loss with soft targets and per-sample weights.

    Args:
        logits: [B, C]
        soft_targets: [B, C] probability distribution
        sample_weights: [B] per-sample importance weight
        temperature: softening temperature
    """
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    # KL(soft || pred) = sum(soft * log(soft / pred))
    loss_per_sample = F.kl_div(log_probs, soft_targets, reduction="none").sum(dim=-1)  # [B]

    if sample_weights is not None:
        loss_per_sample = loss_per_sample * sample_weights

    return loss_per_sample.mean() * (temperature ** 2)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_noise_aware(config_path: str, resume: str = None):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    train_csv = str(cfg["paths"]["train_csv"])  # full_quality csv
    val_csv = str(cfg["paths"]["val_csv"])
    output_dir = Path(cfg["paths"]["output_dir"])
    ensure_dir(output_dir)

    logger = setup_logger("train", str(output_dir))
    metric_log = MetricLogger(output_dir / "history.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Labels & Regions (from val_csv which has original format)
    label2id = build_label_mapping(val_csv)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    region_prefixes = find_region_prefixes(val_csv)
    num_au = len(region_prefixes)

    logger.info(f"Classes ({num_classes}): {label2id}")
    logger.info(f"AU regions ({num_au}): {region_prefixes}")

    # Model
    mcfg = cfg["model"]
    model = AUFERModel(
        backbone_name=str(mcfg.get("backbone", "mobilevitv2_150")),
        pretrained=bool(mcfg.get("pretrained", True)),
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
        img_size=int(cfg["augmentation"].get("global_img_size", 224)),
        use_au_pool=bool(mcfg.get("use_au_pool", False)),
    ).to(device)

    mean = model.backbone.norm_mean
    std = model.backbone.norm_std
    img_size = int(cfg["augmentation"].get("global_img_size", 224))

    # Noise-aware dataset (soft labels + sample weights)
    emotion_order = sorted(label2id.keys())
    train_set = NoiseAwareDataset(
        train_csv, label2id, region_prefixes, emotion_order,
        img_size=img_size, mean=mean, std=std, is_train=True,
    )

    from data.dataset import AUFERDataset
    val_set = AUFERDataset(
        val_csv, label2id, region_prefixes,
        img_size=img_size, mean=mean, std=std, is_train=False,
    )

    tcfg = cfg["training"]
    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg["num_workers"])

    use_ema = bool(tcfg.get("ema", False)) and HAS_EMA
    ema_model = None
    ema_decay = float(tcfg.get("ema_decay", 0.9998))
    if use_ema:
        ema_model = ModelEmaV3(model, decay=ema_decay)

    def noise_collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "au_coords": torch.stack([b["au_coords"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "soft_label": torch.stack([b["soft_label"] for b in batch]),
            "sample_weight": torch.stack([b["sample_weight"] for b in batch]),
            "path": [b["path"] for b in batch],
        }

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True, collate_fn=noise_collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # Training params
    epochs = int(tcfg["epochs"])
    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 5))
    base_lr = float(tcfg["base_lr"])
    backbone_lr_scale = float(tcfg.get("backbone_lr_scale", 0.1))
    weight_decay = float(tcfg.get("weight_decay", 0.05))
    grad_clip = float(tcfg.get("grad_clip_norm", 5.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.05))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    early_stop_patience = int(tcfg.get("early_stop_patience", 0))
    soft_label_weight = float(tcfg.get("soft_label_weight", 0.5))
    hard_label_weight = float(tcfg.get("hard_label_weight", 0.5))

    # Loss
    alpha = compute_class_weights(val_csv, label2id).to(device)
    hard_criterion = FocalLoss(
        gamma=float(tcfg.get("focal_gamma", 2.0)),
        alpha=alpha,
    )

    logger.info(f"Soft label weight: {soft_label_weight}, Hard label weight: {hard_label_weight}")
    logger.info(f"Train: {len(train_set):,}, Val: {len(val_set):,}")

    # Phase 1: freeze backbone
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr, backbone_lr_scale),
        weight_decay=weight_decay,
    )
    scheduler_mode = str(tcfg.get("scheduler", "cosine_warmup"))
    phase1_steps = len(train_loader) * freeze_epochs
    scheduler = build_scheduler(optimizer, phase1_steps, warmup_ratio=warmup_ratio, mode=scheduler_mode)
    scaler = _make_scaler(enabled=use_amp)

    best_f1 = -1.0
    start_epoch = 1
    no_improve = 0

    if resume:
        start_epoch, best_f1 = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()

        if epoch == freeze_epochs + 1:
            logger.info("=== Unfreezing backbone ===")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.get_param_groups(base_lr, backbone_lr_scale),
                weight_decay=weight_decay,
            )
            remaining = len(train_loader) * (epochs - freeze_epochs)
            restart_epochs = int(tcfg.get("restart_epochs", 0))
            restart_period = len(train_loader) * restart_epochs if restart_epochs > 0 else 0
            scheduler = build_scheduler(optimizer, remaining, warmup_ratio=warmup_ratio,
                                        mode=scheduler_mode, restart_period=restart_period)
            scaler = _make_scaler(enabled=use_amp)
            if use_ema:
                ema_model = ModelEmaV3(model, decay=ema_decay)

        running_loss = 0.0
        running_correct = 0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            au_coords = batch["au_coords"].to(device, non_blocking=True)
            hard_labels = batch["label"].to(device, non_blocking=True)
            soft_labels = batch["soft_label"].to(device, non_blocking=True)
            weights = batch["sample_weight"].to(device, non_blocking=True)

            with _autocast("cuda", enabled=use_amp):
                logits = model(images, au_coords)

                # Combined loss: hard focal + soft KL
                loss_hard = hard_criterion(logits, hard_labels)
                loss_soft = soft_label_loss(logits, soft_labels, sample_weights=weights)
                loss = hard_label_weight * loss_hard + soft_label_weight * loss_soft

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if ema_model is not None:
                ema_model.update(model)

            bs = hard_labels.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(1) == hard_labels).sum().item()
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        train_acc = running_correct / max(1, n_samples)

        # Validate
        eval_model = ema_model.module if ema_model is not None else model
        val_result = evaluate(eval_model, val_loader, device)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_result["loss"], 4),
            "val_f1": round(val_result["f1_macro"], 4),
            "val_acc": round(val_result["accuracy"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": round(time.time() - t0, 1),
        }
        metric_log.log(log)

        logger.info(
            f"[E{epoch:03d}/{epochs}] loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val f1={val_result['f1_macro']:.4f} acc={val_result['accuracy']:.4f} | "
            f"lr={log['lr']:.2e} | {log['epoch_time']:.0f}s"
        )

        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, best_f1, cfg)

        if val_result["f1_macro"] > best_f1:
            best_f1 = val_result["f1_macro"]
            no_improve = 0
            save_model = ema_model.module if ema_model is not None else model
            save_checkpoint(output_dir / "best.pth", save_model, optimizer, scheduler, epoch, best_f1, cfg)
            class_names = [id2label[i] for i in range(num_classes)]
            save_confusion_matrix(val_result["trues"], val_result["preds"], class_names, output_dir / "confusion_best.png")
            save_report(val_result["trues"], val_result["preds"], class_names, output_dir / "report_best.txt")
            logger.info(f"  ★ New best F1: {best_f1:.4f}")
        else:
            no_improve += 1

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"[DONE] Best F1 = {best_f1:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    train_noise_aware(args.config, args.resume)
