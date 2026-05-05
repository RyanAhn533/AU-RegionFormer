"""
Main Training Loop
====================
기존 코드 대비 수정사항:
  1. Backbone freeze → unfreeze 시 remaining_steps로 scheduler 재계산
  2. Focal loss 단독 사용 (label_smoothing + class_weight 충돌 제거)
  3. Differentiated LR (backbone vs head)
  4. Expression magnitude scorer의 neutral centroid EMA 업데이트
  5. 단일 backbone → 321x 연산 감소

버그 수정:
  - YAML config 값 float/int 캐스팅 (PyYAML string 파싱 이슈)
  - torch.cuda.amp deprecated API 호환 (torch 2.0+)
"""

import time
import yaml
import numpy as np
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
from models.core.fer_model import AUFERModel
from training.losses import FocalLoss, compute_class_weights
from training.scheduler import build_scheduler
from training.evaluator import evaluate, save_confusion_matrix, save_report
from utils.seed import set_seed, ensure_dir
from utils.logging import setup_logger, MetricLogger
from utils.checkpoint import save_checkpoint, load_checkpoint

try:
    from timm.utils import ModelEmaV3
    HAS_EMA = True
except ImportError:
    HAS_EMA = False

# ── AMP 호환 (torch 2.0 vs 2.1+) ──
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str, resume: str = None):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    # Paths
    train_csv = str(cfg["paths"]["train_csv"])
    val_csv = str(cfg["paths"]["val_csv"])
    output_dir = Path(cfg["paths"]["output_dir"])
    ensure_dir(output_dir)

    # Logger
    logger = setup_logger("train", str(output_dir))
    metric_log = MetricLogger(output_dir / "history.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Labels & Regions ──
    label2id = build_label_mapping(train_csv)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    region_prefixes = find_region_prefixes(train_csv)
    num_au = len(region_prefixes)

    logger.info(f"Classes ({num_classes}): {label2id}")
    logger.info(f"AU regions ({num_au}): {region_prefixes}")

    # ── Model ──
    mcfg = cfg["model"]
    model = AUFERModel(
        backbone_name=str(mcfg.get("backbone", "mobilevitv2_100")),
        pretrained=bool(mcfg.get("pretrained", True)),
        num_au=num_au,
        num_classes=num_classes,
        d_emb=int(mcfg.get("d_emb", 384)),
        n_heads=int(mcfg.get("n_heads", 8)),
        n_fusion_layers=int(mcfg.get("n_fusion_layers", 1)),
        roi_mode=str(mcfg.get("roi_mode", "bilinear")),
        roi_spatial=int(mcfg.get("roi_spatial", 1)),
        dropout=float(mcfg.get("dropout", 0.1)),
        head_dropout=float(mcfg.get("head_dropout", 0.2)),
        gate_init=float(mcfg.get("gate_init", 0.0)),
        drop_path=float(mcfg.get("drop_path", 0.0)),
        img_size=int(cfg["augmentation"].get("global_img_size", 224)),
        use_au_pool=bool(mcfg.get("use_au_pool", False)),
    ).to(device)

    # Normalization from backbone
    mean = model.backbone.norm_mean
    std = model.backbone.norm_std

    # ── Dataset ──
    img_size = int(cfg["augmentation"].get("global_img_size", 224))
    train_set = AUFERDataset(
        train_csv, label2id, region_prefixes,
        img_size=img_size, mean=mean, std=std, is_train=True
    )
    val_set = AUFERDataset(
        val_csv, label2id, region_prefixes,
        img_size=img_size, mean=mean, std=std, is_train=False
    )

    tcfg = cfg["training"]

    # EMA (Exponential Moving Average)
    use_ema = bool(tcfg.get("ema", False)) and HAS_EMA
    ema_model = None
    ema_decay = float(tcfg.get("ema_decay", 0.9998))
    if use_ema:
        ema_model = ModelEmaV3(model, decay=ema_decay)
        logger.info(f"EMA enabled: decay={ema_decay}")

    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg["num_workers"])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_fn
    )

    # ── Loss ──
    loss_type = str(tcfg.get("loss", "focal"))
    if loss_type == "focal":
        alpha = compute_class_weights(train_csv, label2id).to(device)
        criterion = FocalLoss(
            gamma=float(tcfg.get("focal_gamma", 2.0)),
            alpha=alpha,
            label_smoothing=float(tcfg.get("label_smoothing", 0.0)),
        )
    else:
        alpha = compute_class_weights(train_csv, label2id).to(device)
        criterion = nn.CrossEntropyLoss(weight=alpha)

    # ── Training phases (모든 config 값 명시적 캐스팅) ──
    epochs = int(tcfg["epochs"])
    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 3))
    base_lr = float(tcfg["base_lr"])
    backbone_lr_scale = float(tcfg.get("backbone_lr_scale", 0.1))
    weight_decay = float(tcfg.get("weight_decay", 0.05))
    grad_clip = float(tcfg.get("grad_clip_norm", 5.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.05))
    save_every = int(tcfg.get("save_every", 10))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    mixup_alpha = float(tcfg.get("mixup_alpha", 0.0))
    distill_alpha = float(tcfg.get("distill_alpha", 0.0))  # self-distillation weight
    distill_temp = float(tcfg.get("distill_temp", 4.0))    # temperature for KL
    early_stop_patience = int(tcfg.get("early_stop_patience", 0))  # 0 = disabled

    # Neutral class ID for expression magnitude scorer
    neutral_id = label2id.get("neutral", label2id.get("Neutral", -1))

    # ── Phase 1: Backbone frozen ──
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr, backbone_lr_scale),
        weight_decay=weight_decay
    )
    scheduler_mode = str(tcfg.get("scheduler", "cosine_warmup"))
    restart_epochs = int(tcfg.get("restart_epochs", 0))

    phase1_steps = len(train_loader) * freeze_epochs
    scheduler = build_scheduler(optimizer, phase1_steps, warmup_ratio=warmup_ratio,
                                mode=scheduler_mode)
    scaler = _make_scaler(enabled=use_amp)

    best_f1 = -1.0
    start_epoch = 1
    no_improve_count = 0  # early stopping counter
    train_start_time = time.time()

    # Resume
    if resume:
        start_epoch, best_f1 = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch - 1}, best_f1={best_f1:.4f}")

    step = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()

        # ── Phase transition: unfreeze backbone ──
        if epoch == freeze_epochs + 1:
            logger.info("=== Unfreezing backbone ===")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.get_param_groups(base_lr, backbone_lr_scale),
                weight_decay=weight_decay
            )
            remaining_steps = len(train_loader) * (epochs - freeze_epochs)
            restart_period = len(train_loader) * restart_epochs if restart_epochs > 0 else 0
            scheduler = build_scheduler(optimizer, remaining_steps,
                                        warmup_ratio=warmup_ratio,
                                        mode=scheduler_mode,
                                        restart_period=restart_period)
            scaler = _make_scaler(enabled=use_amp)
            # Re-init EMA after unfreeze (all params now trainable)
            if use_ema:
                ema_model = ModelEmaV3(model, decay=ema_decay)
                logger.info("EMA re-initialized after backbone unfreeze")

        running_loss = 0.0
        running_correct = 0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            au_coords = batch["au_coords"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Manifold MixUp
            mixup_lambda = None
            mixup_index = None
            mixed_targets = labels  # default: hard labels
            if mixup_alpha > 0 and model.training:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1 - lam)  # ensure lam >= 0.5
                mixup_index = torch.randperm(images.size(0), device=device)
                mixup_lambda = lam
                # Soft targets for loss
                C = num_classes
                targets_a = F.one_hot(labels, C).float()
                targets_b = F.one_hot(labels[mixup_index], C).float()
                mixed_targets = lam * targets_a + (1 - lam) * targets_b

            with _autocast("cuda", enabled=use_amp):
                logits, global_feat = model(
                    images, au_coords, return_features=True,
                    mixup_lambda=mixup_lambda, mixup_index=mixup_index,
                )
                loss = criterion(logits, mixed_targets)

                # Self-distillation: EMA teacher provides soft targets
                if distill_alpha > 0 and ema_model is not None and epoch > freeze_epochs:
                    with torch.no_grad():
                        teacher_logits = ema_model.module(images, au_coords)
                    T = distill_temp
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=-1),
                        F.softmax(teacher_logits.detach() / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                    loss = (1 - distill_alpha) * loss + distill_alpha * kd_loss

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            # EMA update
            if ema_model is not None:
                ema_model.update(model)

            # Update expression magnitude centroid
            if neutral_id >= 0:
                model.expr_scorer.update_centroid(
                    global_feat.detach(), labels, neutral_id
                )

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(1) == labels).sum().item()  # always vs hard labels
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        train_acc = running_correct / max(1, n_samples)
        epoch_time = time.time() - t0

        # ── Validation (EMA model if available) ──
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
            "epoch_time": round(epoch_time, 1),
        }

        # Progress tracking
        elapsed_total = time.time() - train_start_time
        epochs_done = epoch - start_epoch + 1
        epochs_remain = epochs - epoch
        avg_epoch_time = elapsed_total / max(1, epochs_done)
        eta_seconds = avg_epoch_time * epochs_remain

        log["elapsed_total_min"] = round(elapsed_total / 60, 1)
        log["eta_min"] = round(eta_seconds / 60, 1)
        log["avg_epoch_sec"] = round(avg_epoch_time, 1)

        # GPU memory (if available)
        if device.type == "cuda":
            log["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            log["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
            log["gpu_max_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

        metric_log.log(log)

        # Enhanced console output with ETA
        eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m" if eta_seconds > 3600 else f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
        logger.info(
            f"[E{epoch:03d}/{epochs}] train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_result['loss']:.4f} f1={val_result['f1_macro']:.4f} "
            f"acc={val_result['accuracy']:.4f} | lr={log['lr']:.2e} | "
            f"{epoch_time:.1f}s/ep | ETA: {eta_str}"
        )

        # ── Save last ──
        save_checkpoint(
            output_dir / "last.pth", model, optimizer, scheduler,
            epoch, best_f1, cfg
        )

        # ── Save best ──
        if val_result["f1_macro"] > best_f1:
            best_f1 = val_result["f1_macro"]
            no_improve_count = 0
            save_model = ema_model.module if ema_model is not None else model
            save_checkpoint(
                output_dir / "best.pth", save_model, optimizer, scheduler,
                epoch, best_f1, cfg
            )
            class_names = [id2label[i] for i in range(num_classes)]
            save_confusion_matrix(
                val_result["trues"], val_result["preds"],
                class_names, output_dir / "confusion_best.png"
            )
            save_report(
                val_result["trues"], val_result["preds"],
                class_names, output_dir / "report_best.txt"
            )
            logger.info(f"  ★ New best F1: {best_f1:.4f}")
        else:
            no_improve_count += 1

        # ── Early stopping ──
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

        # ── Periodic save ──
        if epoch % save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pth", model, optimizer,
                scheduler, epoch, best_f1, cfg
            )

    total_train_time = time.time() - train_start_time
    logger.info(f"[DONE] Best macro-F1 = {best_f1:.4f} | Total time: {total_train_time/3600:.1f}h")

    # Save experiment summary
    import json
    exp_summary = {
        "best_f1": round(best_f1, 4),
        "total_epochs": epochs,
        "total_time_hours": round(total_train_time / 3600, 2),
        "total_time_minutes": round(total_train_time / 60, 1),
        "avg_epoch_seconds": round(total_train_time / max(1, epochs), 1),
        "device": str(device),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "num_classes": num_classes,
        "num_au_regions": num_au,
    }
    if device.type == "cuda":
        exp_summary["gpu_name"] = torch.cuda.get_device_name(0)
        exp_summary["gpu_max_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    # ── Generate paper artifacts from best model ──
    logger.info("=== Generating paper-grade artifacts ===")
    from training.evaluator import generate_all_artifacts

    # Load best model
    best_ckpt = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    generate_all_artifacts(
        model=model,
        val_loader=val_loader,
        device=device,
        id2label=id2label,
        output_dir=str(output_dir),
        history_path=str(output_dir / "history.jsonl"),
        config=cfg,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    train(args.config, args.resume)
