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
from data.dataset_v2 import AUFERDatasetV2, collate_fn_v2
from models.core.fer_model import AUFERModel
from models.core.fer_model_beta import AUFERModelBeta
from models.core.fer_model_v2 import AURegionFormerV2
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
        # bf16 doesn't need GradScaler — disable to save overhead
        return torch.amp.GradScaler("cuda", enabled=False)
    def _autocast(device_type, enabled, dtype=torch.bfloat16):
        return torch.amp.autocast(device_type, enabled=enabled, dtype=dtype)
else:
    def _make_scaler(enabled):
        return torch.cuda.amp.GradScaler(enabled=False)
    def _autocast(device_type, enabled, dtype=torch.bfloat16):
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str, resume: str = None, init_from: str = None):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    # ─── Throughput knobs (TF32 + cudnn autotune + expandable_segments) ───
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    auto_regions = find_region_prefixes(train_csv)
    # Optional region subset override from dataset.region_subset config
    region_subset = cfg.get("dataset", {}).get("region_subset", None)
    if region_subset:
        missing = [r for r in region_subset if r not in auto_regions]
        if missing:
            raise ValueError(f"region_subset has unknown regions: {missing}. Available: {auto_regions}")
        region_prefixes = list(region_subset)
    else:
        region_prefixes = auto_regions
    num_au = len(region_prefixes)

    logger.info(f"Classes ({num_classes}): {label2id}")
    logger.info(f"AU regions ({num_au}): {region_prefixes}")

    # ── Model ──
    mcfg = cfg["model"]
    model_version = str(mcfg.get("version", "v1"))
    img_size = int(cfg["augmentation"].get("global_img_size", 224))

    use_beta = False  # set in v1 branch when applicable
    if model_version == "v2":
        # New patch-based AURegionFormerV2
        v2cfg = mcfg.get("v2", {}) or {}
        model = AURegionFormerV2(
            global_encoder=str(v2cfg.get("global_encoder", "mobilenetv4_conv_medium")),
            patch_encoder=str(v2cfg.get("patch_encoder", "mobilenetv4_conv_small")),
            pretrained=bool(mcfg.get("pretrained", True)),
            num_au=num_au, num_classes=num_classes,
            d_emb=int(mcfg.get("d_emb", 384)),
            patch_size=int(v2cfg.get("patch_size", 96)),
            img_size=img_size,
            use_stage_a=bool(v2cfg.get("use_stage_a", True)),
            stage_a_layers=int(v2cfg.get("stage_a_layers", 1)),
            stage_a_heads=int(v2cfg.get("stage_a_heads", 8)),
            use_stage_b=bool(v2cfg.get("use_stage_b", True)),
            use_iso=bool(v2cfg.get("use_iso", True)),
            use_rel=bool(v2cfg.get("use_rel", True)),
            combine=str(v2cfg.get("combine", "weighted")),
            tau_init=float(v2cfg.get("tau_init", 2.0)),
            n_fusion_layers=int(mcfg.get("n_fusion_layers", 1)),
            n_fusion_heads=int(mcfg.get("n_heads", 8)),
            dropout=float(mcfg.get("dropout", 0.1)),
            head_dropout=float(mcfg.get("head_dropout", 0.2)),
            gate_init=float(mcfg.get("gate_init", 0.0)),
            drop_path=float(mcfg.get("drop_path", 0.0)),
            use_au_pool=bool(mcfg.get("use_au_pool", False)),
        ).to(device)
        logger.info(f"[V2] global={v2cfg.get('global_encoder','mobilenetv4_conv_medium')} "
                    f"patch={v2cfg.get('patch_encoder','mobilenetv4_conv_small')} "
                    f"A={model.use_stage_a} B={model.use_stage_b} "
                    f"combine={v2cfg.get('combine','weighted')} "
                    f"tau_init={float(v2cfg.get('tau_init',2.0))}")
        mean, std = model.norm_mean, model.norm_std
    else:
        model_kwargs = dict(
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
            img_size=img_size,
            use_au_pool=bool(mcfg.get("use_au_pool", False)),
        )
        use_beta_v1 = bool(mcfg.get("use_beta", False))
        use_beta = use_beta_v1
        if use_beta_v1:
            bcfg = mcfg.get("beta", {}) or {}
            model = AUFERModelBeta(
                use_l2=bool(bcfg.get("use_l2", True)),
                l2_mode=str(bcfg.get("l2_mode", "scale")),
                use_l3=bool(bcfg.get("use_l3", False)),
                tau_init=float(bcfg.get("tau_init", 2.0)),
                **model_kwargs,
            ).to(device)
            logger.info(f"[BETA] use_l2={model.use_l2} l2_mode={bcfg.get('l2_mode','scale')} "
                        f"use_l3={model.use_l3} tau_init={float(bcfg.get('tau_init',2.0))}")
        else:
            model = AUFERModel(**model_kwargs).to(device)
        mean = model.backbone.norm_mean
        std = model.backbone.norm_std

    # ── Dataset ──
    if model_version == "v2":
        v2cfg = mcfg.get("v2", {}) or {}
        patch_size = int(v2cfg.get("patch_size", 96))
        # Patch scale ablation: global + per-region overrides
        dscfg = cfg.get("dataset", {}) or {}
        patch_scale = float(dscfg.get("patch_scale", 1.0))
        patch_scale_per_region = dscfg.get("patch_scale_per_region", {}) or {}
        if patch_scale != 1.0 or patch_scale_per_region:
            logger.info(f"[PATCH-SCALE] global={patch_scale} per_region={patch_scale_per_region}")
        train_set = AUFERDatasetV2(
            train_csv, label2id, region_prefixes,
            img_size=img_size, patch_size=patch_size,
            mean=mean, std=std,
            patch_mean=model.patch_norm_mean, patch_std=model.patch_norm_std,
            is_train=True,
            patch_dropout_n=int(cfg.get("training", {}).get("patch_dropout_n", 0)),
            patch_scale=patch_scale,
            patch_scale_per_region=patch_scale_per_region,
        )
        val_set = AUFERDatasetV2(
            val_csv, label2id, region_prefixes,
            img_size=img_size, patch_size=patch_size,
            mean=mean, std=std,
            patch_mean=model.patch_norm_mean, patch_std=model.patch_norm_std,
            is_train=False,
            patch_scale=patch_scale,
            patch_scale_per_region=patch_scale_per_region,
        )
        active_collate = collate_fn_v2
    else:
        train_set = AUFERDataset(
            train_csv, label2id, region_prefixes,
            img_size=img_size, mean=mean, std=std, is_train=True
        )
        val_set = AUFERDataset(
            val_csv, label2id, region_prefixes,
            img_size=img_size, mean=mean, std=std, is_train=False
        )
        active_collate = collate_fn

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
        drop_last=True, collate_fn=active_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        drop_last=False, collate_fn=active_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
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
    use_quality_weighting = bool(tcfg.get("use_quality_weighting", False))
    if use_quality_weighting:
        # switch criterion to per-sample reduction so we can multiply by quality_score
        if isinstance(criterion, FocalLoss):
            criterion.reduction = "none"
        logger.info("[QUALITY] sample-level loss reweighting via batch.quality_score enabled")

    # Beta-anchored aux loss (Stage 5): couples model.r_combined.mean to quality_score
    beta_anchor_weight = float(tcfg.get("beta_anchor_weight", 0.0))
    beta_var_weight    = float(tcfg.get("beta_var_weight", 0.0))   # encourage per-AU r diversity
    if beta_anchor_weight > 0:
        logger.info(f"[BETA-ANCHOR] aux = {beta_anchor_weight} * MSE(mean(r), q) + {beta_var_weight} * (1 - var(r))")

    # JEPA aux loss (Stage 7): mask AU patches, predict via Stage A
    jepa_weight = float(tcfg.get("jepa_weight", 0.0))
    jepa_n_mask = int(tcfg.get("jepa_n_mask", 1))
    if jepa_weight > 0:
        logger.info(f"[JEPA] aux_loss = {jepa_weight} * (1 - cos(predicted, target)) on {jepa_n_mask} masked AU patch(es)")

    # Stage 14: Probabilistic Beta uncertainty (KL Beta(α,β) || Beta(1,1) prior)
    beta_prior_kl_weight = float(tcfg.get("beta_prior_kl_weight", 0.0))
    if beta_prior_kl_weight > 0:
        logger.info(f"[BETA-PRIOR-KL] aux = {beta_prior_kl_weight} * KL(Beta(α,β) || Beta(1,1))")

    # Stage 11: Human-KL distillation (Yonsei → soft label distribution)
    human_kl_weight = float(tcfg.get("human_kl_weight", 0.0))
    if human_kl_weight > 0:
        logger.info(f"[HUMAN-KL] aux = {human_kl_weight} * KL(p_human || p_model). "
                    f"p_human = (1-mean_reject) on intended class, "
                    f"mean_reject/(C-1) uniform on other classes.")

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

    # Resume / init-from
    if init_from:
        load_checkpoint(init_from, model, optimizer=None, scheduler=None, init_only=True)
        logger.info(f"Initialized weights from {init_from} (fresh optimizer/scheduler/epoch)")
    elif resume:
        start_epoch, best_f1 = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch - 1}, best_f1={best_f1:.4f}")

    # Beta tau schedule (cosine 2.0 → 1.0 over training)
    bcfg = mcfg.get("beta", {}) or {}
    tau_start = float(bcfg.get("tau_init", 2.0))
    tau_end = float(bcfg.get("tau_end", 1.0))

    step = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()

        # ── Beta tau anneal (cosine) ──
        if use_beta or (model_version == "v2" and getattr(model, "use_stage_b", False)):
            progress = (epoch - 1) / max(1, epochs - 1)
            tau_now = tau_end + 0.5 * (tau_start - tau_end) * (1 + np.cos(np.pi * progress))
            model.set_tau(tau_now)

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
            labels = batch["label"].to(device, non_blocking=True)
            if model_version == "v2":
                au_input = batch["au_patches"].to(device, non_blocking=True)
            else:
                au_input = batch["au_coords"].to(device, non_blocking=True)

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

            with _autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits, global_feat = model(
                    images, au_input, return_features=True,
                    mixup_lambda=mixup_lambda, mixup_index=mixup_index,
                )
                loss = criterion(logits, mixed_targets)
                if use_quality_weighting and loss.dim() > 0:
                    qs = batch.get("quality_score")
                    if qs is not None:
                        qs = qs.to(device)
                        loss = (loss * qs).mean() / max(qs.mean().item(), 1e-6)
                    else:
                        loss = loss.mean()
                elif use_quality_weighting:
                    pass  # loss is already scalar (e.g., CrossEntropyLoss with reduction='mean')

                # ── Stage 14: KL(Beta(α,β) || Beta(1,1)) prior — encourages non-trivial uncertainty ──
                if beta_prior_kl_weight > 0 and hasattr(model, "get_last_reliability"):
                    rel = model.get_last_reliability().get("aux") or {}
                    # Beta(α,β) || Beta(1,1) closed-form KL:
                    # KL = (α-1)*ψ(α) + (β-1)*ψ(β) - (α+β-2)*ψ(α+β) - logB(α,β)
                    # We approximate by reading α/β from BetaHead output if accessible.
                    # Simpler proxy: use entropy of r as regularizer (penalize r too far from uniform)
                    r_combined = model.get_last_reliability().get("r_combined")
                    if r_combined is not None:
                        # encourage r to be neither all-near-1 nor all-near-0
                        ent = -(r_combined * torch.log(r_combined.clamp(min=1e-6)) +
                               (1-r_combined) * torch.log((1-r_combined).clamp(min=1e-6)))
                        # we want HIGH entropy → minimize negative entropy
                        loss = loss - beta_prior_kl_weight * ent.mean()

                # ── Stage 5: Beta-anchored aux loss + diversity penalty ──
                if beta_anchor_weight > 0 and hasattr(model, "get_last_reliability"):
                    rel = model.get_last_reliability().get("r_combined")
                    qs = batch.get("quality_score")
                    if rel is not None and qs is not None:
                        qs = qs.to(device).float()
                        r_mean = rel.mean(dim=1)            # [B]
                        anchor_term = ((r_mean - qs) ** 2).mean()
                        loss = loss + beta_anchor_weight * anchor_term
                        # Diversity penalty: penalize uniform r across AUs (prevents trivial collapse)
                        if beta_var_weight > 0:
                            r_var = rel.var(dim=1).mean()
                            div_term = (1.0 - 4.0 * r_var).clamp(min=0.0)  # max var of {0,1}-bounded ≈ 0.25
                            loss = loss + beta_var_weight * div_term

                # ── Stage 7: JEPA aux loss ──
                if jepa_weight > 0 and hasattr(model, "jepa_loss") and getattr(model, "use_stage_a", False):
                    jepa_aux = model.jepa_loss(images, au_input, n_mask=jepa_n_mask)
                    loss = loss + jepa_weight * jepa_aux

                # ── Stage 11: Human-KL distillation ──
                if human_kl_weight > 0:
                    mr = batch.get("mean_is_selected")
                    if mr is not None:
                        mr = mr.to(device).float().clamp(0.0, 1.0)        # [B]
                        C = num_classes
                        # p_human: (1-mr) on intended label, mr/(C-1) on others
                        p_human = (mr.unsqueeze(-1) / max(C - 1, 1)).expand(-1, C).clone()  # [B, C] uniform residual
                        # set intended class probability to (1 - mr)
                        idx = labels.long().unsqueeze(-1)
                        p_human.scatter_(1, idx, (1.0 - mr).unsqueeze(-1))
                        log_p_model = F.log_softmax(logits, dim=-1)
                        # KL(p_human || p_model) = sum p_human * (log p_human - log p_model)
                        # using F.kl_div with input=log_p_model, target=p_human (with log_target=False)
                        kl = F.kl_div(log_p_model, p_human, reduction="batchmean")
                        loss = loss + human_kl_weight * kl

                # Self-distillation: EMA teacher provides soft targets
                if distill_alpha > 0 and ema_model is not None and epoch > freeze_epochs:
                    with torch.no_grad():
                        teacher_logits = ema_model.module(images, au_input)
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
    ap.add_argument("--init_from", type=str, default=None,
                    help="Load only model weights (filter shape mismatches), reset optimizer/scheduler/epoch.")
    args = ap.parse_args()
    train(args.config, args.resume, args.init_from)
