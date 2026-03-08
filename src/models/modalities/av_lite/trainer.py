"""
V4-lite v2 Training Script
============================
Continuous + Binary A/V multi-task learning with:
  - Uncertainty-weighted loss (auto-balances tasks)
  - CCC / qAcc / Pearson metrics
  - GroupKFold CV (group by pid)
  - Gate temperature annealing
  - Modality dropout robustness
  - Ablation support (bio_variant, conditioner_variant)
  - Paper artifact generation

Usage:
  python -m modalities.av_lite.trainer --base_dir /path/to/precessed_data

  # TCN baseline (ablation)
  python -m modalities.av_lite.trainer --bio_variant tcn --conditioner cross_attention

  # Disable modality dropout (ablation)
  python -m modalities.av_lite.trainer --no_modality_dropout
"""

import argparse
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from modalities.av_lite.model import AVLiteModel
from modalities.av_lite.dataset import AVLiteDataset, collate_av_lite
from modalities.shared.audio_encoder import AudioTeacherAST


# ─────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────
def concordance_cc(y_true, y_pred):
    """Concordance Correlation Coefficient (CCC)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2 or y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return 0.0
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    ccc = 2.0 * cov / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-8)
    return float(np.clip(ccc, -1, 1))


def quadrant_accuracy(a_true, v_true, a_pred, v_pred, threshold=0.0):
    """4-quadrant classification accuracy from continuous A/V."""
    q_true = (np.array(a_true) >= threshold).astype(int) * 2 + (np.array(v_true) >= threshold).astype(int)
    q_pred = (np.array(a_pred) >= threshold).astype(int) * 2 + (np.array(v_pred) >= threshold).astype(int)
    return float(accuracy_score(q_true, q_pred))


def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2 or y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, teacher, loader, optimizer, scaler, device, mu_kd, use_amp):
    model.train()
    total_loss = 0
    n = 0

    for batch in loader:
        bio = batch["bio_raw"].to(device)
        mel = batch["logmel"].to(device)
        spk = batch["speaker_ids"].to(device)

        # Move targets to device
        batch_dev = {
            "a_norm": batch["a_norm"].to(device),
            "v_norm": batch["v_norm"].to(device),
            "label_A": batch["label_A"].to(device),
            "label_V": batch["label_V"].to(device),
        }

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(mel, bio, speaker_ids=spk)

            # Teacher for KD
            z_teacher = None
            if teacher is not None and teacher.available and mu_kd > 0:
                with torch.no_grad():
                    z_teacher = teacher(mel)

            loss, _ = model.compute_loss(out, batch_dev, mu_kd=mu_kd, z_teacher=z_teacher)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * bio.size(0)
        n += bio.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    # Collect predictions
    all_a_reg, all_v_reg = [], []
    all_a_logit, all_v_logit = [], []
    all_a_true, all_v_true = [], []
    all_a_bin_true, all_v_bin_true = [], []
    all_pids = []
    total_loss = 0
    n = 0

    for batch in loader:
        bio = batch["bio_raw"].to(device)
        mel = batch["logmel"].to(device)
        spk = batch["speaker_ids"].to(device)

        batch_dev = {
            "a_norm": batch["a_norm"].to(device),
            "v_norm": batch["v_norm"].to(device),
            "label_A": batch["label_A"].to(device),
            "label_V": batch["label_V"].to(device),
        }

        out = model(mel, bio, speaker_ids=spk)
        loss, _ = model.compute_loss(out, batch_dev)
        total_loss += loss.item() * bio.size(0)
        n += bio.size(0)

        all_a_reg.extend(out["A_reg"].squeeze(-1).cpu().numpy())
        all_v_reg.extend(out["V_reg"].squeeze(-1).cpu().numpy())
        all_a_logit.extend(out["A_logit"].squeeze(-1).cpu().numpy())
        all_v_logit.extend(out["V_logit"].squeeze(-1).cpu().numpy())
        all_a_true.extend(batch["a_norm"].numpy())
        all_v_true.extend(batch["v_norm"].numpy())
        all_a_bin_true.extend(batch["label_A"].numpy())
        all_v_bin_true.extend(batch["label_V"].numpy())
        all_pids.extend(batch["pids"].numpy())

    # Arrays
    a_reg = np.array(all_a_reg)
    v_reg = np.array(all_v_reg)
    a_true = np.array(all_a_true)
    v_true = np.array(all_v_true)

    # Continuous metrics
    ccc_A = concordance_cc(a_true, a_reg)
    ccc_V = concordance_cc(v_true, v_reg)
    pear_A = pearson_r(a_true, a_reg)
    pear_V = pearson_r(v_true, v_reg)
    mse_A = float(np.mean((a_true - a_reg) ** 2))
    mse_V = float(np.mean((v_true - v_reg) ** 2))
    qacc = quadrant_accuracy(a_true, v_true, a_reg, v_reg, threshold=0.0)

    # Binary metrics (from continuous output, threshold=0)
    pred_A_bin = (a_reg >= 0.0).astype(int)
    pred_V_bin = (v_reg >= 0.0).astype(int)
    true_A_bin = np.array(all_a_bin_true).astype(int)
    true_V_bin = np.array(all_v_bin_true).astype(int)

    A_acc = float(accuracy_score(true_A_bin, pred_A_bin))
    V_acc = float(accuracy_score(true_V_bin, pred_V_bin))
    A_f1 = float(f1_score(true_A_bin, pred_A_bin, zero_division=0))
    V_f1 = float(f1_score(true_V_bin, pred_V_bin, zero_division=0))

    # Also from binary head (for comparison)
    a_logit = np.array(all_a_logit)
    v_logit = np.array(all_v_logit)
    pred_A_logit = (a_logit >= 0.0).astype(int)
    pred_V_logit = (v_logit >= 0.0).astype(int)
    A_acc_head = float(accuracy_score(true_A_bin, pred_A_logit))
    V_acc_head = float(accuracy_score(true_V_bin, pred_V_logit))

    metrics = {
        "loss": total_loss / max(n, 1),
        # ★ Primary metrics (paper main table)
        "CCC_A": ccc_A, "CCC_V": ccc_V,
        "qAcc": qacc,
        "pearA": pear_A, "pearV": pear_V,
        "mseA": mse_A, "mseV": mse_V,
        # Binary (from continuous threshold)
        "A_acc": A_acc, "V_acc": V_acc,
        "A_f1": A_f1, "V_f1": V_f1,
        # Binary (from binary head, for ablation)
        "A_acc_head": A_acc_head, "V_acc_head": V_acc_head,
    }

    # ★ Score for checkpoint selection: qAcc-based
    metrics["score"] = 0.5 * qacc + 0.25 * max(ccc_A, 0) + 0.25 * max(ccc_V, 0)

    details = {
        "a_reg": a_reg, "v_reg": v_reg,
        "a_true": a_true, "v_true": v_true,
        "pids": np.array(all_pids),
    }
    return metrics, details


# ─────────────────────────────────────────────────────────────
#  Fold Training
# ─────────────────────────────────────────────────────────────
def train_fold(fold_idx, train_df, val_df, base_dir, out_dir, args, device):
    fold_dir = Path(out_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_set = AVLiteDataset(train_df, base_dir, mel_time_bins=args.mel_time_bins)
    val_set = AVLiteDataset(val_df, base_dir, mel_time_bins=args.mel_time_bins)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=collate_av_lite,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_av_lite,
    )

    # Model
    model = AVLiteModel(
        d_model=args.d_model,
        num_tokens=args.num_tokens,
        bio_channels=7,
        bio_variant=args.bio_variant,
        bio_n_layers=args.bio_n_layers,
        conditioner_variant=args.conditioner,
        n_heads=args.n_heads,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        p_drop_audio=args.p_drop_audio,
        p_drop_bio=args.p_drop_bio,
        use_speaker_embed=True,
        d_teacher=args.d_teacher,
    ).to(device)

    params = model.count_parameters()
    print(f"  Model: {params['trainable_M']}M trainable params")
    print(f"  Bio: {args.bio_variant}, Conditioner: {args.conditioner}")
    print(f"  Modality dropout: {args.modality_dropout} (audio={args.p_drop_audio}, bio={args.p_drop_bio})")

    # Teacher (optional)
    teacher = None
    if args.d_teacher > 0 and args.mu_kd > 0:
        teacher = AudioTeacherAST(d_teacher=args.d_teacher).to(device)
        if args.v4_ckpt:
            teacher.load_from_v4_checkpoint(args.v4_ckpt)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_score = -1
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Gate temperature annealing
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * progress
        model.set_gate_temperature(tau)

        # Train
        train_loss = train_one_epoch(
            model, teacher, train_loader, optimizer, scaler, device,
            args.mu_kd, args.amp and device.type == "cuda",
        )
        scheduler.step()

        # Evaluate
        val_m, val_d = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Uncertainty weights (for logging)
        uwl = model.uncertainty_loss.get_weights()

        row = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            **{k: round(v, 4) for k, v in val_m.items()},
            "tau": round(tau, 3), "lr": lr, "time": round(elapsed, 1),
            **{k: round(v, 4) for k, v in uwl.items()},
        }
        history.append(row)

        print(f"  [F{fold_idx} E{epoch:03d}] loss={train_loss:.4f} | "
              f"CCC_A={val_m['CCC_A']:.3f} CCC_V={val_m['CCC_V']:.3f} | "
              f"qAcc={val_m['qAcc']:.3f} | "
              f"A_acc={val_m['A_acc']:.3f} V_acc={val_m['V_acc']:.3f} | "
              f"score={val_m['score']:.4f} | τ={tau:.2f} {elapsed:.1f}s")

        if val_m["score"] > best_score:
            best_score = val_m["score"]
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch, "score": best_score,
                "metrics": val_m, "config": vars(args),
            }, fold_dir / "best.pth")
            print(f"    ★ New best: score={best_score:.4f}")

    # Save history
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    # Load best and final eval
    ckpt = torch.load(fold_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_m, final_d = evaluate(model, val_loader, device)

    # Save fold results
    result = {
        "fold": fold_idx, "best_epoch": best_epoch, "best_score": best_score,
        "val": final_m,
    }
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Gate analysis
    _save_gate_analysis(model, val_loader, device, fold_dir)

    # Predictions
    _save_predictions(final_d, fold_dir)

    return result


# ─────────────────────────────────────────────────────────────
#  Artifact generation
# ─────────────────────────────────────────────────────────────
def _save_gate_analysis(model, loader, device, out_dir):
    model.eval()
    all_gates, all_q_a, all_q_b = [], [], []
    with torch.no_grad():
        for batch in loader:
            bio = batch["bio_raw"].to(device)
            mel = batch["logmel"].to(device)
            spk = batch["speaker_ids"].to(device)
            out = model(mel, bio, speaker_ids=spk, return_artifacts=True)
            art = out["artifacts"]
            all_gates.append(art["gate_weights"].cpu().numpy())
            all_q_a.append(art["q_audio"].cpu().numpy())
            all_q_b.append(art["q_bio"].cpu().numpy())

    gates = np.concatenate(all_gates, axis=0)
    q_a = np.concatenate(all_q_a, axis=0).flatten()
    q_b = np.concatenate(all_q_b, axis=0).flatten()

    stats = {
        "gate_audio_mean": float(gates[:, 0].mean()),
        "gate_bio_mean": float(gates[:, 1].mean()),
        "quality_audio_mean": float(q_a.mean()),
        "quality_bio_mean": float(q_b.mean()),
        "n_samples": len(gates),
    }
    with open(out_dir / "gate_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def _save_predictions(details, out_dir):
    df = pd.DataFrame({
        "pid": details["pids"],
        "a_true": details["a_true"],
        "v_true": details["v_true"],
        "a_pred": details["a_reg"],
        "v_pred": details["v_reg"],
    })
    df.to_csv(out_dir / "predictions.csv", index=False)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V4-lite v2 Trainer")
    parser.add_argument("--base_dir", type=str, default="/home/jy/260210/precessed_data")
    parser.add_argument("--out_dir", type=str, default="./checkpoints_av_lite_v2")
    parser.add_argument("--index_csv", type=str, default=None)

    # Architecture (ablation knobs)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bio_variant", type=str, default="mamba",
                        choices=["mamba", "tcn"])
    parser.add_argument("--bio_n_layers", type=int, default=2)
    parser.add_argument("--conditioner", type=str, default="selective_ssm",
                        choices=["selective_ssm", "cross_attention"])

    # Modality dropout
    parser.add_argument("--modality_dropout", action="store_true", default=True)
    parser.add_argument("--no_modality_dropout", dest="modality_dropout", action="store_false")
    parser.add_argument("--p_drop_audio", type=float, default=0.1)
    parser.add_argument("--p_drop_bio", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--n_folds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--mel_time_bins", type=int, default=128)

    # Distillation
    parser.add_argument("--d_teacher", type=int, default=768)
    parser.add_argument("--mu_kd", type=float, default=0.0)
    parser.add_argument("--v4_ckpt", type=str, default=None)

    # Gate annealing
    parser.add_argument("--tau_start", type=float, default=2.0)
    parser.add_argument("--tau_end", type=float, default=0.5)

    args = parser.parse_args()
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: bio={args.bio_variant}, conditioner={args.conditioner}")
    print(f"Modality dropout: {args.modality_dropout}")

    # Load index
    index_csv = args.index_csv or str(Path(args.base_dir) / "segments_index_train.csv")
    df = pd.read_csv(index_csv)
    print(f"Total samples: {len(df)}")

    A = df["label_ext_A_norm"].values
    V = df["label_ext_V_norm"].values
    print(f"Arousal: mean={A.mean():.3f}, High%={(A>=0).mean()*100:.1f}%")
    print(f"Valence: mean={V.mean():.3f}, High%={(V>=0).mean()*100:.1f}%")

    # GroupKFold by pid
    groups = df["pid"].values
    gkf = GroupKFold(n_splits=args.n_folds)

    all_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx}/{args.n_folds}")
        print(f"{'='*70}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
        print(f"  Val PIDs: {sorted(val_df['pid'].unique())}")

        result = train_fold(fold_idx, train_df, val_df, args.base_dir,
                            args.out_dir, args, device)
        all_results.append(result)

    # CV Summary
    print(f"\n{'='*70}")
    print(f"Cross-Validation Summary ({args.bio_variant} + {args.conditioner})")
    print(f"{'='*70}")

    summary = defaultdict(list)
    for r in all_results:
        for k, v in r["val"].items():
            summary[k].append(v)

    cv_summary = {}
    for k, vals in summary.items():
        arr = np.array(vals)
        cv_summary[k] = {"mean": round(float(arr.mean()), 4),
                         "std": round(float(arr.std()), 4)}
        print(f"  {k}: {arr.mean():.4f} ± {arr.std():.4f}")

    cv_summary["config"] = {
        "bio_variant": args.bio_variant,
        "conditioner": args.conditioner,
        "modality_dropout": args.modality_dropout,
        "d_model": args.d_model,
        "bio_n_layers": args.bio_n_layers,
        "epochs": args.epochs,
    }

    with open(Path(args.out_dir) / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\nResults saved to: {args.out_dir}")
    print(f"\n{'='*70}")
    print("ABLATION COMMANDS:")
    print(f"{'='*70}")
    print(f"# Full model (default):")
    print(f"python -m modalities.av_lite.trainer --bio_variant mamba --conditioner selective_ssm")
    print(f"\n# TCN baseline:")
    print(f"python -m modalities.av_lite.trainer --bio_variant tcn --conditioner cross_attention --out_dir ./ckpt_tcn_xattn")
    print(f"\n# Mamba + vanilla attention:")
    print(f"python -m modalities.av_lite.trainer --bio_variant mamba --conditioner cross_attention --out_dir ./ckpt_mamba_xattn")
    print(f"\n# No modality dropout:")
    print(f"python -m modalities.av_lite.trainer --no_modality_dropout --out_dir ./ckpt_no_dropout")


if __name__ == "__main__":
    main()
