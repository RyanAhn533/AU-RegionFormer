"""
Paper-Grade Evaluation & Artifact Generation
==============================================

논문/학회 제출에 필요한 모든 산출물을 생성.

저장 목록:
  1. confusion_matrix.png          — Normalized + Raw 둘 다
  2. classification_report.txt     — Per-class precision/recall/F1
  3. per_class_f1_bar.png          — 클래스별 F1 bar chart
  4. training_curves.png           — loss, acc, f1, lr 커브 (4 subplot)
  5. roc_curves.png                — Per-class ROC + AUC
  6. pr_curves.png                 — Per-class Precision-Recall + AP
  7. metrics_summary.json          — 모든 수치 한 파일에
  8. predictions.csv               — 전체 예측 결과 (path, true, pred, probs)
  9. model_summary.txt             — 파라미터 수, FLOPs, 모델 구조
  10. misclassified_samples.csv    — 오분류 샘플 목록
  11. attention_gate_values.json   — Fusion gate 학습 현황
  12. latency_benchmark.json       — 추론 속도 (CPU/GPU)
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, top_k_accuracy_score, cohen_kappa_score
)


# ═══════════════ Core Evaluation ═══════════════

@torch.no_grad()
def evaluate(model, loader, device):
    """Basic evaluation returning preds, trues, logits."""
    model.eval()
    all_preds, all_trues, all_logits, all_losses, all_paths = [], [], [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        # v2 model uses patches; v1 uses coords. Pick whichever the batch has.
        if "au_patches" in batch:
            au_input = batch["au_patches"].to(device, non_blocking=True)
        else:
            au_input = batch["au_coords"].to(device, non_blocking=True)

        logits = model(images, au_input)
        loss = F.cross_entropy(logits, labels)
        all_losses.append(loss.item())

        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_trues.extend(labels.cpu().numpy())
        all_paths.extend(batch["path"])

    all_logits = torch.cat(all_logits, dim=0)  # [N, C]
    all_probs = F.softmax(all_logits, dim=-1).numpy()  # [N, C]

    f1 = f1_score(all_trues, all_preds, average="macro")
    acc = accuracy_score(all_trues, all_preds)

    return {
        "loss": float(np.mean(all_losses)),
        "f1_macro": float(f1),
        "accuracy": float(acc),
        "preds": all_preds,
        "trues": all_trues,
        "probs": all_probs,
        "logits": all_logits.numpy(),
        "paths": all_paths,
    }


# ═══════════════ Full Paper-Grade Artifacts ═══════════════

def generate_all_artifacts(
    model, val_loader, device,
    id2label: Dict[int, str],
    output_dir: str,
    history_path: str = None,
    config: dict = None,
):
    """
    모든 논문용 산출물 생성.
    학습 완료 후 또는 best model 로드 후 호출.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out / "paper_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating paper artifacts → {artifacts_dir}")
    print(f"{'='*60}")

    # 1) Evaluate
    result = evaluate(model, val_loader, device)
    trues = np.array(result["trues"])
    preds = np.array(result["preds"])
    probs = result["probs"]
    paths = result["paths"]

    num_classes = len(id2label)
    class_names = [id2label[i] for i in range(num_classes)]

    # 2) Metrics summary
    metrics = _compute_all_metrics(trues, preds, probs, class_names)
    metrics["config"] = config or {}
    _save_json(metrics, artifacts_dir / "metrics_summary.json")
    print(f"  [1/12] metrics_summary.json ✓")

    # 3) Classification report
    _save_classification_report(trues, preds, class_names, artifacts_dir)
    print(f"  [2/12] classification_report.txt ✓")

    # 4) Confusion matrices (raw + normalized)
    _save_confusion_matrices(trues, preds, class_names, artifacts_dir)
    print(f"  [3/12] confusion_matrix.png ✓")

    # 5) Per-class F1 bar chart
    _save_per_class_f1_bar(trues, preds, class_names, artifacts_dir)
    print(f"  [4/12] per_class_f1_bar.png ✓")

    # 6) ROC curves
    _save_roc_curves(trues, probs, class_names, artifacts_dir)
    print(f"  [5/12] roc_curves.png ✓")

    # 7) PR curves
    _save_pr_curves(trues, probs, class_names, artifacts_dir)
    print(f"  [6/12] pr_curves.png ✓")

    # 8) Training curves
    if history_path and Path(history_path).exists():
        _save_training_curves(history_path, artifacts_dir)
        print(f"  [7/12] training_curves.png ✓")
    else:
        print(f"  [7/12] training_curves.png ✗ (no history)")

    # 9) Predictions CSV
    _save_predictions_csv(paths, trues, preds, probs, class_names, artifacts_dir)
    print(f"  [8/12] predictions.csv ✓")

    # 10) Misclassified samples
    _save_misclassified(paths, trues, preds, probs, class_names, artifacts_dir)
    print(f"  [9/12] misclassified_samples.csv ✓")

    # 11) Model summary
    _save_model_summary(model, device, artifacts_dir,
                        img_size=config.get("augmentation", {}).get("global_img_size", 224) if config else 224,
                        num_au=len([k for k in id2label]) if id2label else 8)
    print(f"  [10/12] model_summary.txt ✓")

    # 12) Attention gate values
    _save_gate_values(model, artifacts_dir)
    print(f"  [11/12] attention_gate_values.json ✓")

    # 13) Latency benchmark
    _save_latency_benchmark(model, device, artifacts_dir,
                            img_size=config.get("augmentation", {}).get("global_img_size", 224) if config else 224)
    print(f"  [12/14] latency_benchmark.json ✓")

    # 14) AU attention heatmap
    region_names = None
    try:
        from data.dataset import find_region_prefixes
        csv_path = config.get("paths", {}).get("val_csv") or config.get("paths", {}).get("train_csv")
        if csv_path:
            region_names = find_region_prefixes(str(csv_path))
    except Exception:
        pass
    save_au_attention_map(model, val_loader, device, class_names, artifacts_dir,
                          region_names=region_names)
    print(f"  [13/14] au_attention_heatmap.png ✓")

    # 15) Experiment elapsed time
    _save_json({"generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
               artifacts_dir / "experiment_meta.json")
    print(f"  [14/14] experiment_meta.json ✓")

    print(f"\n  All artifacts saved to: {artifacts_dir}")
    return metrics


# ═══════════════ Individual Generators ═══════════════

def _compute_all_metrics(trues, preds, probs, class_names):
    """논문 Table 1에 들어갈 모든 수치."""
    num_classes = len(class_names)
    metrics = {}

    # Overall
    metrics["accuracy"] = float(accuracy_score(trues, preds))
    metrics["f1_macro"] = float(f1_score(trues, preds, average="macro"))
    metrics["f1_weighted"] = float(f1_score(trues, preds, average="weighted"))
    metrics["precision_macro"] = float(precision_score(trues, preds, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(trues, preds, average="macro", zero_division=0))
    metrics["cohen_kappa"] = float(cohen_kappa_score(trues, preds))

    # Top-k accuracy
    if num_classes > 2:
        try:
            metrics["top2_accuracy"] = float(top_k_accuracy_score(trues, probs, k=2))
            metrics["top3_accuracy"] = float(top_k_accuracy_score(trues, probs, k=3))
        except Exception:
            pass

    # AUC (one-vs-rest)
    try:
        metrics["auc_macro"] = float(roc_auc_score(trues, probs, multi_class="ovr", average="macro"))
        metrics["auc_weighted"] = float(roc_auc_score(trues, probs, multi_class="ovr", average="weighted"))
    except Exception:
        metrics["auc_macro"] = None

    # mAP
    try:
        from sklearn.preprocessing import label_binarize
        trues_bin = label_binarize(trues, classes=list(range(num_classes)))
        metrics["mAP"] = float(average_precision_score(trues_bin, probs, average="macro"))
    except Exception:
        metrics["mAP"] = None

    # Per-class
    per_class = {}
    f1_per = f1_score(trues, preds, average=None, zero_division=0)
    prec_per = precision_score(trues, preds, average=None, zero_division=0)
    rec_per = recall_score(trues, preds, average=None, zero_division=0)

    for i, name in enumerate(class_names):
        per_class[name] = {
            "f1": round(float(f1_per[i]), 4),
            "precision": round(float(prec_per[i]), 4),
            "recall": round(float(rec_per[i]), 4),
            "support": int(np.sum(np.array(trues) == i)),
        }
    metrics["per_class"] = per_class
    metrics["total_samples"] = len(trues)

    return metrics


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _save_classification_report(trues, preds, class_names, out_dir):
    report = classification_report(trues, preds, target_names=class_names, digits=4)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")


def _save_confusion_matrices(trues, preds, class_names, out_dir):
    """Raw + Normalized confusion matrices side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_raw = confusion_matrix(trues, preds)
        cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, cm, title, fmt in [
            (axes[0], cm_raw, "Confusion Matrix (Count)", "d"),
            (axes[1], cm_norm, "Confusion Matrix (Normalized)", ".2f"),
        ]:
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(title, fontsize=13, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)

            ticks = np.arange(len(class_names))
            ax.set_xticks(ticks)
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(ticks)
            ax.set_yticklabels(class_names, fontsize=9)

            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > thresh else "black")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")

        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Also save raw numpy for LaTeX table generation
        np.save(out_dir / "confusion_matrix_raw.npy", cm_raw)
        np.save(out_dir / "confusion_matrix_norm.npy", cm_norm)
    except Exception as e:
        print(f"  [WARN] confusion matrix: {e}")


def _save_per_class_f1_bar(trues, preds, class_names, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f1_per = f1_score(trues, preds, average=None, zero_division=0)
        f1_macro = f1_score(trues, preds, average="macro")

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(class_names))
        bars = ax.bar(x, f1_per, color="#4C72B0", alpha=0.85, edgecolor="white")

        # Macro line
        ax.axhline(y=f1_macro, color="red", linestyle="--", linewidth=1.5,
                    label=f"Macro F1 = {f1_macro:.4f}")

        # Value labels on bars
        for bar, val in zip(bars, f1_per):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(f1_per) + 0.1))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "per_class_f1_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] per-class F1 bar: {e}")


def _save_roc_curves(trues, probs, class_names, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import label_binarize

        num_classes = len(class_names)
        trues_bin = label_binarize(trues, classes=list(range(num_classes)))

        fig, ax = plt.subplots(figsize=(8, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

        for i, (name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(trues_bin[:, i], probs[:, i])
            auc_val = roc_auc_score(trues_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{name} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "roc_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] ROC curves: {e}")


def _save_pr_curves(trues, probs, class_names, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import label_binarize

        num_classes = len(class_names)
        trues_bin = label_binarize(trues, classes=list(range(num_classes)))

        fig, ax = plt.subplots(figsize=(8, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

        for i, (name, color) in enumerate(zip(class_names, colors)):
            prec, rec, _ = precision_recall_curve(trues_bin[:, i], probs[:, i])
            ap = average_precision_score(trues_bin[:, i], probs[:, i])
            ax.plot(rec, prec, color=color, lw=1.5,
                    label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "pr_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] PR curves: {e}")


def _save_training_curves(history_path, out_dir):
    """history.jsonl → 4-panel training curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(history_path, "r") as f:
            history = [json.loads(line) for line in f if line.strip()]

        if not history:
            return

        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, [h["train_loss"] for h in history], label="Train", linewidth=1.5)
        ax.plot(epochs, [h["val_loss"] for h in history], label="Val", linewidth=1.5)
        ax.set_title("Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

        # Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, [h["train_acc"] for h in history], label="Train", linewidth=1.5)
        ax.plot(epochs, [h["val_acc"] for h in history], label="Val", linewidth=1.5)
        ax.set_title("Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

        # F1
        ax = axes[1, 0]
        ax.plot(epochs, [h["val_f1"] for h in history], color="green", linewidth=1.5)
        best_idx = np.argmax([h["val_f1"] for h in history])
        ax.axvline(x=epochs[best_idx], color="red", linestyle="--", alpha=0.5,
                    label=f"Best: {history[best_idx]['val_f1']:.4f} @ E{epochs[best_idx]}")
        ax.set_title("Validation Macro-F1", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

        # Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, [h["lr"] for h in history], color="orange", linewidth=1.5)
        ax.set_title("Learning Rate", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

        plt.suptitle("Training Curves", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] training curves: {e}")


def _save_predictions_csv(paths, trues, preds, probs, class_names, out_dir):
    """전체 예측 결과 CSV. 리뷰어가 특정 샘플 검토할 때 필요."""
    rows = []
    for i in range(len(trues)):
        row = {
            "path": paths[i],
            "true_label": class_names[trues[i]],
            "pred_label": class_names[preds[i]],
            "correct": int(trues[i] == preds[i]),
        }
        for j, name in enumerate(class_names):
            row[f"prob_{name}"] = round(float(probs[i, j]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)


def _save_misclassified(paths, trues, preds, probs, class_names, out_dir):
    """오분류 샘플만 추출. Failure analysis 섹션에 필수."""
    rows = []
    for i in range(len(trues)):
        if trues[i] != preds[i]:
            rows.append({
                "path": paths[i],
                "true_label": class_names[trues[i]],
                "pred_label": class_names[preds[i]],
                "true_prob": round(float(probs[i, trues[i]]), 4),
                "pred_prob": round(float(probs[i, preds[i]]), 4),
                "confidence_gap": round(float(probs[i, preds[i]] - probs[i, trues[i]]), 4),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("confidence_gap", ascending=False)
    df.to_csv(out_dir / "misclassified_samples.csv", index=False)


def _save_model_summary(model, device, out_dir, img_size=224, num_au=8):
    """파라미터 수, 구조, FLOPs 추정."""
    lines = []

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"Total Parameters:     {total_params:,}")
    lines.append(f"Trainable Parameters: {trainable_params:,}")
    lines.append(f"Model Size (MB):      {total_params * 4 / 1024 / 1024:.1f}")
    lines.append("")

    # Per-module breakdown
    lines.append("Module Parameter Breakdown:")
    lines.append("-" * 50)
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        lines.append(f"  {name:30s} {n_params:>12,}")
    lines.append("")

    # FLOPs estimation (rough)
    try:
        from fvcore.nn import FlopCountAnalysis
        is_v2 = hasattr(model, "patch_enc")
        patch_size = getattr(model, "patch_size", 96)
        dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
        if is_v2:
            dummy_au = torch.randn(1, num_au, 3, patch_size, patch_size).to(device)
        else:
            dummy_au = torch.randn(1, num_au, 2).to(device)
        flops = FlopCountAnalysis(model, (dummy_img, dummy_au))
        lines.append(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    except ImportError:
        lines.append("FLOPs: (install fvcore for FLOPs counting)")
    except Exception as e:
        lines.append(f"FLOPs: error ({e})")

    lines.append("")
    lines.append("Model Architecture:")
    lines.append("=" * 50)
    lines.append(str(model))

    (out_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _save_gate_values(model, out_dir):
    """Cross-attention fusion의 gate 값 저장. 학습이 제대로 됐는지 확인용."""
    gate_info = {}
    for name, param in model.named_parameters():
        if "gate" in name.lower():
            val = torch.sigmoid(param.detach().cpu())
            gate_info[name] = {
                "mean": round(float(val.mean()), 4),
                "std": round(float(val.std()), 4),
                "min": round(float(val.min()), 4),
                "max": round(float(val.max()), 4),
            }
    _save_json(gate_info, out_dir / "attention_gate_values.json")


def _save_latency_benchmark(model, device, out_dir, img_size=224, num_au=8,
                            n_warmup=10, n_runs=50):
    """추론 속도 벤치마크. 논문 Table에 latency/throughput 필수."""
    model.eval()
    # v2 model takes [B, K, 3, P, P] patches; v1 takes [B, K, 2] coords.
    is_v2 = hasattr(model, "patch_enc")  # AURegionFormerV2 has patch_enc
    patch_size = getattr(model, "patch_size", 96)

    dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
    if is_v2:
        dummy_au = torch.randn(1, num_au, 3, patch_size, patch_size).to(device)
    else:
        dummy_au = torch.randn(1, num_au, 2).to(device)

    results = {}

    # GPU benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            model(dummy_img, dummy_au)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy_img, dummy_au)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        results["gpu_latency_ms"] = round(float(np.mean(times)) * 1000, 2)
        results["gpu_latency_std_ms"] = round(float(np.std(times)) * 1000, 2)
        results["gpu_fps"] = round(1.0 / float(np.mean(times)), 1)

    # CPU benchmark
    model_cpu = model.cpu()
    dummy_img_cpu = dummy_img.cpu()
    dummy_coords_cpu = dummy_au.cpu()

    for _ in range(n_warmup):
        model_cpu(dummy_img_cpu, dummy_coords_cpu)

    times_cpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_cpu(dummy_img_cpu, dummy_coords_cpu)
        times_cpu.append(time.perf_counter() - t0)

    results["cpu_latency_ms"] = round(float(np.mean(times_cpu)) * 1000, 2)
    results["cpu_latency_std_ms"] = round(float(np.std(times_cpu)) * 1000, 2)
    results["cpu_fps"] = round(1.0 / float(np.mean(times_cpu)), 1)

    results["batch_size"] = 1
    results["img_size"] = img_size
    results["num_au"] = num_au
    results["n_runs"] = n_runs

    # Move model back to original device
    model.to(device)

    _save_json(results, out_dir / "latency_benchmark.json")


# ═══════════════ AU Attention Analysis ═══════════════

@torch.no_grad()
def save_au_attention_map(model, loader, device, class_names, out_dir,
                          region_names=None, max_samples=500):
    """
    Cross-attention에서 CLS→AU attention weight 추출.
    클래스별 평균 AU attention heatmap 생성.
    → 논문 Figure: "Which AU regions are most important for each emotion?"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        model.eval()

        # Hook으로 cross-attention weight 캡처
        attn_weights_all = []
        labels_all = []

        def _hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2:
                attn_weights_all.append(output[1].detach().cpu())

        # Register hook on cross-attention
        hooks = []
        for name, module in model.named_modules():
            if "cross_attn" in name and hasattr(module, "register_forward_hook"):
                hooks.append(module.register_forward_hook(_hook_fn))

        if not hooks:
            print("  [WARN] No cross_attn module found for hooking")
            return

        count = 0
        for batch in loader:
            if count >= max_samples:
                break
            images = batch["image"].to(device, non_blocking=True)
            au_coords = batch["au_coords"].to(device, non_blocking=True)
            labels = batch["label"]

            model(images, au_coords)
            labels_all.extend(labels.numpy())
            count += len(labels)

        for h in hooks:
            h.remove()

        if not attn_weights_all:
            print("  [WARN] No attention weights captured")
            return

        # attn_weights: [N_batches, B, n_heads, Q=2, K=num_au]
        # We want CLS token (Q=0) attention over AU tokens
        all_attn = torch.cat(attn_weights_all, dim=0)  # [N, heads, Q, K]
        # Average over heads, take CLS query (index 0)
        cls_attn = all_attn[:, :, 0, :].mean(dim=1).numpy()  # [N, K]
        labels_arr = np.array(labels_all[:cls_attn.shape[0]])

        num_classes = len(class_names)
        K = cls_attn.shape[1]
        if region_names is None:
            region_names = [f"AU_{i}" for i in range(K)]

        # Per-class average attention
        class_attn = np.zeros((num_classes, K))
        for c in range(num_classes):
            mask = labels_arr == c
            if mask.sum() > 0:
                class_attn[c] = cls_attn[mask].mean(axis=0)

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(8, K * 0.8), max(5, num_classes * 0.6)))
        im = ax.imshow(class_attn, cmap="YlOrRd", aspect="auto")
        fig.colorbar(im, ax=ax, label="Attention Weight")

        ax.set_xticks(range(K))
        ax.set_xticklabels(region_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_title("CLS → AU Region Attention (per emotion)", fontsize=13, fontweight="bold")
        ax.set_xlabel("AU Region")
        ax.set_ylabel("Emotion Class")

        for i in range(num_classes):
            for j in range(K):
                ax.text(j, i, f"{class_attn[i,j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if class_attn[i,j] > class_attn.max()*0.6 else "black")

        plt.tight_layout()
        fig.savefig(out_dir / "au_attention_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Also save as JSON for LaTeX
        attn_data = {}
        for c in range(num_classes):
            attn_data[class_names[c]] = {region_names[k]: round(float(class_attn[c, k]), 4)
                                          for k in range(K)}
        _save_json(attn_data, out_dir / "au_attention_per_class.json")

        print(f"  [+] au_attention_heatmap.png ✓")
        print(f"  [+] au_attention_per_class.json ✓")

    except Exception as e:
        print(f"  [WARN] AU attention map: {e}")


# ═══════════════ Legacy compatibility ═══════════════
# trainer.py에서 호출하는 기존 함수들 유지

def save_confusion_matrix(trues, preds, class_names, out_path):
    """Backward compatible: trainer.py에서 epoch마다 호출."""
    out_dir = Path(out_path).parent
    _save_confusion_matrices(np.array(trues), np.array(preds), class_names, out_dir)


def save_report(trues, preds, class_names, out_path):
    report = classification_report(trues, preds, target_names=class_names, digits=4)
    Path(out_path).write_text(report, encoding="utf-8")
    return report
