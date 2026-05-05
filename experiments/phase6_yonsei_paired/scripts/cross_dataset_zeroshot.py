#!/usr/bin/env python3
"""
Cross-dataset zero-shot eval: Stage N best.pth → arbitrary master-format CSV.

Outputs per-class F1, accuracy, Beta r per class+region.
Useful for Korean (master_val) vs Western (sfew_val_4c, etc.) comparison.

Usage:
  python cross_dataset_zeroshot.py \
      --config phase6/configs/stage6_full.yaml \
      --ckpt /mnt/hdd/.../phase6_stage6_full/best.pth \
      --val_csv phase6/csvs/sfew_val_4c.csv \
      --out_dir phase6/results/crossdataset_sfew_val
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root / "src"))

from data.dataset import build_label_mapping, find_region_prefixes
from data.dataset_v2 import AUFERDatasetV2, collate_fn_v2
from models.core.fer_model_v2 import AURegionFormerV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--val_csv", required=True, help="target val CSV (master format)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_csv = str(cfg["paths"]["train_csv"])
    label2id = build_label_mapping(train_csv)
    id2label = {v: k for k, v in label2id.items()}
    regions  = find_region_prefixes(train_csv)
    img_size = int(cfg["augmentation"].get("global_img_size", 224))
    num_au = len(regions)
    num_classes = cfg["model"]["num_classes"]

    # build same model
    mcfg = cfg["model"]
    v2 = mcfg.get("v2", {}) or {}
    model = AURegionFormerV2(
        global_encoder=str(v2.get("global_encoder", "mobilenetv4_conv_medium")),
        patch_encoder=str(v2.get("patch_encoder", "mobilenetv4_conv_small")),
        pretrained=False,
        num_au=num_au, num_classes=num_classes,
        d_emb=int(mcfg.get("d_emb", 384)),
        patch_size=int(v2.get("patch_size", 96)),
        img_size=img_size,
        use_stage_a=bool(v2.get("use_stage_a", True)),
        stage_a_layers=int(v2.get("stage_a_layers", 1)),
        stage_a_heads=int(v2.get("stage_a_heads", 8)),
        use_stage_b=bool(v2.get("use_stage_b", True)),
        use_iso=bool(v2.get("use_iso", True)),
        use_rel=bool(v2.get("use_rel", True)),
        combine=str(v2.get("combine", "weighted")),
        tau_init=float(v2.get("tau_init", 1.0)),
        n_fusion_layers=int(mcfg.get("n_fusion_layers", 1)),
        n_fusion_heads=int(mcfg.get("n_heads", 8)),
        dropout=float(mcfg.get("dropout", 0.1)),
        head_dropout=float(mcfg.get("head_dropout", 0.2)),
        gate_init=float(mcfg.get("gate_init", 0.0)),
        drop_path=float(mcfg.get("drop_path", 0.0)),
        use_au_pool=bool(mcfg.get("use_au_pool", False)),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()

    # build val dataset on the new CSV
    patch_size = model.patch_size
    val = AUFERDatasetV2(args.val_csv, label2id, regions,
                         img_size=img_size, patch_size=patch_size,
                         mean=model.norm_mean, std=model.norm_std,
                         patch_mean=model.patch_norm_mean, patch_std=model.patch_norm_std,
                         is_train=False)
    print(f"target val: {args.val_csv} — {len(val)} samples")
    loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=collate_fn_v2)

    all_preds, all_labels, all_r = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device); au = batch["au_patches"].to(device)
            logits = model(img, au)
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
            rel = model.get_last_reliability().get("r_combined")
            if rel is not None:
                all_r.append(rel.cpu().numpy())
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    all_r = np.concatenate(all_r) if all_r else None

    # metrics
    F1 = float(f1_score(all_labels, all_preds, average="macro"))
    ACC = float(accuracy_score(all_labels, all_preds))
    cls_names = [id2label[i] for i in range(num_classes)]
    rep = classification_report(all_labels, all_preds, labels=list(range(num_classes)),
                                target_names=cls_names, zero_division=0, output_dict=True)
    summary = {
        "ckpt": str(args.ckpt),
        "val_csv": str(args.val_csv),
        "n_samples": int(len(all_labels)),
        "f1_macro": F1,
        "accuracy": ACC,
        "per_class": {cls_names[i]: rep.get(cls_names[i], {}) for i in range(num_classes)},
    }
    if all_r is not None:
        summary["beta_r_mean_per_class"] = {}
        for c in range(num_classes):
            m = all_labels == c
            if m.any():
                summary["beta_r_mean_per_class"][cls_names[c]] = {
                    "mean": float(all_r[m].mean()),
                    "per_au": all_r[m].mean(axis=0).tolist(),
                    "n": int(m.sum()),
                }
        summary["regions"] = regions

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"crossdataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[summary]")
    print(f"F1 macro: {F1:.4f}  Acc: {ACC:.4f}  N: {len(all_labels)}")
    print(f"saved → {out_dir}/crossdataset_summary.json")


if __name__ == "__main__":
    main()
