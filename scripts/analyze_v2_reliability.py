#!/usr/bin/env python3
"""
Analyze per-AU reliability from a trained AURegionFormerV2 checkpoint.

For DualBetaGate, dumps both r_iso, r_rel (when applicable) and the combined r,
plus the learnable mix weight if combine='weighted'.

Outputs (under {output_dir}/beta_analysis/):
  - per_au_reliability.json : means/stds for r_combined, r_iso, r_rel per AU and per class
  - per_au_reliability.png  : grouped barplot of combined r per AU × class
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from data.dataset import build_label_mapping, find_region_prefixes
from data.dataset_v2 import AUFERDatasetV2, collate_fn_v2
from models.core.fer_model_v2 import AURegionFormerV2


def build_model_from_cfg(cfg, num_au, num_classes, img_size, device):
    mcfg = cfg["model"]
    v2cfg = mcfg.get("v2", {}) or {}
    return AURegionFormerV2(
        global_encoder=str(v2cfg.get("global_encoder", "mobilenetv4_conv_medium")),
        patch_encoder=str(v2cfg.get("patch_encoder", "mobilenetv4_conv_small")),
        pretrained=False,
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
        tau_init=float(v2cfg.get("tau_init", 1.0)),
        n_fusion_layers=int(mcfg.get("n_fusion_layers", 1)),
        n_fusion_heads=int(mcfg.get("n_heads", 8)),
        dropout=float(mcfg.get("dropout", 0.1)),
        head_dropout=float(mcfg.get("head_dropout", 0.2)),
        gate_init=float(mcfg.get("gate_init", 0.0)),
        drop_path=float(mcfg.get("drop_path", 0.0)),
        use_au_pool=bool(mcfg.get("use_au_pool", False)),
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max_batches", type=int, default=1000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = str(cfg["paths"]["train_csv"])
    val_csv = str(cfg["paths"]["val_csv"])
    csv_path = train_csv if args.split == "train" else val_csv

    label2id = build_label_mapping(train_csv)
    id2label = {v: k for k, v in label2id.items()}
    regions = find_region_prefixes(train_csv)
    num_au = len(regions)
    num_classes = len(label2id)
    img_size = int(cfg["augmentation"].get("global_img_size", 224))

    model = build_model_from_cfg(cfg, num_au, num_classes, img_size, device)
    if not model.use_stage_b:
        print("[WARN] config has use_stage_b=false → no Beta gate to analyze. Exit.")
        return
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    patch_size = model.patch_size
    ds = AUFERDatasetV2(
        csv_path, label2id, regions,
        img_size=img_size, patch_size=patch_size,
        mean=model.norm_mean, std=model.norm_std,
        patch_mean=model.patch_norm_mean, patch_std=model.patch_norm_std,
        is_train=False,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=collate_fn_v2)

    r_buf, riso_buf, rrel_buf, label_buf = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            au_patches = batch["au_patches"].to(device, non_blocking=True)
            _ = model(images, au_patches)
            rel = model.get_last_reliability()
            r_buf.append(rel["r_combined"].cpu().numpy())
            aux = rel["aux"] or {}
            if "r_iso" in aux:
                riso_buf.append(aux["r_iso"].cpu().numpy())
            if "r_rel" in aux:
                rrel_buf.append(aux["r_rel"].cpu().numpy())
            label_buf.append(batch["label"].cpu().numpy())

    r = np.concatenate(r_buf, axis=0)
    labels = np.concatenate(label_buf, axis=0)
    r_iso = np.concatenate(riso_buf, axis=0) if riso_buf else None
    r_rel = np.concatenate(rrel_buf, axis=0) if rrel_buf else None

    out = {
        "regions": regions,
        "n_eval": int(len(labels)),
        "combine": cfg["model"]["v2"].get("combine", "weighted"),
        "overall": {
            "r_combined_mean": r.mean(axis=0).tolist(),
            "r_combined_std": r.std(axis=0).tolist(),
        },
        "per_class": {},
    }
    if r_iso is not None:
        out["overall"]["r_iso_mean"] = r_iso.mean(axis=0).tolist()
    if r_rel is not None:
        out["overall"]["r_rel_mean"] = r_rel.mean(axis=0).tolist()

    for c in range(num_classes):
        m = labels == c
        if m.any():
            cls_entry = {
                "r_combined_mean": r[m].mean(axis=0).tolist(),
                "n": int(m.sum()),
            }
            if r_iso is not None:
                cls_entry["r_iso_mean"] = r_iso[m].mean(axis=0).tolist()
            if r_rel is not None:
                cls_entry["r_rel_mean"] = r_rel[m].mean(axis=0).tolist()
            out["per_class"][id2label[c]] = cls_entry

    out_dir = Path(cfg["paths"]["output_dir"]) / "beta_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / "per_au_reliability.json", "w"), indent=2)

    # Plot combined r per AU × class
    K = len(regions)
    classes = list(out["per_class"].keys())
    width = 0.8 / max(1, len(classes))
    x = np.arange(K)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cls in enumerate(classes):
        ax.bar(x + i * width, out["per_class"][cls]["r_combined_mean"], width, label=cls)
    ax.set_xticks(x + width * (len(classes) - 1) / 2)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel("Combined reliability r")
    ax.set_title(f"Per-AU reliability by class ({out['combine']})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "per_au_reliability.png", dpi=200)
    print(f"[saved] {out_dir}/per_au_reliability.json + .png")
    print(f"Overall r_combined: {dict(zip(regions, [f'{v:.3f}' for v in out['overall']['r_combined_mean']]))}")


if __name__ == "__main__":
    main()
