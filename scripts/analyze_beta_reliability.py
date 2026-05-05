#!/usr/bin/env python3
"""
Analyze per-AU Beta reliability map from a trained AUFERModelBeta checkpoint.

Outputs:
  - per_au_reliability.json : mean/std reliability per AU region, overall and per class
  - per_au_reliability.png  : grouped barplot (AU × class)
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

from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
from models.core.fer_model_beta import AUFERModelBeta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="best.pth from beta training")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max_batches", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = str(cfg["paths"]["train_csv"])
    val_csv = str(cfg["paths"]["val_csv"])
    csv_path = train_csv if args.split == "train" else val_csv

    label2id = build_label_mapping(train_csv)
    id2label = {v: k for k, v in label2id.items()}
    region_prefixes = find_region_prefixes(train_csv)
    num_au = len(region_prefixes)
    num_classes = len(label2id)

    mcfg = cfg["model"]
    bcfg = mcfg.get("beta", {}) or {}
    img_size = int(cfg["augmentation"].get("global_img_size", 224))

    model = AUFERModelBeta(
        use_l2=bool(bcfg.get("use_l2", True)),
        l2_mode=str(bcfg.get("l2_mode", "scale")),
        use_l3=bool(bcfg.get("use_l3", False)),
        tau_init=float(bcfg.get("tau_init", 2.0)),
        backbone_name=str(mcfg.get("backbone", "mobilevitv2_100")),
        pretrained=False,
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
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = model.backbone.norm_mean
    std = model.backbone.norm_std
    ds = AUFERDataset(csv_path, label2id, region_prefixes, img_size=img_size,
                      mean=mean, std=std, is_train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=collate_fn)

    rel_buf = []          # [N, K]
    label_buf = []        # [N]
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            au_coords = batch["au_coords"].to(device, non_blocking=True)
            _ = model(images, au_coords)
            r = model.get_last_reliability()["l2_per_token"]
            rel_buf.append(r.cpu().numpy())
            label_buf.append(batch["label"].cpu().numpy())

    rel = np.concatenate(rel_buf, axis=0)            # [N, K]
    labels = np.concatenate(label_buf, axis=0)       # [N]

    out = {
        "regions": region_prefixes,
        "overall_mean": rel.mean(axis=0).tolist(),
        "overall_std": rel.std(axis=0).tolist(),
        "per_class": {},
    }
    for c in range(num_classes):
        m = labels == c
        if m.any():
            out["per_class"][id2label[c]] = {
                "mean": rel[m].mean(axis=0).tolist(),
                "std": rel[m].std(axis=0).tolist(),
                "n": int(m.sum()),
            }

    out_dir = Path(cfg["paths"]["output_dir"]) / "beta_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / "per_au_reliability.json", "w"), indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    K = len(region_prefixes)
    classes = list(out["per_class"].keys())
    width = 0.8 / max(1, len(classes))
    x = np.arange(K)
    for i, c in enumerate(classes):
        ax.bar(x + i * width, out["per_class"][c]["mean"], width, label=c)
    ax.set_xticks(x + width * (len(classes) - 1) / 2)
    ax.set_xticklabels(region_prefixes, rotation=30, ha="right")
    ax.set_ylabel("Beta reliability r = α/(α+β)")
    ax.set_title("Per-AU reliability by class (L2 gate)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "per_au_reliability.png", dpi=200)
    print(f"[saved] {out_dir}/per_au_reliability.json + .png")
    print(f"Overall mean r: {dict(zip(region_prefixes, [f'{v:.3f}' for v in out['overall_mean']]))}")


if __name__ == "__main__":
    main()
