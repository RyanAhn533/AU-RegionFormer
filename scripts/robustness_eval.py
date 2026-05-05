#!/usr/bin/env python3
"""
Robustness eval: compare base.pth vs Beta.pth on val set under perturbations.

Perturbations:
  - clean: no change (control)
  - drop_<region>: replace that AU's token with zeros (after au_roi extract)
  - noise_<sigma>: add N(0, sigma^2) px noise to all AU coords (clamped)

Output:
  - robustness_results.json
  - robustness_plot.png : F1 per perturbation, base vs Beta side-by-side
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
from models.core.fer_model import AUFERModel
from models.core.fer_model_beta import AUFERModelBeta


def build_model(cfg, num_au, num_classes, img_size, device):
    mcfg = cfg["model"]
    common = dict(
        backbone_name=str(mcfg.get("backbone", "mobilevitv2_100")),
        pretrained=False,
        num_au=num_au, num_classes=num_classes,
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
    if mcfg.get("use_beta", False):
        bcfg = mcfg.get("beta", {}) or {}
        return AUFERModelBeta(
            use_l2=bool(bcfg.get("use_l2", True)),
            l2_mode=str(bcfg.get("l2_mode", "softmax")),
            use_l3=bool(bcfg.get("use_l3", False)),
            tau_init=float(bcfg.get("tau_init", 1.0)),
            **common,
        ).to(device)
    return AUFERModel(**common).to(device)


def patch_drop_region(model, region_idx):
    """Hook au_roi forward to zero out one region's token output.
    Returns a remove() function to undo."""
    orig = model.au_roi.forward
    def hooked(feat_map, au_coords, img_size):
        out = orig(feat_map, au_coords, img_size)
        # out: [B, K, D]
        out = out.clone()
        out[:, region_idx, :] = 0
        return out
    model.au_roi.forward = hooked
    def remove():
        model.au_roi.forward = orig
    return remove


def perturb_eval(model, loader, device, perturb=None):
    """perturb: None | ('drop', region_idx) | ('noise', sigma)"""
    remove = None
    if perturb and perturb[0] == "drop":
        remove = patch_drop_region(model, perturb[1])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            au = batch["au_coords"].to(device, non_blocking=True)
            if perturb and perturb[0] == "noise":
                sigma = perturb[1]
                au = au + torch.randn_like(au) * sigma
                au = au.clamp(0, model.img_size - 1)
            logits = model(images, au)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(batch["label"].cpu().numpy())
    if remove:
        remove()
    return {
        "f1": float(f1_score(trues, preds, average="macro")),
        "acc": float(accuracy_score(trues, preds)),
    }


def run_one_model(cfg_path, ckpt_path, regions, loader, device):
    cfg = yaml.safe_load(open(cfg_path))
    img_size = int(cfg["augmentation"].get("global_img_size", 224))
    num_au = len(regions)
    num_classes = cfg["model"]["num_classes"]
    model = build_model(cfg, num_au, num_classes, img_size, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    results = {}
    print(f"\n[{ckpt_path}]")
    res = perturb_eval(model, loader, device, None)
    print(f"  clean         : F1={res['f1']:.4f}  acc={res['acc']:.4f}")
    results["clean"] = res
    for i, r in enumerate(regions):
        res = perturb_eval(model, loader, device, ("drop", i))
        print(f"  drop_{r:14s}: F1={res['f1']:.4f}  acc={res['acc']:.4f}  Δ={res['f1']-results['clean']['f1']:+.4f}")
        results[f"drop_{r}"] = res
    for sigma in [5.0, 10.0, 20.0]:
        torch.manual_seed(0)  # deterministic noise
        res = perturb_eval(model, loader, device, ("noise", sigma))
        print(f"  noise_sig{sigma:>4.0f}    : F1={res['f1']:.4f}  acc={res['acc']:.4f}  Δ={res['f1']-results['clean']['f1']:+.4f}")
        results[f"noise_{int(sigma)}"] = res
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--beta_cfg", required=True)
    ap.add_argument("--beta_ckpt", required=True)
    ap.add_argument("--out_dir", default="/mnt/hdd/ajy_25/results/robustness_4c")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build val loader once (shared)
    base_cfg = yaml.safe_load(open(args.base_cfg))
    train_csv = str(base_cfg["paths"]["train_csv"])
    val_csv = str(base_cfg["paths"]["val_csv"])
    label2id = build_label_mapping(train_csv)
    regions = find_region_prefixes(train_csv)
    img_size = int(base_cfg["augmentation"].get("global_img_size", 224))

    # Use a temporary model just for normalization stats
    tmp = AUFERModel(num_au=len(regions), num_classes=base_cfg["model"]["num_classes"],
                     pretrained=True, img_size=img_size,
                     backbone_name=str(base_cfg["model"].get("backbone","mobilevitv2_100")))
    mean, std = tmp.backbone.norm_mean, tmp.backbone.norm_std
    del tmp; torch.cuda.empty_cache()

    val_set = AUFERDataset(val_csv, label2id, regions, img_size=img_size,
                           mean=mean, std=std, is_train=False)
    val_loader = DataLoader(val_set, batch_size=192, shuffle=False, num_workers=8,
                            pin_memory=True, collate_fn=collate_fn,
                            persistent_workers=True, prefetch_factor=4)
    print(f"val_set={len(val_set)}, regions={regions}")

    results = {
        "regions": regions,
        "base": run_one_model(args.base_cfg, args.base_ckpt, regions, val_loader, device),
        "beta": run_one_model(args.beta_cfg, args.beta_ckpt, regions, val_loader, device),
    }
    with open(out_dir / "robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    keys = ["clean"] + [f"drop_{r}" for r in regions] + [f"noise_{s}" for s in [5,10,20]]
    base_f1 = [results["base"][k]["f1"] for k in keys]
    beta_f1 = [results["beta"][k]["f1"] for k in keys]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(keys)); w = 0.4
    ax.bar(x - w/2, base_f1, w, label="Base", color="#888")
    ax.bar(x + w/2, beta_f1, w, label="Beta L2", color="#1976d2")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 macro")
    ax.set_title("Robustness under AU perturbation: Base vs Beta L2")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(min(base_f1), min(beta_f1)) - 0.02, max(max(base_f1), max(beta_f1)) + 0.01)
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_plot.png", dpi=200)
    print(f"\n[saved] {out_dir}/robustness_results.json + .png")

    # Summary table
    print(f"\n{'perturbation':<22s} | {'Base F1':>8s} | {'Beta F1':>8s} | {'Δ(Beta-Base)':>14s}")
    print("-" * 60)
    for k in keys:
        b = results["base"][k]["f1"]; t = results["beta"][k]["f1"]
        print(f"{k:<22s} | {b:>8.4f} | {t:>8.4f} | {t-b:>+14.4f}")


if __name__ == "__main__":
    main()
