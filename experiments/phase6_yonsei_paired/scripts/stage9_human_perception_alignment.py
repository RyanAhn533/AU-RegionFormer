#!/usr/bin/env python3
"""
Stage 9++ — Human Perception Alignment Analysis Pipeline.

Outputs C1~C8 figures + JSON stats for paper main claims.
Runs on Stage N best.pth (default: stage6_full).

C1. per-class Beta r ↔ per-class Yonsei reject rate (Spearman, n=4)
C2. per-image Beta r ↔ mean_is_selected (Pearson, n=val)
C3. consensus-agree vs consensus-reject vs split — model entropy boxplot + t-test
C4. per-AU per-class Beta r heatmap (4x8 = Korean canonical AU map)
C5. Human accept_prob (1 - mean_reject) ↔ model max-softmax (calibration scatter)
C6. Random global test — global token shuffle → r alignment break check
C7. AU shuffle test — AU patch order shuffle → r alignment break check
C8. Per-class perceptual confusion matrix (model + 'human-implicit' alt distribution)
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr, pearsonr, ttest_ind
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root / "src"))

from data.dataset import build_label_mapping, find_region_prefixes
from data.dataset_v2 import AUFERDatasetV2, collate_fn_v2
from models.core.fer_model_v2 import AURegionFormerV2


def build_model(cfg, num_au, num_classes, img_size, device):
    mcfg = cfg["model"]
    v2 = mcfg.get("v2", {}) or {}
    return AURegionFormerV2(
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


@torch.no_grad()
def collect(model, loader, device, perturb=None):
    """Run inference, return per-image arrays.

    perturb: None | 'random_global' | 'au_shuffle'
    """
    model.eval()
    all_logits, all_r, all_riso, all_rrel = [], [], [], []
    all_labels, all_paths = [], []
    all_qual, all_meanrej, all_nrater, all_isel = [], [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        au  = batch["au_patches"].to(device)
        lab = batch["label"].to(device)

        if perturb == "au_shuffle":
            # Shuffle AU patch order per sample (break AU spatial identity)
            B, K = au.shape[:2]
            for b in range(B):
                idx = torch.randperm(K, device=device)
                au[b] = au[b, idx]

        # If perturb 'random_global', we need to inject random global feature.
        # Easiest: hook global_proj output to zeros or shuffle.
        if perturb == "random_global":
            # Forward but replace global_feat with shuffled across batch
            # We do this by intercepting — easier: feed shuffled image globally
            shuf = torch.randperm(img.shape[0], device=device)
            img = img[shuf]   # global feature derived from shuffled images
            # AU patches kept original

        logits, _ = model(img, au, return_features=True)
        rel = model.get_last_reliability()
        all_logits.append(logits.cpu())
        if rel.get("r_combined") is not None:
            all_r.append(rel["r_combined"].cpu())
        aux = rel.get("aux") or {}
        if "r_iso" in aux:  all_riso.append(aux["r_iso"].cpu())
        if "r_rel" in aux:  all_rrel.append(aux["r_rel"].cpu())
        all_labels.append(lab.cpu())
        all_paths.extend(batch["path"])
        all_qual.append(batch["quality_score"])
        # Need to read is_selected/n_raters/mean_is_selected from CSV (not in batch)
    logits = torch.cat(all_logits).numpy()
    probs  = torch.softmax(torch.cat(all_logits), dim=-1).numpy()
    r      = torch.cat(all_r).numpy()  if all_r  else None
    riso   = torch.cat(all_riso).numpy() if all_riso else None
    rrel   = torch.cat(all_rrel).numpy() if all_rrel else None
    labels = torch.cat(all_labels).numpy()
    qual   = torch.cat(all_qual).numpy()
    return dict(logits=logits, probs=probs, r=r, r_iso=riso, r_rel=rrel,
                labels=labels, paths=all_paths, quality_score=qual)


def join_meta(out, master_val_path):
    df = pd.read_csv(master_val_path,
                     usecols=["path", "is_selected", "n_raters", "mean_is_selected"])
    df_out = pd.DataFrame({"path": out["paths"]})
    df = df_out.merge(df, on="path", how="left")
    out["is_selected"]      = df["is_selected"].values
    out["n_raters"]         = df["n_raters"].values
    out["mean_is_selected"] = df["mean_is_selected"].values
    return out


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1)
    return -(p * np.log(p)).sum(axis=-1)


def run_analysis(out, id2label, region_prefixes, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(id2label)
    K = out["r"].shape[1] if out.get("r") is not None else 0
    classes = [id2label[i] for i in range(n_classes)]

    # ── Per-class summary ──
    cls_stats = {}
    for c, name in enumerate(classes):
        m = out["labels"] == c
        if not m.any(): continue
        cls_stats[name] = {
            "n": int(m.sum()),
            "yonsei_reject_rate": float(np.nanmean(out["is_selected"][m])),
            "yonsei_mean_reject": float(np.nanmean(out["mean_is_selected"][m])),
            "model_acc": float(((out["probs"].argmax(axis=-1) == c) & m).sum() / max(m.sum(), 1)),
            "model_max_softmax_mean": float(out["probs"][m].max(axis=-1).mean()),
            "model_entropy_mean": float(entropy(out["probs"][m]).mean()),
            "beta_r_mean": float(np.nanmean(out["r"][m])) if out.get("r") is not None else None,
            "beta_r_per_au":     out["r"][m].mean(axis=0).tolist() if out.get("r") is not None else None,
            "beta_riso_per_au": out["r_iso"][m].mean(axis=0).tolist() if out.get("r_iso") is not None else None,
            "beta_rrel_per_au": out["r_rel"][m].mean(axis=0).tolist() if out.get("r_rel") is not None else None,
        }

    # ── C1 per-class Spearman ──
    c1 = {}
    if out.get("r") is not None:
        x = [cls_stats[c]["yonsei_reject_rate"] for c in classes]
        y = [cls_stats[c]["beta_r_mean"]        for c in classes]
        rho, p = spearmanr(x, y)
        c1 = {"yonsei_reject": x, "beta_r_mean": y, "spearman_rho": float(rho), "spearman_p": float(p)}
        # plot
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(x, y, s=80)
        for i,c in enumerate(classes):
            ax.annotate(c, (x[i], y[i]), fontsize=10)
        ax.set_xlabel("Yonsei reject rate (per class)")
        ax.set_ylabel("Beta r (model, per class mean)")
        ax.set_title(f"C1 per-class alignment  Spearman={rho:.3f} p={p:.3f}")
        plt.tight_layout(); plt.savefig(out_dir/"C1_per_class_spearman.png", dpi=200); plt.close()

    # ── C2 per-image Pearson ──
    c2 = {}
    if out.get("r") is not None:
        r_img = out["r"].mean(axis=1)               # [N]
        m_img = out["mean_is_selected"].astype(float)
        valid = ~np.isnan(m_img)
        rp, p = pearsonr(r_img[valid], m_img[valid])
        c2 = {"pearson_r": float(rp), "pearson_p": float(p), "n_valid": int(valid.sum())}
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hexbin(m_img[valid], r_img[valid], gridsize=40, bins="log", cmap="Blues")
        ax.set_xlabel("Yonsei mean_is_selected (per image)")
        ax.set_ylabel("Beta r mean (model, per image)")
        ax.set_title(f"C2 per-image alignment  Pearson={rp:.3f}  n={valid.sum()}")
        plt.tight_layout(); plt.savefig(out_dir/"C2_per_image_pearson.png", dpi=200); plt.close()

    # ── C3 consensus subset entropy comparison ──
    c3 = {}
    nr = np.nan_to_num(out["n_raters"], nan=0).astype(int)
    isel = np.nan_to_num(out["is_selected"], nan=-1).astype(int)
    mrej = np.nan_to_num(out["mean_is_selected"], nan=-1)
    cons_agree  = (nr >= 2) & (mrej == 0.0)
    cons_reject = (nr >= 2) & (mrej == 1.0)
    split       = (nr >= 2) & (mrej > 0.0) & (mrej < 1.0)
    H = entropy(out["probs"])
    c3 = {
        "consensus_agree":  {"n": int(cons_agree.sum()),  "model_entropy_mean": float(H[cons_agree].mean()  if cons_agree.any() else 0)},
        "consensus_reject": {"n": int(cons_reject.sum()), "model_entropy_mean": float(H[cons_reject].mean() if cons_reject.any() else 0)},
        "split":            {"n": int(split.sum()),       "model_entropy_mean": float(H[split].mean()       if split.any() else 0)},
    }
    if cons_agree.any() and split.any():
        t, p = ttest_ind(H[split], H[cons_agree], equal_var=False)
        c3["t_split_vs_agree"] = {"t": float(t), "p": float(p)}
    fig, ax = plt.subplots(figsize=(5,4))
    data = [H[cons_agree] if cons_agree.any() else [], H[split] if split.any() else [], H[cons_reject] if cons_reject.any() else []]
    ax.boxplot(data, labels=["consensus-agree", "split", "consensus-reject"])
    ax.set_ylabel("Model predictive entropy")
    ax.set_title("C3 model uncertainty by Yonsei consensus class")
    plt.tight_layout(); plt.savefig(out_dir/"C3_entropy_by_consensus.png", dpi=200); plt.close()

    # ── C4 per-AU per-class heatmap ──
    c4 = {}
    if out.get("r") is not None and K > 0:
        H4x8 = np.zeros((n_classes, K))
        for c in range(n_classes):
            m = out["labels"] == c
            if m.any(): H4x8[c] = out["r"][m].mean(axis=0)
        c4 = {"matrix": H4x8.tolist(), "regions": region_prefixes, "classes": classes}
        fig, ax = plt.subplots(figsize=(8,4))
        im = ax.imshow(H4x8, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_xticklabels(region_prefixes, rotation=30, ha="right")
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
        for c in range(n_classes):
            for k in range(K):
                ax.text(k, c, f"{H4x8[c,k]:.2f}", ha="center", va="center",
                        color="white" if H4x8[c,k]<0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title("C4 per-AU per-class Beta r (Korean canonical map)")
        plt.tight_layout(); plt.savefig(out_dir/"C4_per_au_class_heatmap.png", dpi=200); plt.close()

    # ── C5 calibration: human accept ↔ model max-softmax ──
    accept = 1.0 - np.nan_to_num(out["mean_is_selected"], nan=0.0)  # 1 - mean_reject for intended class
    msoft = out["probs"].max(axis=-1)
    valid = ~np.isnan(out["mean_is_selected"])
    c5_p, c5_pp = pearsonr(accept[valid], msoft[valid])
    c5 = {"pearson_r": float(c5_p), "pearson_p": float(c5_pp), "n_valid": int(valid.sum())}
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hexbin(accept[valid], msoft[valid], gridsize=40, bins="log", cmap="Greens")
    ax.set_xlabel("Human accept_prob (1 - mean_reject)")
    ax.set_ylabel("Model max-softmax")
    ax.set_title(f"C5 calibration  Pearson={c5_p:.3f}  n={valid.sum()}")
    plt.tight_layout(); plt.savefig(out_dir/"C5_calibration.png", dpi=200); plt.close()

    # ── package all ──
    summary = {
        "per_class": cls_stats,
        "C1_per_class_spearman": c1,
        "C2_per_image_pearson": c2,
        "C3_consensus_entropy": c3,
        "C4_per_au_class_heatmap": c4,
        "C5_human_accept_vs_softmax": c5,
    }
    with open(out_dir/"stage9_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_dir}/stage9_summary.json + 5 PNG figures")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_batches", type=int, default=10000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_csv = str(cfg["paths"]["val_csv"])
    train_csv = str(cfg["paths"]["train_csv"])
    label2id = build_label_mapping(train_csv)
    id2label = {v: k for k, v in label2id.items()}
    regions = find_region_prefixes(train_csv)
    img_size = int(cfg["augmentation"].get("global_img_size", 224))
    num_au = len(regions)
    num_classes = cfg["model"]["num_classes"]

    model = build_model(cfg, num_au, num_classes, img_size, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    patch_size = model.patch_size
    val = AUFERDatasetV2(val_csv, label2id, regions,
                         img_size=img_size, patch_size=patch_size,
                         mean=model.norm_mean, std=model.norm_std,
                         patch_mean=model.patch_norm_mean, patch_std=model.patch_norm_std,
                         is_train=False)
    loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=8,
                        pin_memory=True, collate_fn=collate_fn_v2)

    print(f"[1/4] Collecting clean inference ({len(val)} samples)")
    out = collect(model, loader, device, perturb=None)
    out = join_meta(out, val_csv)
    summary_clean = run_analysis(out, id2label, regions, args.out_dir)

    print(f"[2/4] Random global perturbation")
    out_rg = collect(model, loader, device, perturb="random_global")
    out_rg = join_meta(out_rg, val_csv)
    summary_rg = run_analysis(out_rg, id2label, regions, Path(args.out_dir)/"perturb_random_global")

    print(f"[3/4] AU shuffle perturbation")
    out_au = collect(model, loader, device, perturb="au_shuffle")
    out_au = join_meta(out_au, val_csv)
    summary_au = run_analysis(out_au, id2label, regions, Path(args.out_dir)/"perturb_au_shuffle")

    # ── Compare alignment break ──
    print(f"[4/4] Alignment break comparison")
    breakdown = {
        "clean":         {"C1_rho": summary_clean["C1_per_class_spearman"].get("spearman_rho"),
                          "C2_r":  summary_clean["C2_per_image_pearson"].get("pearson_r"),
                          "C5_r":  summary_clean["C5_human_accept_vs_softmax"].get("pearson_r")},
        "random_global": {"C1_rho": summary_rg["C1_per_class_spearman"].get("spearman_rho"),
                          "C2_r":  summary_rg["C2_per_image_pearson"].get("pearson_r"),
                          "C5_r":  summary_rg["C5_human_accept_vs_softmax"].get("pearson_r")},
        "au_shuffle":    {"C1_rho": summary_au["C1_per_class_spearman"].get("spearman_rho"),
                          "C2_r":  summary_au["C2_per_image_pearson"].get("pearson_r"),
                          "C5_r":  summary_au["C5_human_accept_vs_softmax"].get("pearson_r")},
    }
    with open(Path(args.out_dir)/"alignment_break_summary.json", "w") as f:
        json.dump(breakdown, f, indent=2)
    print(f"\nAlignment break summary: {json.dumps(breakdown, indent=2)}")
    print(f"\nDONE. all outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
