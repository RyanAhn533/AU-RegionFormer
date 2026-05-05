"""
Phase 0.2: OpenGraphAU 41-AU intensity extraction for 237K Korean images.

Output:
  data/label_quality/au_features/opengraphau_41au_237k.parquet
    columns: image_path, emotion, is_selected, AU1, AU2, ..., AU39,
             AUL1, AUR1, ..., AUR14  (41 total)
"""
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# OpenGraphAU 모듈 import 경로
OGAU_ROOT = Path(__file__).resolve().parents[3] / "libs" / "OpenGraphAU"
sys.path.insert(0, str(OGAU_ROOT))
# resnet.py가 relative path 'pretrain_models/...' 사용하므로 CWD를 맞춰야 함
os.chdir(str(OGAU_ROOT))

from model.ANFL import MEFARG
from utils import load_state_dict, image_eval

# 41 AU names (main 27 + sub 14)
AU_NAMES = [
    # Main (27)
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU11","AU12",
    "AU13","AU14","AU15","AU16","AU17","AU18","AU19","AU20","AU22","AU23",
    "AU24","AU25","AU26","AU27","AU32","AU38","AU39",
    # Sub (14) — L/R variants
    "AUL1","AUR1","AUL2","AUR2","AUL4","AUR4","AUL6","AUR6",
    "AUL10","AUR10","AUL12","AUR12","AUL14","AUR14",
]


class ImageListDataset(Dataset):
    def __init__(self, paths, image_root, transform):
        self.paths = paths
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        full = p if os.path.isabs(p) else str(self.image_root / p)
        try:
            img = Image.open(full).convert("RGB")
            img = self.transform(img)
            return img, idx, 1  # success
        except Exception as e:
            # return zero image to preserve index
            return torch.zeros(3, 224, 224), idx, 0


EMOTION_KO_TO_EN = {
    "기쁨": "happy",
    "분노": "angry",
    "슬픔": "sad",
    "중립": "neutral",
    "불안": "anxious",
    "상처": "hurt",
    "당황": "surprised",
}


def find_image_path(path_in_meta, emotion_ko, image_roots):
    """meta CSV의 path에서 basename 추출 + emotion 영어 매핑 후 data_processed_korea에서 찾기."""
    p = Path(path_in_meta)
    if p.exists():
        return str(p)
    basename = p.name
    emotion_en = EMOTION_KO_TO_EN.get(emotion_ko)
    if emotion_en is None:
        return None
    for root in image_roots:
        cand = Path(root) / emotion_en / basename
        if cand.exists():
            return str(cand)
    return None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    # --- Load model ---
    print("[info] loading OpenGraphAU MEFARG-ResNet50 stage1...")
    net = MEFARG(
        num_main_classes=27, num_sub_classes=14,
        backbone="resnet50", neighbor_num=4, metric="dots"
    )
    net = load_state_dict(net, args.checkpoint)
    net = net.to(device).eval()
    transform = image_eval()

    # --- Load meta ---
    print(f"[info] loading meta: {args.meta_csv}")
    meta = pd.read_csv(args.meta_csv)
    if args.limit:
        meta = meta.head(args.limit).copy()
    print(f"[info] total samples: {len(meta)}")

    # --- Resolve image paths ---
    image_roots = [
        args.image_root,
        "/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea",
        "/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea_validation",
    ]
    print("[info] resolving paths...")
    resolved = []
    missing = 0
    for p, emo in tqdm(zip(meta["path"].tolist(), meta["emotion"].tolist()),
                       total=len(meta), desc="resolve"):
        rp = find_image_path(p, emo, image_roots)
        if rp is None:
            missing += 1
        resolved.append(rp)
    meta["resolved_path"] = resolved
    print(f"[info] missing files: {missing}/{len(meta)}")
    meta_ok = meta[meta["resolved_path"].notna()].reset_index(drop=True)
    print(f"[info] usable: {len(meta_ok)}")

    # --- Dataset/loader ---
    ds = ImageListDataset(
        paths=meta_ok["resolved_path"].tolist(),
        image_root=args.image_root,
        transform=transform,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Inference ---
    n = len(ds)
    all_preds = np.full((n, 41), np.nan, dtype=np.float32)
    all_ok = np.zeros(n, dtype=np.int8)

    # Save checkpoint every N batches
    ckpt_every = args.ckpt_every
    ckpt_path = Path(args.output).with_suffix(".partial.npz")

    with torch.no_grad():
        for batch_idx, (imgs, idxs, ok) in enumerate(tqdm(dl, desc="infer")):
            imgs = imgs.to(device, non_blocking=True)
            preds = net(imgs).cpu().numpy()  # (B, 41)
            idxs = idxs.numpy()
            ok = ok.numpy()
            all_preds[idxs] = preds
            all_ok[idxs] = ok

            if (batch_idx + 1) % ckpt_every == 0:
                np.savez(ckpt_path, preds=all_preds, ok=all_ok,
                         last_batch=batch_idx)

    # --- Save final ---
    out_df = meta_ok[["path", "resolved_path", "emotion", "is_selected"]].copy()
    for i, au in enumerate(AU_NAMES):
        out_df[au] = all_preds[:, i]
    out_df["load_ok"] = all_ok

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".parquet":
        out_df.to_parquet(output, index=False)
    else:
        out_df.to_csv(output, index=False)

    # Clean up partial ckpt
    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"[info] saved: {output}")
    print(f"[info] rows: {len(out_df)}, load_ok: {int(out_df['load_ok'].sum())}")

    # Quick sanity: mean AU activity per emotion
    print("\n[sanity] per-emotion mean AU activity:")
    for emo in out_df["emotion"].unique():
        sub = out_df[out_df["emotion"] == emo]
        top5 = sub[AU_NAMES].mean().nlargest(5)
        print(f"  {emo}: {', '.join(f'{a}={v:.1f}' for a, v in top5.items())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="/home/ajy/AU-RegionFormer/libs/OpenGraphAU/checkpoints/OpenGprahAU-ResNet50_first_stage.pth")
    parser.add_argument("--meta-csv",
                        default="/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings/meta_mobilevitv2_150.csv")
    parser.add_argument("--image-root",
                        default="/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea")
    parser.add_argument("--output",
                        default="/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=50,
                        help="Save partial checkpoint every N batches")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit samples for dry-run (0=full)")
    args = parser.parse_args()
    main(args)
