"""
AU 영역별 embedding 추출
=========================
연세대 237K 이미지에서:
1. InsightFace로 얼굴 detection + crop
2. MediaPipe 468 landmark로 8개 AU region 중심 좌표
3. AU region crop (64x64)
4. pretrained backbone으로 각 region feature 추출
5. "맞다" vs "아니다" 비교용 embedding 저장

backbone: MobileViTv2-150 / ConvNeXt-Base / Swin-Base
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import timm
import time
import warnings
warnings.filterwarnings('ignore')

import mediapipe as mp
from insightface.app import FaceAnalysis

# AU region → MediaPipe landmark indices
AU_LANDMARKS = {
    'forehead':    [10, 67, 109, 297, 338],
    'eyes_left':   [33, 133, 159, 145, 153, 160],
    'eyes_right':  [362, 263, 386, 374, 380, 385],
    'nose':        [1, 2, 98, 327],
    'cheek_left':  [50, 101, 36, 205],
    'cheek_right': [280, 330, 266, 425],
    'mouth':       [61, 291, 13, 14, 78, 308],
    'chin':        [152, 175, 400, 199, 428],
}

REGION_NAMES = list(AU_LANDMARKS.keys())
CROP_SIZE = 64  # AU region crop size


def get_au_crops(face_rgb, landmarks_468, crop_size=64):
    """468 landmark에서 8개 AU region crop 추출."""
    h, w = face_rgb.shape[:2]
    crops = {}

    for region, indices in AU_LANDMARKS.items():
        # region 중심 좌표
        pts = np.array([[landmarks_468[i].x * w, landmarks_468[i].y * h] for i in indices])
        cx, cy = pts.mean(axis=0).astype(int)

        # crop 범위 (얼굴 크기 대비 비율로)
        half = int(min(w, h) * 0.15)  # 얼굴의 15%
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        crop = face_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

        crop = cv2.resize(crop, (crop_size, crop_size))
        crops[region] = crop

    return crops


def extract_embeddings(
    csv_path="/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv",
    output_dir="/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings",
    backbone_name="mobilevitv2_150",
    batch_size=64,
    max_gpu_mem_gb=38,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")

    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Total: {total:,}, Backbone: {backbone_name}")

    # Face detector
    print("Loading InsightFace...")
    face_app = FaceAnalysis(name='buffalo_sc', providers=['CUDAExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    # MediaPipe
    print("Loading MediaPipe FaceMesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.3,
    )

    # Backbone
    print(f"Loading {backbone_name}...")
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0).to(device).eval()

    # Get feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE).to(device)
        feat_dim = backbone(dummy).shape[-1]
    print(f"Feature dim: {feat_dim}")

    # Normalization
    data_cfg = timm.data.resolve_model_data_config(backbone)
    mean = torch.tensor(data_cfg['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(data_cfg['std']).view(1, 3, 1, 1).to(device)

    # Storage: per-region embeddings
    all_embeddings = {r: [] for r in REGION_NAMES}
    all_meta = []  # (path, emotion, is_selected)
    failed = 0
    t0 = time.time()

    # Batch accumulator
    batch_crops = {r: [] for r in REGION_NAMES}
    batch_meta = []

    for idx, row in df.iterrows():
        path = row['path']
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                failed += 1; continue

            # Face detection
            faces = face_app.get(img_bgr)
            if not faces:
                failed += 1; continue

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            margin = int((x2-x1) * 0.2)
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1-margin), max(0, y1-margin)
            x2, y2 = min(w, x2+margin), min(h, y2+margin)

            face_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            # MediaPipe landmarks
            result = face_mesh.process(face_rgb)
            if not result.multi_face_landmarks:
                failed += 1; continue

            lm = result.multi_face_landmarks[0].landmark

            # AU region crops
            crops = get_au_crops(face_rgb, lm, CROP_SIZE)

            for r in REGION_NAMES:
                batch_crops[r].append(crops[r])
            batch_meta.append((path, row.get('emotion',''), int(row.get('is_selected', -1))))

        except Exception:
            failed += 1
            continue

        # Process batch
        if len(batch_meta) >= batch_size:
            with torch.no_grad():
                for r in REGION_NAMES:
                    # [B, H, W, C] → [B, C, H, W] → normalize → backbone
                    batch_tensor = torch.from_numpy(
                        np.stack(batch_crops[r])
                    ).permute(0, 3, 1, 2).float().to(device) / 255.0
                    batch_tensor = (batch_tensor - mean) / std
                    features = backbone(batch_tensor).cpu().numpy()
                    all_embeddings[r].append(features)
                    batch_crops[r] = []

            all_meta.extend(batch_meta)
            batch_meta = []

        # Progress
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (total - idx - 1) / max(speed, 0.1)
            n_done = len(all_meta)
            print(f"[{idx+1:,}/{total:,}] {speed:.1f} it/s | "
                  f"done={n_done:,} | failed={failed} | ETA={eta/60:.0f}min")

    # Process remaining batch
    if batch_meta:
        with torch.no_grad():
            for r in REGION_NAMES:
                if batch_crops[r]:
                    batch_tensor = torch.from_numpy(
                        np.stack(batch_crops[r])
                    ).permute(0, 3, 1, 2).float().to(device) / 255.0
                    batch_tensor = (batch_tensor - mean) / std
                    features = backbone(batch_tensor).cpu().numpy()
                    all_embeddings[r].append(features)
        all_meta.extend(batch_meta)

    face_mesh.close()

    # Save
    print(f"\nSaving embeddings...")
    meta_df = pd.DataFrame(all_meta, columns=['path', 'emotion', 'is_selected'])
    meta_df.to_csv(f"{output_dir}/meta_{backbone_name}.csv", index=False)

    for r in REGION_NAMES:
        emb = np.concatenate(all_embeddings[r], axis=0)
        np.save(f"{output_dir}/{r}_{backbone_name}.npy", emb)
        print(f"  {r}: {emb.shape}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_meta):,}/{total:,} | "
          f"Failed: {failed} | Time: {elapsed/60:.1f}min | "
          f"Speed: {len(all_meta)/elapsed:.1f} it/s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilevitv2_150",
                    choices=["mobilevitv2_150", "convnext_base.fb_in22k_ft_in1k", "swin_base_patch4_window7_224.ms_in22k_ft_in1k"])
    args = ap.parse_args()
    extract_embeddings(backbone_name=args.backbone)
