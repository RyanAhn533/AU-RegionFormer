#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AU 좌표 인덱서 (학습 전처리와 동일 로직)
---------------------------------------
• 1) 입력 이미지를 (img_size x img_size) 로 '강제 리사이즈'
• 2) 리사이즈된 프레임에서 FaceMesh 실행(refine_landmarks=옵션)
• 3) AU 중심 = 지정 랜드마크(튜플이면 평균)의 픽셀 좌표
• 4) patch_size 기준으로 중심에서 박스(x1,y1,x2,y2) 생성(경계 보정)
• 5) CSV: path,label,ux0,uy0,...,ux5,uy5[, bx0_1,by0_1,bx0_2,by0_2, ...]  ← --write-boxes 켜면 추가
• 6) NPZ: paths, labels, ux, uy[, boxes]  ← boxes는 (N,6,4) (x1,y1,x2,y2)

예:
python au_index_like_training.py \
  --train /data/train --val /data/val --outdir /mnt/hdd/out \
  --img-size 224 --patch-size 128 --workers 12 --chunksize 32 \
  --backend pil --refine --write-boxes
"""

import csv
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# ─────────── 설정: AU 랜드마크 ───────────
REGION_LANDMARKS = {
    "forehead": (69, 10, 151, 299),
    "eyes":     (159, 386),
    "nose":      5,
    "cheeks":   (205, 425),
    "mouth":    13,
    "chin":     200,
}
AU_ORDER = ["forehead", "eyes", "nose", "cheeks", "mouth", "chin"]
NUM_AU = 6

# ─────────── 기본 경로 ───────────
DEFAULT_TRAIN = "/home/ajy/AI_hub_250704/data2/data_processed_korea"
DEFAULT_VAL   = "/home/ajy/AI_hub_250704/data2/data_processed_korea_validation"
DEFAULT_OUT   = "/mnt/hdd/ajy_25/Aihub_data_AU_Croped"

# ─────────── 수집 ───────────
def collect_samples(root: Path):
    classes = sorted(p.name for p in root.iterdir() if (root / p.name).is_dir())
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    for cls in classes:
        for p in (root / cls).iterdir():
            if p.suffix.lower() in exts:
                samples.append((p, cls))
    return classes, samples

# ─────────── 워커 전역 ───────────
_face_mesh = None
_backend = "pil"
_refine = True
_img_size = 224
_patch_size = 128
_cv2 = None
_Image = None

def _init_worker(backend: str, refine: bool, img_size: int, patch_size: int):
    global _face_mesh, _backend, _refine, _img_size, _patch_size, _cv2, _Image
    _backend = backend
    _refine = refine
    _img_size = img_size
    _patch_size = patch_size

    import mediapipe as mp
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=_refine
    )

    if _backend == "cv2":
        import cv2
        _cv2 = cv2
    else:
        from PIL import Image
        _Image = Image

def _read_resize_rgb(path: Path):
    """
    리사이즈된 RGB ndarray 반환.
    너의 전처리와 동일하게 '정확히 (img_size, img_size)' 로 맞춘다.
    """
    if _backend == "cv2":
        arr = _cv2.imread(str(path), _cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("cv2.imread 실패")
        arr = _cv2.cvtColor(arr, _cv2.COLOR_BGR2RGB)
        arr = _cv2.resize(arr, (_img_size, _img_size), interpolation=_cv2.INTER_AREA)
        return arr
    else:
        with _Image.open(path).convert("RGB") as img:
            img = img.resize((_img_size, _img_size), resample=_Image.BILINEAR)
            return np.array(img)

def _center_from_indices(lm, indices: Tuple[int, ...], w: int, h: int):
    """튜플이면 평균, 단일이면 그대로 픽셀 좌표로 변환."""
    if isinstance(indices, tuple):
        xs = [lm[i].x for i in indices]
        ys = [lm[i].y for i in indices]
        x = float(np.mean(xs)) * w
        y = float(np.mean(ys)) * h
    else:
        x = lm[indices].x * w
        y = lm[indices].y * h
    return x, y

def _clamp_box(cx: float, cy: float, w: int, h: int, ps: int):
    """중심(cx,cy)와 patch_size로 (x1,y1,x2,y2) 박스를 경계 보정 포함 계산."""
    half = ps // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + ps
    y2 = y1 + ps
    # 경계 보정
    if x1 < 0:   x1, x2 = 0, ps
    if y1 < 0:   y1, y2 = 0, ps
    if x2 > w:   x2, x1 = w, w - ps
    if y2 > h:   y2, y1 = h, h - ps
    return x1, y1, x2, y2

def _random_center(w: int, h: int, ps: int):
    """실패 시 랜덤 패치 중심(전처리와 동작 일치)."""
    x = random.randint(0, w - ps) + ps / 2.0
    y = random.randint(0, h - ps) + ps / 2.0
    return x, y

def _process_one(args):
    """
    단일 샘플 처리:
    • 이미지를 (img_size,img_size)로 리사이즈
    • FaceMesh → AU 중심(cx,cy) 픽셀 좌표
    • patch_size로 박스 계산
    • 실패 시 랜덤 위치
    """
    img_path, cls = args
    try:
        arr = _read_resize_rgb(img_path)  # (H=W=img_size, 3)
    except Exception:
        # 완전 실패: ux/uy는 -1, 박스도 -1
        centers = [(-1.0, -1.0)] * NUM_AU
        boxes = [(-1, -1, -1, -1)] * NUM_AU
        return (str(img_path), cls, centers, boxes, True)

    H, W = arr.shape[:2]
    res = _face_mesh.process(arr)

    centers = []
    boxes = []
    failed = False

    if not res.multi_face_landmarks:
        failed = True
        for _ in range(NUM_AU):
            cx, cy = _random_center(W, H, _patch_size)
            centers.append((cx, cy))
            boxes.append(_clamp_box(cx, cy, W, H, _patch_size))
    else:
        lm = res.multi_face_landmarks[0].landmark
        for key in AU_ORDER:
            cx, cy = _center_from_indices(lm, REGION_LANDMARKS[key], W, H)
            # 전처리와 동일하게 중심에서 patch_size 박스
            x1, y1, x2, y2 = _clamp_box(cx, cy, W, H, _patch_size)
            centers.append((cx, cy))
            boxes.append((x1, y1, x2, y2))

    return (str(img_path), cls, centers, boxes, failed)

# ─────────── 메인 처리 ───────────
def run_split_like_training(
    src_root: Path,
    out_csv: Path,
    out_npz: Optional[Path] = None,
    workers: Optional[int] = None,
    chunksize: int = 32,
    backend: str = "pil",
    refine: bool = True,
    img_size: int = 224,
    patch_size: int = 128,
    write_boxes: bool = False,
):
    classes, samples = collect_samples(src_root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 워커 수
    if workers is None:
        try:
            import multiprocessing as mp
            workers = min(mp.cpu_count(), 12)
        except Exception:
            workers = 8

    from multiprocessing import Pool
    rows = []
    fail = 0

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(backend, refine, img_size, patch_size),
    ) as pool:
        for img_path, cls, centers, boxes, failed in tqdm(
            pool.imap(_process_one, samples, chunksize=chunksize),
            total=len(samples), desc=f"{src_root.name}", ncols=100
        ):
            if failed:
                fail += 1

            # CSV 행 구성: ux0,uy0,...,ux5,uy5  (픽셀 좌표)
            flat = []
            for (cx, cy) in centers:
                flat.extend([f"{cx:.4f}", f"{cy:.4f}"])

            # 옵션: 박스도 저장
            if write_boxes:
                for (x1, y1, x2, y2) in boxes:
                    flat.extend([str(x1), str(y1), str(x2), str(y2)])

            rows.append([img_path, cls] + flat)

    # 헤더
    header = ["path", "label"] + [f"ux{i}" for i in range(NUM_AU)] + [f"uy{i}" for i in range(NUM_AU)]
    if write_boxes:
        # 각 AU마다 bx_i_1, by_i_1, bx_i_2, by_i_2
        for i in range(NUM_AU):
            header += [f"bx{i}_1", f"by{i}_1", f"bx{i}_2", f"by{i}_2"]

    # CSV 저장
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # NPZ 저장
    if out_npz is not None:
        paths  = np.array([r[0] for r in rows])
        labels = np.array([r[1] for r in rows])

        # ux,uy 복원
        # 행은 [path,label, ux0,ux1,...,ux5, uy0,uy1,...,uy5, (opt: boxes...)]
        ux_start = 2
        uy_start = 2 + NUM_AU
        ux = np.array([[float(rows[k][ux_start + i]) for i in range(NUM_AU)] for k in range(len(rows))], dtype=np.float32)
        uy = np.array([[float(rows[k][uy_start + i]) for i in range(NUM_AU)] for k in range(len(rows))], dtype=np.float32)

        save_dict = dict(paths=paths, labels=labels, ux=ux, uy=uy, classes=np.array(classes))

        if write_boxes:
            # 박스는 ux/uy 뒤에 이어짐
            box_start = 2 + NUM_AU + NUM_AU
            boxes = np.zeros((len(rows), NUM_AU, 4), dtype=np.int32)
            for r_i in range(len(rows)):
                off = box_start
                for a in range(NUM_AU):
                    x1 = int(rows[r_i][off]);   y1 = int(rows[r_i][off+1])
                    x2 = int(rows[r_i][off+2]); y2 = int(rows[r_i][off+3])
                    boxes[r_i, a] = [x1, y1, x2, y2]
                    off += 4
            save_dict["boxes"] = boxes

        np.savez_compressed(out_npz, **save_dict)

    print(f"[완료] {src_root} → {out_csv.name} (총 {len(rows)}건, 실패 {fail}건)")

# ─────────── CLI ───────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path(DEFAULT_TRAIN))
    ap.add_argument("--val",   type=Path, default=Path(DEFAULT_VAL))
    ap.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUT))
    # 성능
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=32)
    ap.add_argument("--backend", choices=["pil", "cv2"], default="pil")
    ap.add_argument("--refine", action="store_true")
    # 전처리 파라미터(너 코드와 동일 디폴트)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=128)
    # 출력 옵션
    ap.add_argument("--write-boxes", action="store_true", help="CSV/NPZ에 박스(x1,y1,x2,y2)도 저장")
    ap.add_argument("--no-npz", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    out_csv_tr = args.outdir / "index_train.csv"
    out_csv_vl = args.outdir / "index_val.csv"
    out_npz_tr = None if args.no_npz else (args.outdir / "index_train.npz")
    out_npz_vl = None if args.no_npz else (args.outdir / "index_val.npz")

    run_split_like_training(
        args.train, out_csv_tr, out_npz_tr,
        workers=args.workers, chunksize=args.chunksize,
        backend=args.backend, refine=bool(args.refine),
        img_size=args.img_size, patch_size=args.patch_size,
        write_boxes=bool(args.write_boxes),
    )
    run_split_like_training(
        args.val, out_csv_vl, out_npz_vl,
        workers=args.workers, chunksize=args.chunksize,
        backend=args.backend, refine=bool(args.refine),
        img_size=args.img_size, patch_size=args.patch_size,
        write_boxes=bool(args.write_boxes),
    )
    print(f"[저장 경로] {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
