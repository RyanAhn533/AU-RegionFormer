#!/usr/bin/env python3
"""
Extract face frames from AFEW/DFEW video datasets.

Strategy: take 3 frames per video (start/middle/end thirds). MediaPipe will
filter non-face frames downstream.

Usage:
  python extract_video_frames.py \
      --src "/mnt/ssd2/AFEW(이미지)/Train_AFEW" \
      --out /tmp/afew_train_frames \
      --classes "Angry,Happy,Neutral,Sad" \
      --n_per_video 3
"""
import argparse, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
cv2.setNumThreads(1)
from tqdm import tqdm


def extract_one(args):
    video_path, out_dir, n_per_video = args
    p = Path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nframes <= 0: cap.release(); return 0
    if n_per_video == 1:
        idxs = [nframes // 2]
    else:
        idxs = [int(nframes * (i + 0.5) / n_per_video) for i in range(n_per_video)]
    n_saved = 0
    for j, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        out_path = Path(out_dir) / f"{p.stem}_f{j}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_saved += 1
    cap.release()
    return n_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", type=str, required=True,
                    help="comma-sep folder names (these become subfolders in out)")
    ap.add_argument("--n_per_video", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cls in classes:
        cls_dir = args.src / cls
        out_cls = args.out / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
            for v in cls_dir.glob(ext):
                tasks.append((str(v), str(out_cls), args.n_per_video))
    print(f"videos found: {len(tasks)} across {len(classes)} classes")
    if not tasks: return

    n_total = 0
    with Pool(args.workers) as pool:
        for n in tqdm(pool.imap_unordered(extract_one, tasks, chunksize=4),
                      total=len(tasks), desc="extract"):
            n_total += n
    print(f"saved {n_total} frames to {args.out}")


if __name__ == "__main__":
    main()
