#!/usr/bin/env python3
"""
Build master-format CSV for Western/cross-cultural FER datasets.

Walks a class-folder root (each emotion dir = label), runs MediaPipe FaceMesh
8-region AU center+bbox detection, outputs master-style CSV (without Yonsei fields).

Usage:
  python build_crossdataset_csv.py \
      --src /mnt/ssd2/SFEW_2/Train \
      --out csvs/sfew_train_4c.csv \
      --label_map "Angry:angry, Happy:happy, Neutral:neutral, Sad:sad"

Output cols match master_train.csv but is_selected/n_raters/mean_is_selected/quality_score
are filled with defaults (0, 0, 0.0, 0.85) since no Yonsei eval available.
"""
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse, csv, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
import cv2
cv2.setNumThreads(1)
from tqdm import tqdm

REGIONS = {
    "forehead":    [10,  67,  109, 297, 338],
    "eyes_left":   [33,  133, 159, 145, 153, 160],
    "eyes_right":  [362, 263, 386, 374, 380, 385],
    "nose":        [1,   2,   98,  327],
    "cheek_left":  [50,  101, 36,  205],
    "cheek_right": [280, 330, 266, 425],
    "mouth":       [61,  291, 13,  14,  78,  308],
    "chin":        [152, 175, 400, 199, 428],
}
ORDER = list(REGIONS.keys())
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
PAD_RATIO = 0.06


_fm = None
def worker_init():
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    cv2.setNumThreads(1)
    try: cv2.ocl.setUseOpenCL(False)
    except: pass
    global _fm
    import mediapipe as mp
    _fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


def resize_short(img, s=800):
    h, w = img.shape[:2]
    if min(h, w) == s: return img
    sc = s / float(min(h, w))
    return cv2.resize(img, (int(round(w*sc)), int(round(h*sc))),
                      interpolation=cv2.INTER_CUBIC if sc>1 else cv2.INTER_AREA)


def auto_orient(img):
    best = (-1, None, None)
    for deg in (0, 180):
        t = img if deg==0 else cv2.rotate(img, cv2.ROTATE_180)
        r = _fm.process(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks: continue
        lm = r.multi_face_landmarks[0]
        score = float(np.hypot(lm.landmark[159].x*t.shape[1]-lm.landmark[386].x*t.shape[1],
                               lm.landmark[159].y*t.shape[0]-lm.landmark[386].y*t.shape[0]))
        if score > best[0]: best = (score, t, lm)
    return best[1], best[2]


def region_cb(lms, idxs, w, h, fs):
    xs = np.array([lms[i].x for i in idxs]) * w
    ys = np.array([lms[i].y for i in idxs]) * h
    cx, cy = float(xs.mean()), float(ys.mean())
    pad = max(fs * PAD_RATIO, 8)
    return cx, cy, max(0, min(xs.min(), cx-pad)), max(0, min(ys.min(), cy-pad)), \
                  min(w, max(xs.max(), cx+pad)), min(h, max(ys.max(), cy+pad))


def ear(lms, idxs, w, h):
    pts = [(lms[i].x*w, lms[i].y*h) for i in idxs]
    def d(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])
    v1, v2, hz = d(pts[1], pts[5]), d(pts[2], pts[4]), d(pts[0], pts[3])
    return float((v1+v2)/(2*hz)) if hz>1e-6 else 0.0


def process(args):
    path, label = args
    img = cv2.imread(path)
    if img is None: return None
    img = resize_short(img, 800)
    img, lms = auto_orient(img)
    if lms is None: return None
    h, w = img.shape[:2]; fs = float(min(w, h))
    L = lms.landmark
    row = {"path": path, "label": label, "work_w": w, "work_h": h}
    for name in ORDER:
        cx, cy, x1, y1, x2, y2 = region_cb(L, REGIONS[name], w, h, fs)
        row.update({f"{name}_cx": round(cx,2), f"{name}_cy": round(cy,2),
                    f"{name}_wx1": round(x1,2), f"{name}_wy1": round(y1,2),
                    f"{name}_wx2": round(x2,2), f"{name}_wy2": round(y2,2)})
    row["ear_left"]  = round(ear(L, LEFT_EYE,  w, h), 4)
    row["ear_right"] = round(ear(L, RIGHT_EYE, w, h), 4)
    row.update({"subject_hash": "", "image_key": "",
                "is_selected": 0, "n_raters": 0,
                "mean_is_selected": 0.0, "quality_score": 0.85})
    return row


def collect(root: Path, label_map: dict):
    out = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir(): continue
        if cls_dir.name not in label_map: continue
        en = label_map[cls_dir.name]
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            for f in cls_dir.rglob(ext):   # rglob = recursive (handles SFEW nested 1-level)
                out.append((str(f), en))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label_map", type=str, required=True,
                    help='comma-sep "FolderName:englishlabel"')
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    label_map = {}
    for kv in args.label_map.split(","):
        k, v = kv.strip().split(":")
        label_map[k.strip()] = v.strip()
    print(f"label map: {label_map}")

    samples = collect(args.src, label_map)
    print(f"images found: {len(samples)}")
    if not samples: return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path","label","work_w","work_h"]
    for n in ORDER:
        cols += [f"{n}_cx", f"{n}_cy", f"{n}_wx1", f"{n}_wy1", f"{n}_wx2", f"{n}_wy2"]
    cols += ["ear_left","ear_right","subject_hash","image_key",
             "is_selected","n_raters","mean_is_selected","quality_score"]

    set_start_method("spawn", force=True)
    n_ok = 0; n_fail = 0
    with Pool(args.workers, initializer=worker_init) as pool, \
         open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in tqdm(pool.imap(process, samples, chunksize=32), total=len(samples), desc=args.src.name):
            if r is not None:
                wr.writerow(r); n_ok += 1
            else:
                n_fail += 1
    print(f"saved {n_ok} (failed {n_fail}) → {args.out}")


if __name__ == "__main__":
    main()
