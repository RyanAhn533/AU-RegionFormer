#!/usr/bin/env python3
"""
CK+ master CSV builder.

Structure:
  cohn-kanade-images/S{subj}/{session}/S{subj}_{session}_NNNNNNNN.png
  Emotion/S{subj}/{session}/S{subj}_{session}_NNNNNNNN_emotion.txt    (contains code 0-7)

CK+ emotion codes:
  0: neutral, 1: anger, 2: contempt, 3: disgust,
  4: fear,    5: happy, 6: sadness, 7: surprise

For 4-class match: 0→neutral, 1→angry, 5→happy, 6→sad. Skip 2/3/4/7.

Strategy: take peak-frame (last frame of session) where emotion is fully expressed,
PLUS optionally first frame as 'neutral' control.
"""
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse, csv, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
import cv2
cv2.setNumThreads(1)
from tqdm import tqdm

# Reuse REGIONS / process from build_crossdataset_csv.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_crossdataset_csv import REGIONS, ORDER, LEFT_EYE, RIGHT_EYE, PAD_RATIO, \
    worker_init, resize_short, auto_orient, region_cb, ear

CODE_TO_EN = {0: "neutral", 1: "angry", 5: "happy", 6: "sad"}


def collect(ck_root: Path):
    """Walk CK+ images dir + matching Emotion dir, build (image, label, subject) list."""
    images_root = ck_root / "cohn-kanade-images"
    emotion_root = ck_root / "Emotion"
    if not images_root.exists() or not emotion_root.exists():
        print(f"missing: {images_root} or {emotion_root}")
        return []
    out = []
    for subj_dir in sorted(images_root.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("S"): continue
        subj = subj_dir.name
        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir(): continue
            sess = sess_dir.name
            # find emotion code file
            emo_dir = emotion_root / subj / sess
            emo_code = None
            if emo_dir.exists():
                txts = list(emo_dir.glob("*.txt"))
                if txts:
                    try:
                        emo_code = int(float(open(txts[0]).read().strip()))
                    except Exception:
                        pass
            if emo_code not in CODE_TO_EN: continue
            label_en = CODE_TO_EN[emo_code]
            # peak frame = last image in session
            imgs = sorted(sess_dir.glob("*.png"))
            if not imgs: continue
            peak = imgs[-1]
            out.append((str(peak), label_en, subj))
            # also use first frame as neutral if original session is non-neutral
            if emo_code != 0 and len(imgs) > 1:
                out.append((str(imgs[0]), "neutral", subj))
    return out


def process(args):
    path, label, subj = args
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
    row.update({"subject_hash": subj, "image_key": "",
                "is_selected": 0, "n_raters": 0,
                "mean_is_selected": 0.0, "quality_score": 0.85})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck_root", type=Path, required=True,
                    help="CK+ root (contains cohn-kanade-images/ and Emotion/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    samples = collect(args.ck_root)
    print(f"images found: {len(samples)}  (peak frames + neutral controls)")
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
        for r in tqdm(pool.imap(process, samples, chunksize=8), total=len(samples), desc="CK+"):
            if r is not None: wr.writerow(r); n_ok += 1
            else: n_fail += 1
    print(f"saved {n_ok} (failed {n_fail}) → {args.out}")


if __name__ == "__main__":
    main()
