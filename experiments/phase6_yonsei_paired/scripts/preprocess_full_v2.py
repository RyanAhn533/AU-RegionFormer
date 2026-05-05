#!/usr/bin/env python3
"""
Phase 6 — Full preprocess pipeline v2.

For ALL datasets (Yonsei + Western), produces:
  1. v4-schema 10-region landmark CSV (forehead_69/299/9, eyes_159/386, nose, cheeks_186/410, mouth, chin)
  2. face bbox crop (work_short=800 in face-centered coord) — fixes SFEW/AFEW small-face problem
  3. 256×256 patch coords around each landmark
  4. Per-image sanity check (face detect ok, OOB count, region order correct)
  5. Aggregate health stats per dataset

Usage:
  python preprocess_full_v2.py \
      --src /mnt/ssd2/SFEW_2/Train \
      --out csvs_v2/sfew_train_4c.csv \
      --label_map "Angry:angry,Happy:happy,Neutral:neutral,Sad:sad" \
      [--workers 12]

For Yonsei:
  --src /home/ajy/AI_hub_250704/data2/data_processed_korea \
  --label_map "기쁨:happy,분노:angry,슬픔:sad,중립:neutral" \
  --yonsei_csv /home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv \
  --sad_map /home/ajy/AU-RegionFormer/data/label_quality/sad_to_orig_mapping.csv

Output sanity-check report (stderr): n_total, n_face_detected, n_oob, n_sanity_violation,
                                     dataset health summary.
"""
import os, argparse, csv, sys, json, re
from pathlib import Path
from multiprocessing import Pool

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import cv2
cv2.setNumThreads(1)
from tqdm import tqdm

# v4 region landmarks (from FER_03_aihub_au_vit/code/_archive/250809_AU_crop_csv copy 3.py)
REGIONS = {
    "forehead_69":  69,
    "forehead_299": 299,
    "forehead_9":   9,
    "eyes_159":     159,
    "eyes_386":     386,
    "nose":         195,
    "cheeks_186":   186,
    "cheeks_410":   410,
    "mouth":        13,
    "chin":         18,
}
ORDER = list(REGIONS.keys())

WORK_SHORT = 800
PATCH_IN   = 256
HALF       = PATCH_IN // 2
FACE_PAD_RATIO = 0.40  # padding around face bbox (40% of face size — wider to keep all 256-patches in-bounds)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


_fm = None
def worker_init():
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    cv2.setNumThreads(1)
    try: cv2.ocl.setUseOpenCL(False)
    except: pass
    global _fm
    import mediapipe as mp
    _fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


def resize_short(img, s=WORK_SHORT):
    h, w = img.shape[:2]
    if min(h, w) == s: return img
    sc = s / float(min(h, w))
    return cv2.resize(img, (int(round(w*sc)), int(round(h*sc))),
                      interpolation=cv2.INTER_CUBIC if sc>1 else cv2.INTER_AREA)


def auto_orient(img):
    """Try 0/180 deg, pick orientation with larger eye distance."""
    best = (-1, None, None)
    for deg in (0, 180):
        t = img if deg==0 else cv2.rotate(img, cv2.ROTATE_180)
        r = _fm.process(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks: continue
        lm = r.multi_face_landmarks[0]
        h, w = t.shape[:2]
        score = float(np.hypot(lm.landmark[159].x*w - lm.landmark[386].x*w,
                               lm.landmark[159].y*h - lm.landmark[386].y*h))
        if score > best[0]: best = (score, t, lm)
    return best[1], best[2]


def face_bbox(lms, w, h):
    """Tight face bbox from all 468 landmarks."""
    xs = np.array([pt.x for pt in lms.landmark]) * w
    ys = np.array([pt.y for pt in lms.landmark]) * h
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def ear(lms, idxs, w, h):
    pts = [(lms.landmark[i].x*w, lms.landmark[i].y*h) for i in idxs]
    def d(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])
    v1, v2, hz = d(pts[1], pts[5]), d(pts[2], pts[4]), d(pts[0], pts[3])
    return float((v1+v2)/(2*hz)) if hz>1e-6 else 0.0


def process(args):
    """Returns (row_dict, sanity_dict) or (None, sanity_dict_with_fail_reason).
    Saves face-cropped+resized image to FACE_OUT_ROOT/<class>/<basename> if FACE_OUT_ROOT set globally.
    """
    path, label = args
    sanity = {"path": path, "label": label}

    img = cv2.imread(path)
    if img is None:
        return None, {**sanity, "fail": "cv2_imread_failed"}

    # Step 1: face detect on original (or resized for performance)
    detect_img = resize_short(img, WORK_SHORT) if min(img.shape[:2]) > WORK_SHORT * 1.5 else img
    detect_img, lms_pre = auto_orient(detect_img)
    if lms_pre is None:
        return None, {**sanity, "fail": "no_face_detected"}

    h_d, w_d = detect_img.shape[:2]
    fx1, fy1, fx2, fy2 = face_bbox(lms_pre, w_d, h_d)

    # Step 2: face bbox crop with padding
    face_w, face_h = fx2 - fx1, fy2 - fy1
    pad = max(face_w, face_h) * FACE_PAD_RATIO
    cx1 = max(0, int(fx1 - pad)); cy1 = max(0, int(fy1 - pad))
    cx2 = min(w_d, int(fx2 + pad)); cy2 = min(h_d, int(fy2 + pad))
    if cx2 - cx1 < 32 or cy2 - cy1 < 32:
        return None, {**sanity, "fail": f"face_crop_too_small_{cx2-cx1}x{cy2-cy1}"}

    face_crop = detect_img[cy1:cy2, cx1:cx2]
    sanity["face_pct_of_frame"] = round(((cx2-cx1)*(cy2-cy1)) / (w_d*h_d), 3)

    # Step 3: resize_short(800) on face crop, re-detect landmarks for accurate region positions
    work = resize_short(face_crop, WORK_SHORT)
    work, lms = auto_orient(work)
    if lms is None:
        return None, {**sanity, "fail": "redetect_failed_post_crop"}
    h, w = work.shape[:2]

    # Step 4: v4 10-region patches
    regions_data = {}
    oob_count = 0
    for name, lidx in REGIONS.items():
        lm = lms.landmark[lidx]
        cx, cy = lm.x * w, lm.y * h
        wx1, wy1 = cx - HALF, cy - HALF
        wx2, wy2 = cx + HALF, cy + HALF
        # Track OOB before clipping
        if wx1 < 0 or wy1 < 0 or wx2 > w or wy2 > h:
            oob_count += 1
        # Clip
        wx1c = max(0, wx1); wy1c = max(0, wy1)
        wx2c = min(w, wx2); wy2c = min(h, wy2)
        regions_data[name] = (cx, cy, wx1c, wy1c, wx2c, wy2c)

    sanity["oob_count"] = oob_count

    # Step 5: sanity check region order
    fh_y = regions_data["forehead_9"][1]
    eyes_y = (regions_data["eyes_159"][1] + regions_data["eyes_386"][1]) / 2
    mouth_y = regions_data["mouth"][1]
    chin_y = regions_data["chin"][1]
    sanity_violations = []
    if fh_y >= eyes_y: sanity_violations.append("forehead_below_eyes")
    if chin_y <= mouth_y: sanity_violations.append("chin_above_mouth")
    if mouth_y <= eyes_y: sanity_violations.append("mouth_above_eyes")
    sanity["sanity_violations"] = ",".join(sanity_violations) if sanity_violations else ""

    # Eye distance for normalized face size
    ed_norm = float(np.hypot(regions_data["eyes_159"][0] - regions_data["eyes_386"][0],
                             regions_data["eyes_159"][1] - regions_data["eyes_386"][1])) / w
    sanity["eye_distance_norm"] = round(ed_norm, 3)

    # Save face-cropped+resized image to disk (for inference consistency)
    save_path = path  # default: original
    face_out_root = os.environ.get("FACE_OUT_ROOT", "")
    if face_out_root:
        # Mirror dir structure under face_out_root, suffix .face.jpg
        rel = Path(path).name
        save_dir = Path(face_out_root) / label
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / (Path(rel).stem + ".face.jpg"))
        cv2.imwrite(save_path, work, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Build CSV row (path = face-cropped image path if FACE_OUT_ROOT set)
    row = {"path": save_path, "orig_path": path, "label": label,
           "work_w": w, "work_h": h, "patch_in": PATCH_IN, "patch_out": PATCH_IN}
    for name in ORDER:
        cx, cy, x1, y1, x2, y2 = regions_data[name]
        row.update({f"{name}_cx": round(cx,2), f"{name}_cy": round(cy,2),
                    f"{name}_wx1": round(x1,2), f"{name}_wy1": round(y1,2),
                    f"{name}_wx2": round(x2,2), f"{name}_wy2": round(y2,2)})
    row["ear_left"]  = round(ear(lms, LEFT_EYE,  w, h), 4)
    row["ear_right"] = round(ear(lms, RIGHT_EYE, w, h), 4)
    return row, sanity


def collect_class_dirs(root: Path, label_map: dict):
    out = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir(): continue
        if cls_dir.name not in label_map: continue
        en = label_map[cls_dir.name]
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            for f in cls_dir.rglob(ext):
                out.append((str(f), en))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None,
                    help="Class-dir root (Folder=label). Mutually exclusive with --csv.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Existing CSV with path/label columns. Re-extracts regions.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label_map", type=str, default=None,
                    help="Required if --src; ignored if --csv (label already in CSV)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sanity_out", type=Path, default=None,
                    help="Optional path to write per-image sanity report (jsonl).")
    args = ap.parse_args()

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        samples = list(zip(df['path'].astype(str).tolist(), df['label'].astype(str).tolist()))
        print(f"[preprocess_v2] from csv {args.csv}: {len(samples)} samples", file=sys.stderr)
    else:
        if not args.src or not args.label_map:
            ap.error("--src and --label_map required if --csv not given")
        label_map = {kv.strip().split(":")[0].strip(): kv.strip().split(":")[1].strip()
                     for kv in args.label_map.split(",")}
        print(f"[preprocess_v2] label map: {label_map}", file=sys.stderr)
        samples = collect_class_dirs(args.src, label_map)
    print(f"[preprocess_v2] images found: {len(samples)}", file=sys.stderr)
    if not samples: return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path","orig_path","label","work_w","work_h","patch_in","patch_out"]
    for n in ORDER:
        cols += [f"{n}_cx", f"{n}_cy", f"{n}_wx1", f"{n}_wy1", f"{n}_wx2", f"{n}_wy2"]
    cols += ["ear_left", "ear_right"]

    sanity_log = open(args.sanity_out, "w") if args.sanity_out else None

    n_ok = 0
    n_fail = {"cv2_imread_failed":0, "no_face_detected":0,
              "redetect_failed_post_crop":0, "face_crop_too_small":0}
    sanity_stats = {"oob": 0, "violations": {}, "face_pct": []}

    from multiprocessing import set_start_method
    try: set_start_method("spawn", force=True)
    except: pass

    with Pool(args.workers, initializer=worker_init) as pool, \
         open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        desc = (args.src.name if args.src else args.csv.name) if (args.src or args.csv) else "preprocess"
        for row, sanity in tqdm(pool.imap(process, samples, chunksize=16), total=len(samples), desc=desc, file=sys.stderr):
            if sanity_log:
                sanity_log.write(json.dumps(sanity, ensure_ascii=False) + "\n")
            if row is None:
                fail = sanity.get("fail", "unknown")
                # bucket fail reasons
                if "too_small" in fail:
                    n_fail["face_crop_too_small"] = n_fail.get("face_crop_too_small", 0) + 1
                else:
                    n_fail[fail] = n_fail.get(fail, 0) + 1
            else:
                wr.writerow(row); n_ok += 1
                if sanity.get("oob_count", 0) > 0: sanity_stats["oob"] += 1
                v = sanity.get("sanity_violations", "")
                if v:
                    for vv in v.split(","):
                        sanity_stats["violations"][vv] = sanity_stats["violations"].get(vv, 0) + 1
                sanity_stats["face_pct"].append(sanity.get("face_pct_of_frame", 0))

    if sanity_log: sanity_log.close()

    # Report
    print(f"\n=== {desc} sanity report ===", file=sys.stderr)
    print(f"  total samples: {len(samples)}", file=sys.stderr)
    print(f"  saved (ok):    {n_ok}", file=sys.stderr)
    print(f"  failures:", file=sys.stderr)
    for k, v in n_fail.items():
        if v > 0: print(f"    {k}: {v}", file=sys.stderr)
    print(f"  OOB (any region clipped): {sanity_stats['oob']} / {n_ok}", file=sys.stderr)
    print(f"  sanity violations:", file=sys.stderr)
    for k, v in sanity_stats["violations"].items():
        print(f"    {k}: {v}", file=sys.stderr)
    if sanity_stats["face_pct"]:
        fps = np.array(sanity_stats["face_pct"])
        print(f"  face % of frame: min={fps.min():.3f} median={np.median(fps):.3f} max={fps.max():.3f}", file=sys.stderr)
    print(f"  saved: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
