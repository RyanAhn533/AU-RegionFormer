#!/usr/bin/env python3
"""
v3 preprocess — 옛날 v4 정확한 방식 그대로:
  Step 1: YOLOv8 face detection → face bbox crop → save face image
  Step 2: face image → mediapipe FaceMesh → v4 10-region landmark
  Step 3: 256×256 patch coords → CSV

Output:
  - face image: /mnt/hdd/ajy_25/face_cropped_v3/<dataset>/<class>/<basename>.face.jpg
  - csv: /home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs_v3/<dataset>_4c.csv

CSV schema (matches old v4):
  path, orig_path, label, work_w, work_h, patch_in, patch_out,
  forehead_69_{cx,cy,wx1,wy1,wx2,wy2}, forehead_299_*, forehead_9_*,
  eyes_159_*, eyes_386_*, nose_*, cheeks_186_*, cheeks_410_*,
  mouth_*, chin_*, ear_left, ear_right
"""
import os, argparse, csv, sys, json
from pathlib import Path

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["YOLO_VERBOSE"] = "False"

import numpy as np
import cv2
cv2.setNumThreads(1)
from PIL import Image
from tqdm import tqdm
import pandas as pd

# v4 region landmarks
REGIONS = {
    "forehead_69": 69, "forehead_299": 299, "forehead_9": 9,
    "eyes_159": 159, "eyes_386": 386,
    "nose": 195,
    "cheeks_186": 186, "cheeks_410": 410,
    "mouth": 13, "chin": 18,
}
ORDER = list(REGIONS.keys())

WORK_SHORT = 800
PATCH_IN = 256
HALF = PATCH_IN // 2

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# globals for worker
_yolo = None
_fm = None
_face_out_root = None


def worker_init(face_out_root, yolo_ckpt):
    """Initialize per-worker YOLO and mediapipe FaceMesh."""
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    cv2.setNumThreads(1)
    try: cv2.ocl.setUseOpenCL(False)
    except: pass
    global _yolo, _fm, _face_out_root
    from ultralytics import YOLO
    _yolo = YOLO(yolo_ckpt)
    import mediapipe as mp
    _fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    _face_out_root = face_out_root


def resize_short(img, s=WORK_SHORT):
    h, w = img.shape[:2]
    if min(h, w) == s: return img
    sc = s / float(min(h, w))
    return cv2.resize(img, (int(round(w*sc)), int(round(h*sc))),
                      interpolation=cv2.INTER_CUBIC if sc>1 else cv2.INTER_AREA)


def auto_orient(img):
    best = (-1, None, None)
    for deg in (0, 180):
        t = img if deg == 0 else cv2.rotate(img, cv2.ROTATE_180)
        r = _fm.process(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks: continue
        lm = r.multi_face_landmarks[0]
        h, w = t.shape[:2]
        score = float(np.hypot(lm.landmark[159].x*w - lm.landmark[386].x*w,
                               lm.landmark[159].y*h - lm.landmark[386].y*h))
        if score > best[0]: best = (score, t, lm)
    return best[1], best[2]


def ear(lms, idxs, w, h):
    pts = [(lms.landmark[i].x*w, lms.landmark[i].y*h) for i in idxs]
    def d(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])
    v1, v2, hz = d(pts[1], pts[5]), d(pts[2], pts[4]), d(pts[0], pts[3])
    return float((v1+v2)/(2*hz)) if hz>1e-6 else 0.0


def process(args):
    """Returns (row_dict, sanity_dict) or (None, sanity_dict)."""
    path, label, skip_yolo = args
    sanity = {"path": path, "label": label}

    img_pil = None
    try:
        img_pil = Image.open(path).convert("RGB")
    except Exception as e:
        return None, {**sanity, "fail": f"pil_open_failed:{e}"}

    if skip_yolo:
        # Yonsei already face-cropped; use as-is
        face_pil = img_pil
        face_box = (0, 0, img_pil.size[0], img_pil.size[1])
    else:
        # YOLOv8 face detection
        try:
            res = _yolo(img_pil, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy() if res[0].boxes is not None else np.array([])
        except Exception as e:
            return None, {**sanity, "fail": f"yolo_failed:{e}"}
        if len(boxes) == 0:
            return None, {**sanity, "fail": "no_face_yolo"}
        # Pick largest face
        box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_pil.size[0], x2), min(img_pil.size[1], y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None, {**sanity, "fail": f"face_too_small_{x2-x1}x{y2-y1}"}
        face_pil = img_pil.crop((x1, y1, x2, y2))
        face_box = (x1, y1, x2, y2)
    sanity["face_box_orig"] = face_box

    # resize_short(800) on face image
    arr = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
    arr = resize_short(arr, WORK_SHORT)
    arr, lms = auto_orient(arr)
    if lms is None:
        return None, {**sanity, "fail": "no_mesh_post_yolo"}
    h, w = arr.shape[:2]

    # 10-region patches
    regions_data = {}
    oob = 0
    for name, lidx in REGIONS.items():
        lm = lms.landmark[lidx]
        cx, cy = lm.x * w, lm.y * h
        wx1, wy1, wx2, wy2 = cx - HALF, cy - HALF, cx + HALF, cy + HALF
        if wx1 < 0 or wy1 < 0 or wx2 > w or wy2 > h: oob += 1
        wx1c, wy1c = max(0.0, wx1), max(0.0, wy1)
        wx2c, wy2c = min(float(w), wx2), min(float(h), wy2)
        regions_data[name] = (cx, cy, wx1c, wy1c, wx2c, wy2c)
    sanity["oob_count"] = oob

    # sanity violations (region order)
    fh_y = regions_data["forehead_9"][1]
    eyes_y = (regions_data["eyes_159"][1] + regions_data["eyes_386"][1]) / 2
    mouth_y = regions_data["mouth"][1]
    chin_y = regions_data["chin"][1]
    violations = []
    if fh_y >= eyes_y: violations.append("forehead_below_eyes")
    if chin_y <= mouth_y: violations.append("chin_above_mouth")
    if mouth_y <= eyes_y: violations.append("mouth_above_eyes")
    sanity["violations"] = ",".join(violations)

    # save face image
    save_path = path  # default
    if _face_out_root:
        save_dir = Path(_face_out_root) / label
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        save_path = str(save_dir / f"{stem}.face.jpg")
        cv2.imwrite(save_path, arr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # build row
    row = {"path": save_path, "orig_path": path, "label": label,
           "work_w": w, "work_h": h, "patch_in": PATCH_IN, "patch_out": PATCH_IN}
    for name in ORDER:
        cx, cy, x1, y1, x2, y2 = regions_data[name]
        row.update({f"{name}_cx": round(cx, 2), f"{name}_cy": round(cy, 2),
                    f"{name}_wx1": round(x1, 2), f"{name}_wy1": round(y1, 2),
                    f"{name}_wx2": round(x2, 2), f"{name}_wy2": round(y2, 2)})
    row["ear_left"]  = round(ear(lms, LEFT_EYE,  w, h), 4)
    row["ear_right"] = round(ear(lms, RIGHT_EYE, w, h), 4)
    return row, sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with path,label cols")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--face_out_root", type=Path, required=True,
                    help="Where to save face-cropped images")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_yolo", action="store_true",
                    help="Input images are already face-cropped (Yonsei). Skip YOLO step.")
    ap.add_argument("--sanity_out", type=Path, default=None)
    args = ap.parse_args()

    yolo_ckpt = "/home/ajy/.cache/huggingface/hub/models--arnabdhar--YOLOv8-Face-Detection/snapshots/52fa54977207fa4f021de949b515fb19dcab4488/model.pt"

    df = pd.read_csv(args.csv)
    samples = [(r['path'], r['label'], args.skip_yolo) for _, r in df.iterrows()]
    print(f"[v3] {args.csv.name}: {len(samples)} samples", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.face_out_root.mkdir(parents=True, exist_ok=True)

    cols = ["path", "orig_path", "label", "work_w", "work_h", "patch_in", "patch_out"]
    for n in ORDER:
        cols += [f"{n}_cx", f"{n}_cy", f"{n}_wx1", f"{n}_wy1", f"{n}_wx2", f"{n}_wy2"]
    cols += ["ear_left", "ear_right"]

    sanity_log = open(args.sanity_out, "w") if args.sanity_out else None
    n_ok = 0; fails = {}; oob_count = 0; v_count = {}
    face_box_areas = []

    from multiprocessing import Pool, set_start_method
    try: set_start_method("spawn", force=True)
    except: pass

    with Pool(args.workers, initializer=worker_init,
              initargs=(str(args.face_out_root), yolo_ckpt)) as pool, \
         open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row, san in tqdm(pool.imap(process, samples, chunksize=8),
                             total=len(samples), desc=args.csv.stem, file=sys.stderr):
            if sanity_log:
                sanity_log.write(json.dumps(san, ensure_ascii=False) + "\n")
            if row is None:
                key = san.get("fail", "unknown").split(":")[0]
                fails[key] = fails.get(key, 0) + 1
            else:
                wr.writerow(row); n_ok += 1
                if san.get("oob_count", 0) > 0: oob_count += 1
                v = san.get("violations", "")
                if v:
                    for vv in v.split(","):
                        v_count[vv] = v_count.get(vv, 0) + 1
    if sanity_log: sanity_log.close()

    print(f"\n=== {args.csv.name} v3 sanity report ===", file=sys.stderr)
    print(f"  total: {len(samples)}, saved: {n_ok}", file=sys.stderr)
    if fails:
        print(f"  failures:", file=sys.stderr)
        for k, v in sorted(fails.items()): print(f"    {k}: {v}", file=sys.stderr)
    print(f"  OOB (region clipped): {oob_count}/{n_ok}", file=sys.stderr)
    if v_count:
        print(f"  violations:", file=sys.stderr)
        for k, v in sorted(v_count.items()): print(f"    {k}: {v}", file=sys.stderr)
    print(f"  csv: {args.out_csv}", file=sys.stderr)
    print(f"  face imgs: {args.face_out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
