#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# ── 어떤 import보다 먼저 ──
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # GPU 비노출
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"     # 소프트웨어 렌더링 사용
# os.environ["EGL_PLATFORM"] = "surfaceless"  # 문제 지속 시 활성화

import csv, argparse, random
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count, set_start_method

import cv2
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import numpy as np
from tqdm import tqdm

# ================== 기본 경로 ==================
DEFAULT_TRAIN = "/home/ajy/AI_hub_250704/data2/data_processed_korea"
DEFAULT_VAL   = "/home/ajy/AI_hub_250704/data2/data_processed_korea_validation"
DEFAULT_OUT   = "/mnt/hdd/ajy_25/Aihub_data_AU_Croped4"

# ================== AU 정의 ==================
REGION_LANDMARKS = {
    "forehead": (69, 299, 9),
    "eyes": (159, 386),
    "nose": 195,
    "cheeks": (186, 410),
    "mouth": 13,
    "chin": 18}

AU_ORDER = ["forehead", "eyes", "nose", "cheeks", "mouth", "chin"]

def make_expanded_au() -> List[Tuple[str, int]]:
    out = []
    for name in AU_ORDER:
        idx = REGION_LANDMARKS[name]
        if isinstance(idx, tuple):
            for j in idx: out.append((f"{name}_{j}", j))
        else:
            out.append((name, idx))
    return out

EXPANDED_AU = make_expanded_au()

# ================== 유틸 ==================
def list_images(root: Path) -> List[Tuple[Path, str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for cls in sorted(p.name for p in root.iterdir() if (root/p).is_dir()):
        for f in sorted((root/cls).iterdir()):
            if f.suffix.lower() in exts:
                pairs.append((f, cls))
    return pairs

def clamp_window(x1:int,y1:int,x2:int,y2:int,w:int,h:int):
    return max(0,x1), max(0,y1), min(w,x2), min(h,y2)

def eye_distance(lms, w:int, h:int) -> float:
    lx, ly = int(lms.landmark[159].x*w), int(lms.landmark[159].y*h)
    rx, ry = int(lms.landmark[386].x*w), int(lms.landmark[386].y*h)
    return float(np.hypot(rx-lx, ry-ly))

def rotate_0_180(img, deg):
    if deg == 0:   return img
    else:          return cv2.rotate(img, cv2.ROTATE_180)

def resize_to_short_side(img, short_side:int):
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    scale = short_side / float(min(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def auto_orient_180(fm, img):
    """0도/180도만 시도해서 더 좋은(눈간거리 큰) 쪽 선택"""
    best_deg, best_lms, best_score, best_img = -1, None, -1.0, None
    for deg in (0, 180):
        test = rotate_0_180(img, deg)
        res = fm.process(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        lms = res.multi_face_landmarks[0]
        sc = eye_distance(lms, test.shape[1], test.shape[0])
        if sc > best_score:
            best_deg, best_lms, best_score, best_img = deg, lms, sc, test
    if best_img is None:
        return img, None, 0
    return best_img, best_lms, best_deg

def face_bbox_from_mesh(lms, w:int, h:int):
    xs = [int(pt.x*w) for pt in lms.landmark]
    ys = [int(pt.y*h) for pt in lms.landmark]
    return min(xs), min(ys), max(xs), max(ys)

def choose_interp(src_hw:Tuple[int,int], dst_hw:Tuple[int,int]):
    src_h, src_w = src_hw; dst_h, dst_w = dst_hw
    if dst_h < src_h or dst_w < src_w:
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC

# ================== Preview (샘플 저장) ==================
def preview_samples(train_root: Path, outdir: Path, num_samples:int,
                    work_short:int, patch_in:int, patch_out:int):
    outdir.mkdir(parents=True, exist_ok=True)
    samples = list_images(train_root)
    if not samples: return
    picks = random.sample(samples, min(num_samples, len(samples)))

    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        for (img_path, cls) in picks:
            img0 = cv2.imread(str(img_path))
            if img0 is None: continue

            # 1) 먼저 작업 해상도로 통일
            work = resize_to_short_side(img0, work_short)

            # 2) 작업 이미지에서 orientation 선택 + FaceMesh
            work, lms, rot_deg = auto_orient_180(fm, work)
            if lms is None:  continue
            h, w = work.shape[:2]

            half = patch_in // 2

            overlay = work.copy()
            save_dir = outdir / cls / img_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)

            for au_name, lm_idx in EXPANDED_AU:
                lm = lms.landmark[lm_idx]
                cx, cy = int(lm.x*w), int(lm.y*h)
                wx1, wy1, wx2, wy2 = cx-half, cy-half, cx+half, cy+half
                cx1, cy1, cx2, cy2 = clamp_window(wx1, wy1, wx2, wy2, w, h)

                # 패치 추출 + 제로패딩으로 정확히 patch_in
                patch = work[cy1:cy2, cx1:cx2]
                top = max(0, wy1 - cy1); left = max(0, wx1 - cx1)
                bottom = max(0, cy2 - wy2); right = max(0, cx2 - wx2)
                patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                if patch.shape[:2] != (patch_in, patch_in):
                    patch = cv2.resize(patch, (patch_in, patch_in), interpolation=choose_interp(patch.shape[:2], (patch_in, patch_in)))

                # 최종 반환 크기 통일
                if patch_in != patch_out:
                    patch = cv2.resize(patch, (patch_out, patch_out), interpolation=choose_interp((patch_in, patch_in), (patch_out, patch_out)))

                cv2.imwrite(str(save_dir / f"{img_path.stem}_rot{rot_deg}_{patch_out}px_{au_name}.jpg"), patch)

                # 오버레이
                cv2.rectangle(overlay, (cx1,cy1), (cx2,cy2), (0,255,0), 1)
                cv2.putText(overlay, f"{au_name}({patch_in}->{patch_out})", (cx1, max(0, cy1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

            cv2.putText(overlay, f"rot={rot_deg}  work={w}x{h}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imwrite(str(outdir / f"{img_path.stem}_overlay_rot{rot_deg}.jpg"), overlay)

# ================== 멀티프로세싱 ==================
# 워커 전역 상태
_fm = None
_work_short = None
_patch_in = None
_patch_out = None

def _worker_init(work_short_, patch_in_, patch_out_):
    # 환경변수 재보장
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
        _cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # 전역 파라미터
    global _fm, _work_short, _patch_in, _patch_out
    _work_short, _patch_in, _patch_out = work_short_, patch_in_, patch_out_

    # mediapipe는 여기서만 import + 인스턴스 생성
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    _fm = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def process_image(args):
    img_path, cls = args
    img0 = cv2.imread(str(img_path))
    if img0 is None: return None

    # 1) 작업 해상도로 통일
    work = resize_to_short_side(img0, _work_short)

    # 2) 작업 이미지에서 orientation + FaceMesh
    work, lms, rot_deg = auto_orient_180(_fm, work)
    if lms is None: return None
    h, w = work.shape[:2]

    half = _patch_in // 2

    row = [str(img_path), cls, rot_deg, w, h, _patch_in, _patch_out]
    for au_name, lm_idx in EXPANDED_AU:
        lm = lms.landmark[lm_idx]
        cx, cy = int(lm.x*w), int(lm.y*h)
        wx1, wy1, wx2, wy2 = cx - half, cy - half, cx + half, cy + half
        cx1, cy1, cx2, cy2 = clamp_window(wx1, wy1, wx2, wy2, w, h)
        row.extend([cx, cy, wx1, wy1, wx2, wy2, cx1, cy1, cx2, cy2])
    return row

def process_split(src_root: Path, out_csv: Path,
                  work_short:int, patch_in:int, patch_out:int, workers:int):
    samples = list_images(src_root)
    total = len(samples)
    if total == 0:
        print(f"{src_root} 이미지 없음"); return

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["path","label","rot_deg","work_w","work_h","patch_in","patch_out"]
    for au_name, _ in EXPANDED_AU:
        header += [
            f"{au_name}_cx", f"{au_name}_cy",
            f"{au_name}_wx1", f"{au_name}_wy1", f"{au_name}_wx2", f"{au_name}_wy2",
            f"{au_name}_cx1", f"{au_name}_cy1", f"{au_name}_cx2", f"{au_name}_cy2",
        ]

    processed = 0
    # chunksize는 I/O·CPU 혼합이라 16~64 권장
    with Pool(processes=workers, initializer=_worker_init,
              initargs=(work_short, patch_in, patch_out)) as pool, \
         open(out_csv, "w", newline="", encoding="utf-8") as f:

        wr = csv.writer(f); wr.writerow(header)
        for r in tqdm(pool.imap(process_image, samples, chunksize=32),
                      total=total, desc=src_root.name, dynamic_ncols=True):
            processed += 1
            if r is not None:
                wr.writerow(r)

    print(f"[완료] {src_root.name} → {out_csv} (총 {processed}개 처리)")

# ================== 패치 실 저장 유틸(옵션) ==================
def save_patches_from_csv(csv_path: Path, outdir: Path):
    """
    선택적으로, CSV 좌표를 사용해 실제 패치를 저장하고 싶을 때 사용.
    미리보기 외 대량 저장이 필요하면 이 함수를 호출.
    """
    import mediapipe as mp  # 미사용이지만 의존 방지용
    outdir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        # 컬럼 인덱스 맵
        col = {h:i for i,h in enumerate(header)}
        work_w_i = col["work_w"]; work_h_i = col["work_h"]
        patch_in_i = col["patch_in"]; patch_out_i = col["patch_out"]

        # 각 AU별 컬럼 시작 인덱스 기록
        au_cols = {}
        for au_name, _ in EXPANDED_AU:
            au_cols[au_name] = (
                col[f"{au_name}_wx1"], col[f"{au_name}_wy1"],
                col[f"{au_name}_wx2"], col[f"{au_name}_wy2"],
                col[f"{au_name}_cx1"], col[f"{au_name}_cy1"],
                col[f"{au_name}_cx2"], col[f"{au_name}_cy2"],
            )

        for r in tqdm(rd, desc=f"save:{csv_path.name}", dynamic_ncols=True):
            path = Path(r[col["path"]]); label = r[col["label"]]
            rot_deg = int(r[col["rot_deg"]])
            work_w = int(r[work_w_i]); work_h = int(r[work_h_i])
            patch_in = int(r[patch_in_i]); patch_out = int(r[patch_out_i])

            img0 = cv2.imread(str(path))
            if img0 is None: continue

            # CSV는 작업 이미지 좌표계이므로 동일한 리사이즈, 동일 회전 적용
            work = resize_to_short_side(img0, min(work_w, work_h))
            work = rotate_0_180(work, rot_deg)

            save_dir = outdir / label / path.stem
            save_dir.mkdir(parents=True, exist_ok=True)

            for au_name, _ in EXPANDED_AU:
                wx1_i, wy1_i, wx2_i, wy2_i, cx1_i, cy1_i, cx2_i, cy2_i = au_cols[au_name]
                wx1, wy1 = int(r[wx1_i]), int(r[wy1_i])
                wx2, wy2 = int(r[wx2_i]), int(r[wy2_i])
                cx1, cy1 = int(r[cx1_i]), int(r[cy1_i])
                cx2, cy2 = int(r[cx2_i]), int(r[cy2_i])

                patch = work[cy1:cy2, cx1:cx2]
                top = max(0, wy1 - cy1); left = max(0, wx1 - cx1)
                bottom = max(0, cy2 - wy2); right = max(0, cx2 - wx2)
                patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                if patch.shape[:2] != (patch_in, patch_in):
                    patch = cv2.resize(patch, (patch_in, patch_in), interpolation=choose_interp(patch.shape[:2], (patch_in, patch_in)))
                if patch_in != patch_out:
                    patch = cv2.resize(patch, (patch_out, patch_out), interpolation=choose_interp((patch_in, patch_in), (patch_out, patch_out)))

                cv2.imwrite(str(save_dir / f"{path.stem}_rot{rot_deg}_{patch_out}px_{au_name}.jpg"), patch)

# ================== CLI / MAIN ==================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",  type=Path, default=Path(DEFAULT_TRAIN))
    ap.add_argument("--val",    type=Path, default=Path(DEFAULT_VAL))
    ap.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--preview", type=int, default=5, help="시작 전 저장할 샘플 이미지 수(0이면 미리보기 생략)")
    ap.add_argument("--workers", type=int, default=cpu_count())

    # 통일된 입력 크기와 고정 패치 크기
    ap.add_argument("--work-short", type=int, default=800,
                    help="입력 이미지를 짧은 변 기준으로 이 값에 맞춰 리사이즈")
    ap.add_argument("--patch-in", type=int, default=256,
                    help="작업 해상도에서 사용하는 크롭 윈도 크기(고정, px)")
    ap.add_argument("--patch-out", type=int, default=256,
                    help="디스크에 저장할 최종 패치 크기(px). 모든 반환 크기를 동일하게 맞춤")

    # 대량 패치 저장을 CSV 기반으로 추가 생산할 때 사용
    ap.add_argument("--export-from-csv", action="store_true",
                    help="index_train.csv / index_val.csv를 기반으로 실제 패치 저장 실행")

    return ap.parse_args()

def main():
    # ★ fork 대신 spawn 강제
    set_start_method("spawn", force=True)

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1) 미리보기
    if args.preview > 0:
        prev_dir = args.outdir / "preview"
        preview_samples(args.train, prev_dir, args.preview,
                        args.work_short, args.patch_in, args.patch_out)
        print(f"[미리보기 저장 완료] {prev_dir}")

    # 2) 전체 CSV (멀티프로세싱): 좌표와 메타만 저장
    process_split(args.train, args.outdir / "index_train.csv",
                  args.work_short, args.patch_in, args.patch_out, args.workers)
    process_split(args.val,   args.outdir / "index_val.csv",
                  args.work_short, args.patch_in, args.patch_out, args.workers)

    # 3) 필요 시 CSV 기반으로 실제 패치 대량 생성
    if args.export_from_csv:
        save_patches_from_csv(args.outdir / "index_train.csv", args.outdir / "patches_train")
        save_patches_from_csv(args.outdir / "index_val.csv",   args.outdir / "patches_val")

    print(f"[저장 경로] {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
