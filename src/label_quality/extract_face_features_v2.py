"""
얼굴 특징 추출 v2: InsightFace(GPU) crop → MediaPipe FaceMesh(468점)
=====================================================================
1단계: InsightFace SCRFD로 얼굴 detection + crop (GPU, batch)
2단계: crop된 작은 얼굴에 MediaPipe FaceMesh 468 landmark
3단계: landmark에서 AU 관련 17개 특징 계산

기존 FACS AU 체계 호환 유지.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

import insightface
from insightface.app import FaceAnalysis
import mediapipe as mp


# ── Landmark indices (MediaPipe 468점) ──
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
LIP_TOP = 13; LIP_BOTTOM = 14; LIP_LEFT = 61; LIP_RIGHT = 291
NOSE_TIP = 1; NOSE_BRIDGE = [6, 197, 195, 5]
FACE_TOP = 10; FACE_BOTTOM = 152; FACE_LEFT = 234; FACE_RIGHT = 454
LEFT_CHEEK = 50; RIGHT_CHEEK = 280
LEFT_EYE_INNER = 133; RIGHT_EYE_INNER = 362


def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def ear(eye_idx, lm):
    v1 = dist(lm[eye_idx[1]], lm[eye_idx[5]])
    v2 = dist(lm[eye_idx[2]], lm[eye_idx[4]])
    h = dist(lm[eye_idx[0]], lm[eye_idx[3]])
    return (v1+v2)/(2*h) if h > 1e-6 else 0


def compute_features(lm):
    """468 landmarks → 17 features."""
    ref = dist(lm[LEFT_EYE_INNER], lm[RIGHT_EYE_INNER])
    if ref < 1e-6:
        return None

    f = {}
    f['ear_left'] = ear(LEFT_EYE, lm)
    f['ear_right'] = ear(RIGHT_EYE, lm)
    f['ear_avg'] = (f['ear_left'] + f['ear_right']) / 2

    mouth_h = dist(lm[LIP_TOP], lm[LIP_BOTTOM])
    mouth_w = dist(lm[LIP_LEFT], lm[LIP_RIGHT])
    f['mar'] = mouth_h / max(mouth_w, 1e-6)
    f['mouth_width'] = mouth_w / ref

    ley = np.mean([lm[i][1] for i in LEFT_EYE])
    rey = np.mean([lm[i][1] for i in RIGHT_EYE])
    lby = np.mean([lm[i][1] for i in LEFT_EYEBROW])
    rby = np.mean([lm[i][1] for i in RIGHT_EYEBROW])

    f['brow_height_left'] = (ley - lby) / ref
    f['brow_height_right'] = (rey - rby) / ref
    f['brow_height_avg'] = (f['brow_height_left'] + f['brow_height_right']) / 2
    f['brow_furrow'] = ((lm[LEFT_EYEBROW[0]][1] - lm[LEFT_EYEBROW[-1]][1]) +
                        (lm[RIGHT_EYEBROW[0]][1] - lm[RIGHT_EYEBROW[-1]][1])) / (2 * ref)

    f['nose_bridge'] = dist(lm[NOSE_BRIDGE[0]], lm[NOSE_BRIDGE[-1]]) / ref

    f['cheek_raise_left'] = (lm[LEFT_CHEEK][1] - ley) / ref
    f['cheek_raise_right'] = (lm[RIGHT_CHEEK][1] - rey) / ref
    f['cheek_raise_avg'] = (f['cheek_raise_left'] + f['cheek_raise_right']) / 2

    mc_y = (lm[LIP_TOP][1] + lm[LIP_BOTTOM][1]) / 2
    f['lip_corner_angle'] = (mc_y - (lm[LIP_LEFT][1] + lm[LIP_RIGHT][1]) / 2) / ref

    face_h = dist(lm[FACE_TOP], lm[FACE_BOTTOM])
    face_w = dist(lm[FACE_LEFT], lm[FACE_RIGHT])
    f['face_aspect_ratio'] = face_h / max(face_w, 1e-6)
    f['chin_length'] = dist(lm[NOSE_TIP], lm[FACE_BOTTOM]) / ref
    f['forehead_height'] = ((lby + rby) / 2 - lm[FACE_TOP][1]) / ref

    return f


def main(
    csv_path="/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv",
    output_path="/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv",
    batch_save=10000,
):
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Total images: {total:,}")

    # ── 1단계: InsightFace face detector (GPU) ──
    print("Loading InsightFace SCRFD (GPU)...")
    app = FaceAnalysis(name='buffalo_sc', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # ── 2단계: MediaPipe FaceMesh ──
    print("Loading MediaPipe FaceMesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.3,
    )

    results = []
    failed = 0
    t0 = time.time()

    for idx, row in df.iterrows():
        path = row['path']
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                failed += 1; continue

            # 1단계: InsightFace로 얼굴 detection
            faces = app.get(img_bgr)
            if not faces:
                failed += 1; continue

            # 가장 큰 얼굴 선택
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = [int(v) for v in face.bbox]

            # 여유 margin 추가 (20%)
            h, w = img_bgr.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            # crop
            face_crop = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # 2단계: MediaPipe FaceMesh (작은 이미지)
            result = face_mesh.process(face_rgb)
            if not result.multi_face_landmarks:
                failed += 1; continue

            lm_raw = result.multi_face_landmarks[0]
            ch, cw = face_rgb.shape[:2]
            lm = [(l.x * cw, l.y * ch) for l in lm_raw.landmark]

            # 3단계: 특징 계산
            features = compute_features(lm)
            if features is None:
                failed += 1; continue

            features['path'] = path
            features['emotion'] = row.get('emotion', '')
            features['is_selected'] = int(row.get('is_selected', -1))
            results.append(features)

        except Exception:
            failed += 1

        # 진행률
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (total - idx - 1) / max(speed, 0.1)
            sr = len(results) / (idx + 1) * 100
            print(f"[{idx+1:,}/{total:,}] {speed:.1f} it/s | "
                  f"success={sr:.1f}% | failed={failed:,} | "
                  f"ETA={eta/60:.0f}min")

        # 중간 저장
        if (idx + 1) % batch_save == 0 and results:
            pd.DataFrame(results).to_csv(
                output_path + f".part_{(idx+1)//batch_save}", index=False)

    face_mesh.close()

    # 최종 저장
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        elapsed = time.time() - t0
        print(f"\nDone: {len(results):,}/{total:,} ({len(results)/total*100:.1f}%)")
        print(f"Failed: {failed:,}")
        print(f"Time: {elapsed/60:.1f}min ({elapsed/max(len(results),1)*1000:.1f}ms/img)")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
