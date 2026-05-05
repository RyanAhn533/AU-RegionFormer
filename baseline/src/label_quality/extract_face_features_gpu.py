"""
얼굴 특징 추출 — GPU 전용 (batch processing)
InsightFace로 detection + landmark 둘 다 GPU에서 처리.
MediaPipe 대신 InsightFace 2D106 landmark 사용.
"""

import os
import numpy as np
import pandas as pd
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

from insightface.app import FaceAnalysis


def compute_features_106(landmarks, bbox):
    """InsightFace 2D106 landmarks → features.
    106점 landmark로도 EAR, MAR, 눈썹 높이 등 계산 가능.

    InsightFace 106 landmark indices:
    0-32: face contour
    33-37: left eyebrow (inner→outer)
    38-42: right eyebrow (inner→outer)
    43-47: nose bridge
    48-51: nose bottom
    52-57: left eye
    58-63: right eye
    64-67: upper lip outer
    68-71: lower lip outer
    72-75: upper lip inner
    76-79: lower lip inner
    80-82: left pupil area
    83-85: right pupil area
    86-89: more points...
    """
    lm = landmarks

    # Reference distance: inter-eye
    left_eye_center = np.mean(lm[52:58], axis=0)
    right_eye_center = np.mean(lm[58:64], axis=0)
    ref = np.linalg.norm(left_eye_center - right_eye_center)
    if ref < 1e-6:
        return None

    f = {}

    # EAR (Eye Aspect Ratio)
    # Left eye: 52-57, Right eye: 58-63
    def ear_106(eye_pts):
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        h = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (v1+v2)/(2*h) if h > 1e-6 else 0

    f['ear_left'] = ear_106(lm[52:58])
    f['ear_right'] = ear_106(lm[58:64])
    f['ear_avg'] = (f['ear_left'] + f['ear_right']) / 2

    # MAR (Mouth Aspect Ratio)
    lip_top = lm[64]  # upper lip top center
    lip_bottom = lm[68]  # lower lip bottom center
    lip_left = lm[76] if len(lm) > 76 else lm[64]
    lip_right = lm[72] if len(lm) > 72 else lm[66]

    # More robust: use outer lip points
    mouth_top = np.mean(lm[64:68], axis=0)
    mouth_bottom = np.mean(lm[68:72], axis=0)
    mouth_left = lm[64]
    mouth_right = lm[67] if len(lm) > 67 else lm[66]

    mouth_h = np.linalg.norm(mouth_top - mouth_bottom)
    mouth_w = np.linalg.norm(mouth_left - mouth_right)
    f['mar'] = mouth_h / max(mouth_w, 1e-6)
    f['mouth_width'] = mouth_w / ref

    # Eyebrow height
    left_brow_center = np.mean(lm[33:38], axis=0)
    right_brow_center = np.mean(lm[38:43], axis=0)
    f['brow_height_left'] = (left_eye_center[1] - left_brow_center[1]) / ref
    f['brow_height_right'] = (right_eye_center[1] - right_brow_center[1]) / ref
    f['brow_height_avg'] = (f['brow_height_left'] + f['brow_height_right']) / 2

    # Brow furrow (inner vs outer height diff)
    f['brow_furrow'] = ((lm[33][1] - lm[37][1]) + (lm[38][1] - lm[42][1])) / (2 * ref)

    # Nose bridge length
    f['nose_bridge'] = np.linalg.norm(lm[43] - lm[47]) / ref

    # Cheek raise
    left_cheek = lm[14] if len(lm) > 14 else lm[10]  # face contour
    right_cheek = lm[18] if len(lm) > 18 else lm[22]
    f['cheek_raise_left'] = (left_cheek[1] - left_eye_center[1]) / ref
    f['cheek_raise_right'] = (right_cheek[1] - right_eye_center[1]) / ref
    f['cheek_raise_avg'] = (f['cheek_raise_left'] + f['cheek_raise_right']) / 2

    # Lip corner angle
    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
    lip_corners_y = (mouth_left[1] + mouth_right[1]) / 2
    f['lip_corner_angle'] = (mouth_center_y - lip_corners_y) / ref

    # Face aspect ratio (from bbox)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    f['face_aspect_ratio'] = bh / max(bw, 1e-6)

    # Chin length (nose tip to chin)
    nose_tip = lm[48] if len(lm) > 48 else lm[47]
    chin = lm[16]  # face contour bottom
    f['chin_length'] = np.linalg.norm(nose_tip - chin) / ref

    # Forehead height (top contour to brow)
    forehead = lm[0]  # top of face contour
    brow_avg_y = (left_brow_center[1] + right_brow_center[1]) / 2
    f['forehead_height'] = (brow_avg_y - forehead[1]) / ref

    return f


def main(
    csv_path="/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv",
    output_path="/home/ajy/AU-RegionFormer/data/label_quality/face_features_gpu.csv",
    skip_existing=60000,
):
    df = pd.read_csv(csv_path)

    # 이미 처리된 60K 스킵
    if skip_existing > 0:
        df = df.iloc[skip_existing:]
        print(f"Skipping first {skip_existing}, processing {len(df):,}")

    total = len(df)
    print(f"Total: {total:,}")

    # InsightFace — GPU, detection + landmark 동시
    print("Loading InsightFace (GPU)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    results = []
    failed = 0
    t0 = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        path = row['path']
        try:
            img = cv2.imread(path)
            if img is None:
                failed += 1; continue

            faces = app.get(img)
            if not faces:
                failed += 1; continue

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

            if face.landmark_2d_106 is not None:
                features = compute_features_106(face.landmark_2d_106, face.bbox)
                if features:
                    features['path'] = path
                    features['emotion'] = row.get('emotion', '')
                    features['is_selected'] = int(row.get('is_selected', -1))
                    results.append(features)
                else:
                    failed += 1
            else:
                failed += 1

        except Exception:
            failed += 1

        if (idx+1) % 5000 == 0:
            elapsed = time.time()-t0
            speed = (idx+1)/elapsed
            eta = (total-idx-1)/max(speed, 0.1)
            print(f"[{idx+1:,}/{total:,}] {speed:.1f} it/s | "
                  f"success={len(results)/(idx+1)*100:.1f}% | "
                  f"ETA={eta/60:.0f}min")

        if (idx+1) % 10000 == 0 and results:
            pd.DataFrame(results).to_csv(output_path + ".tmp", index=False)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        elapsed = time.time()-t0
        print(f"\nDone: {len(results):,}/{total:,}")
        print(f"Speed: {total/elapsed:.1f} it/s")
        print(f"Time: {elapsed/60:.1f}min")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
