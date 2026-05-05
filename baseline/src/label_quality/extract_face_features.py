"""
연세대 검증 이미지에서 얼굴 특징 추출
======================================
MediaPipe FaceMesh 468 landmark → AU 관련 특징값 계산.

각 이미지에서 추출하는 특징:
  1. 눈 개방도 (Eye Aspect Ratio, EAR) — 좌/우
  2. 입 개방도 (Mouth Aspect Ratio, MAR)
  3. 입 너비 비율 (Mouth Width Ratio)
  4. 눈썹 높이 (Eyebrow Height) — 좌/우
  5. 눈썹 기울기 (Eyebrow Angle) — 찡그림 감지
  6. 코 주름 영역 높이
  7. 볼 올라감 정도 (Cheek Raise)
  8. 턱 길이 비율
  9. 얼굴 종횡비 (Face Aspect Ratio)
  10. 입꼬리 각도 (Lip Corner Angle) — 미소/처짐

출력: CSV (path, emotion, is_selected, feature1, feature2, ...)
"""

import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# MediaPipe FaceMesh landmark indices
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Eye landmarks
LEFT_EYE_TOP = [159, 145]      # upper/lower eyelid
LEFT_EYE_BOTTOM = [145, 153]
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # full eye contour
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Eyebrow
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]

# Mouth
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
LIP_TOP = 13
LIP_BOTTOM = 14
LIP_LEFT = 61
LIP_RIGHT = 291

# Nose
NOSE_TIP = 1
NOSE_BRIDGE = [6, 197, 195, 5]

# Face outline
FACE_TOP = 10      # forehead center
FACE_BOTTOM = 152  # chin
FACE_LEFT = 234
FACE_RIGHT = 454

# Cheek
LEFT_CHEEK = 50
RIGHT_CHEEK = 280

# Inner eye corners (for reference distances)
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362


def compute_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def compute_ear(eye_landmarks, landmarks):
    """Eye Aspect Ratio — 눈 개방도. 높을수록 눈 크게 뜸."""
    # Vertical distances
    v1 = compute_distance(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
    v2 = compute_distance(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])
    # Horizontal distance
    h = compute_distance(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])
    if h < 1e-6: return 0
    return (v1 + v2) / (2.0 * h)


def extract_features_from_landmarks(landmarks):
    """468 landmarks → feature dict."""
    lm = landmarks  # shorthand

    features = {}

    # Reference distance: inter-eye distance (정규화 기준)
    ref_dist = compute_distance(lm[LEFT_EYE_INNER], lm[RIGHT_EYE_INNER])
    if ref_dist < 1e-6:
        return None

    # 1. Eye Aspect Ratio (좌/우)
    features['ear_left'] = compute_ear(LEFT_EYE, lm)
    features['ear_right'] = compute_ear(RIGHT_EYE, lm)
    features['ear_avg'] = (features['ear_left'] + features['ear_right']) / 2

    # 2. Mouth Aspect Ratio
    mouth_h = compute_distance(lm[LIP_TOP], lm[LIP_BOTTOM])
    mouth_w = compute_distance(lm[LIP_LEFT], lm[LIP_RIGHT])
    features['mar'] = mouth_h / max(mouth_w, 1e-6)
    features['mouth_width'] = mouth_w / ref_dist

    # 3. Eyebrow height (눈 대비)
    left_eye_center_y = np.mean([lm[i][1] for i in LEFT_EYE])
    right_eye_center_y = np.mean([lm[i][1] for i in RIGHT_EYE])
    left_brow_y = np.mean([lm[i][1] for i in LEFT_EYEBROW])
    right_brow_y = np.mean([lm[i][1] for i in RIGHT_EYEBROW])
    features['brow_height_left'] = (left_eye_center_y - left_brow_y) / ref_dist
    features['brow_height_right'] = (right_eye_center_y - right_brow_y) / ref_dist
    features['brow_height_avg'] = (features['brow_height_left'] + features['brow_height_right']) / 2

    # 4. Eyebrow angle (찡그림) — 내측 vs 외측 높이 차이
    left_brow_inner_y = lm[LEFT_EYEBROW[0]][1]
    left_brow_outer_y = lm[LEFT_EYEBROW[-1]][1]
    right_brow_inner_y = lm[RIGHT_EYEBROW[0]][1]
    right_brow_outer_y = lm[RIGHT_EYEBROW[-1]][1]
    features['brow_furrow'] = ((left_brow_inner_y - left_brow_outer_y) +
                                (right_brow_inner_y - right_brow_outer_y)) / (2 * ref_dist)

    # 5. Nose wrinkle region height
    nose_bridge_len = compute_distance(lm[NOSE_BRIDGE[0]], lm[NOSE_BRIDGE[-1]])
    features['nose_bridge'] = nose_bridge_len / ref_dist

    # 6. Cheek raise (볼 올라감)
    left_cheek_to_eye = lm[LEFT_CHEEK][1] - left_eye_center_y
    right_cheek_to_eye = lm[RIGHT_CHEEK][1] - right_eye_center_y
    features['cheek_raise_left'] = left_cheek_to_eye / ref_dist
    features['cheek_raise_right'] = right_cheek_to_eye / ref_dist
    features['cheek_raise_avg'] = (features['cheek_raise_left'] + features['cheek_raise_right']) / 2

    # 7. Lip corner angle (입꼬리 — 미소면 올라감)
    mouth_center_y = (lm[LIP_TOP][1] + lm[LIP_BOTTOM][1]) / 2
    lip_left_y = lm[LIP_LEFT][1]
    lip_right_y = lm[LIP_RIGHT][1]
    features['lip_corner_angle'] = (mouth_center_y - (lip_left_y + lip_right_y) / 2) / ref_dist

    # 8. Face aspect ratio
    face_h = compute_distance(lm[FACE_TOP], lm[FACE_BOTTOM])
    face_w = compute_distance(lm[FACE_LEFT], lm[FACE_RIGHT])
    features['face_aspect_ratio'] = face_h / max(face_w, 1e-6)

    # 9. Chin length (코끝 ~ 턱끝)
    features['chin_length'] = compute_distance(lm[NOSE_TIP], lm[FACE_BOTTOM]) / ref_dist

    # 10. Forehead height (이마 ~ 눈썹)
    forehead_y = lm[FACE_TOP][1]
    brow_avg_y = (left_brow_y + right_brow_y) / 2
    features['forehead_height'] = (brow_avg_y - forehead_y) / ref_dist

    return features


def process_images(
    csv_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv",
    output_path: str = "/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv",
    max_images: int = 0,  # 0 = all
):
    """연세대 검증 이미지에서 얼굴 특징 추출."""

    df = pd.read_csv(csv_path)
    if max_images > 0:
        df = df.head(max_images)
    print(f"Processing {len(df):,} images...")

    # MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    feature_names = None
    results = []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        path = row['path']
        try:
            img = np.array(Image.open(path).convert('RGB'))
            result = face_mesh.process(img)

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]
                h, w = img.shape[:2]
                landmarks = [(l.x * w, l.y * h) for l in lm.landmark]

                features = extract_features_from_landmarks(landmarks)
                if features:
                    if feature_names is None:
                        feature_names = sorted(features.keys())
                    features['path'] = path
                    features['emotion'] = row.get('emotion', '')
                    features['is_selected'] = row.get('is_selected', -1)
                    features['n_evals'] = row.get('n_evals', 0)
                    results.append(features)
                else:
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

    face_mesh.close()

    print(f"\nSuccess: {len(results):,}, Failed: {failed:,}")

    # Save
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"Features: {feature_names}")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    process_images(max_images=args.max)
