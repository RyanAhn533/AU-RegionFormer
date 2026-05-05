"""
얼굴 특징 추출 — 고속 버전
==============================
1. 얼굴 먼저 crop (MTCNN or MediaPipe face detection) → 작은 이미지
2. crop된 얼굴에서 FaceMesh landmark 추출
3. 멀티프로세스 (CPU 병렬)
4. 중간 저장 (1만장마다)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import warnings
warnings.filterwarnings('ignore')


# Landmark indices (same as before)
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


def extract_one(row_dict):
    """한 이미지에서 특징 추출. 멀티프로세스용."""
    path = row_dict['path']
    try:
        img = Image.open(path).convert('RGB')

        # 큰 이미지는 resize (얼굴 비율 유지, landmark 정확도 유지)
        w, h = img.size
        max_dim = 640
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        img_np = np.array(img)

        # FaceMesh (per-process instance는 worker_init에서 생성)
        result = _face_mesh.process(img_np)

        if not result.multi_face_landmarks:
            return None

        lm_raw = result.multi_face_landmarks[0]
        ih, iw = img_np.shape[:2]
        lm = [(l.x*iw, l.y*ih) for l in lm_raw.landmark]

        ref = dist(lm[LEFT_EYE_INNER], lm[RIGHT_EYE_INNER])
        if ref < 1e-6:
            return None

        f = {}
        f['ear_left'] = ear(LEFT_EYE, lm)
        f['ear_right'] = ear(RIGHT_EYE, lm)
        f['ear_avg'] = (f['ear_left']+f['ear_right'])/2

        mouth_h = dist(lm[LIP_TOP], lm[LIP_BOTTOM])
        mouth_w = dist(lm[LIP_LEFT], lm[LIP_RIGHT])
        f['mar'] = mouth_h / max(mouth_w, 1e-6)
        f['mouth_width'] = mouth_w / ref

        ley = np.mean([lm[i][1] for i in LEFT_EYE])
        rey = np.mean([lm[i][1] for i in RIGHT_EYE])
        lby = np.mean([lm[i][1] for i in LEFT_EYEBROW])
        rby = np.mean([lm[i][1] for i in RIGHT_EYEBROW])
        f['brow_height_left'] = (ley-lby)/ref
        f['brow_height_right'] = (rey-rby)/ref
        f['brow_height_avg'] = (f['brow_height_left']+f['brow_height_right'])/2

        f['brow_furrow'] = ((lm[LEFT_EYEBROW[0]][1]-lm[LEFT_EYEBROW[-1]][1]) +
                            (lm[RIGHT_EYEBROW[0]][1]-lm[RIGHT_EYEBROW[-1]][1])) / (2*ref)

        f['nose_bridge'] = dist(lm[NOSE_BRIDGE[0]], lm[NOSE_BRIDGE[-1]]) / ref

        f['cheek_raise_left'] = (lm[LEFT_CHEEK][1]-ley)/ref
        f['cheek_raise_right'] = (lm[RIGHT_CHEEK][1]-rey)/ref
        f['cheek_raise_avg'] = (f['cheek_raise_left']+f['cheek_raise_right'])/2

        mc_y = (lm[LIP_TOP][1]+lm[LIP_BOTTOM][1])/2
        f['lip_corner_angle'] = (mc_y-(lm[LIP_LEFT][1]+lm[LIP_RIGHT][1])/2)/ref

        face_h = dist(lm[FACE_TOP], lm[FACE_BOTTOM])
        face_w = dist(lm[FACE_LEFT], lm[FACE_RIGHT])
        f['face_aspect_ratio'] = face_h/max(face_w, 1e-6)
        f['chin_length'] = dist(lm[NOSE_TIP], lm[FACE_BOTTOM])/ref

        brow_avg_y = (lby+rby)/2
        f['forehead_height'] = (brow_avg_y-lm[FACE_TOP][1])/ref

        f['path'] = path
        f['emotion'] = row_dict.get('emotion', '')
        f['is_selected'] = row_dict.get('is_selected', -1)
        return f
    except:
        return None


_face_mesh = None
def worker_init():
    global _face_mesh
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )


def main(
    csv_path="/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv",
    output_path="/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv",
    n_workers=16,
    chunk_size=10000,
):
    df = pd.read_csv(csv_path)
    rows = df.to_dict('records')
    total = len(rows)
    print(f"Total: {total:,}, Workers: {n_workers}")

    all_results = []
    t0 = time.time()

    with Pool(n_workers, initializer=worker_init) as pool:
        for i, result in enumerate(pool.imap(extract_one, rows, chunksize=64)):
            if result is not None:
                all_results.append(result)

            if (i+1) % 5000 == 0:
                elapsed = time.time()-t0
                speed = (i+1)/elapsed
                eta = (total-i-1)/speed
                success_rate = len(all_results)/(i+1)*100
                print(f"[{i+1:,}/{total:,}] {speed:.1f} it/s | "
                      f"success={success_rate:.1f}% | "
                      f"ETA={eta/60:.0f}min")

            # 중간 저장
            if (i+1) % chunk_size == 0 and all_results:
                tmp = pd.DataFrame(all_results)
                tmp.to_csv(output_path + f".part_{(i+1)//chunk_size}", index=False)
                print(f"  Saved checkpoint: {len(all_results):,} results")

    # 최종 저장
    if all_results:
        out_df = pd.DataFrame(all_results)
        out_df.to_csv(output_path, index=False)
        elapsed = time.time()-t0
        print(f"\nDone: {len(all_results):,}/{total:,} ({len(all_results)/total*100:.1f}%)")
        print(f"Time: {elapsed/60:.1f}min ({elapsed/len(all_results)*1000:.1f}ms/img)")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
