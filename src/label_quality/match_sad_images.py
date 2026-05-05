"""
sad 이미지 원본 매칭
=====================
InsightFace face recognition embedding으로
sad_00001.jpg ↔ 원본 슬픔 이미지 매칭.
"""

import os
import numpy as np
import pandas as pd
import cv2
import time
from insightface.app import FaceAnalysis


def extract_embeddings(image_paths, face_app, desc=""):
    """이미지 리스트에서 face embedding 추출."""
    embeddings = {}
    failed = 0
    for i, path in enumerate(image_paths):
        try:
            img = cv2.imread(path)
            if img is None:
                failed += 1; continue
            faces = face_app.get(img)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                embeddings[path] = face.normed_embedding
            else:
                failed += 1
        except:
            failed += 1

        if (i+1) % 5000 == 0:
            print(f"  [{desc}] {i+1:,}/{len(image_paths):,} | failed={failed}")

    print(f"  [{desc}] Done: {len(embeddings):,}/{len(image_paths):,} | failed={failed}")
    return embeddings


def main():
    print("Loading InsightFace...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 1. sad renamed 이미지
    sad_dir = "/home/ajy/AI_hub_250704/data2/data_processed_korea/sad"
    sad_files = sorted([os.path.join(sad_dir, f) for f in os.listdir(sad_dir) if f.endswith('.jpg')])
    print(f"Sad renamed: {len(sad_files):,}")

    # 2. 원본 슬픔 이미지
    orig_base = "/home/ajy/shared_ssd2/shared_ssd2/한국인 감정인식을 위한 복합 영상/Training/슬픔"
    orig_files = []
    for sub in os.listdir(orig_base):
        sp = os.path.join(orig_base, sub)
        if os.path.isdir(sp) and '원천' in sub:
            for f in os.listdir(sp):
                if f.endswith(('.jpg', '.jpeg')):
                    orig_files.append(os.path.join(sp, f))
    orig_files.sort()
    print(f"Original: {len(orig_files):,}")

    t0 = time.time()

    # Embedding 추출
    print("\nExtracting sad embeddings...")
    sad_emb = extract_embeddings(sad_files, app, desc="sad")

    print("\nExtracting original embeddings...")
    orig_emb = extract_embeddings(orig_files, app, desc="orig")

    # 매칭: 각 sad에 대해 가장 가까운 orig 찾기
    print(f"\nMatching {len(sad_emb):,} sad → {len(orig_emb):,} orig...")

    sad_paths = list(sad_emb.keys())
    sad_vecs = np.array([sad_emb[p] for p in sad_paths])

    orig_paths = list(orig_emb.keys())
    orig_vecs = np.array([orig_emb[p] for p in orig_paths])

    # Batch cosine similarity
    # sad_vecs: [N, 512], orig_vecs: [M, 512]
    # 메모리 절약: 청크로 나눠서
    chunk_size = 1000
    matches = []

    for i in range(0, len(sad_vecs), chunk_size):
        chunk = sad_vecs[i:i+chunk_size]  # [chunk, 512]
        sims = chunk @ orig_vecs.T  # [chunk, M]
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(chunk)), best_idx]

        for j in range(len(chunk)):
            matches.append({
                'sad_path': sad_paths[i+j],
                'sad_fname': os.path.basename(sad_paths[i+j]),
                'orig_path': orig_paths[best_idx[j]],
                'orig_fname': os.path.basename(orig_paths[best_idx[j]]),
                'similarity': float(best_sim[j]),
            })

        if (i + chunk_size) % 10000 == 0 or i + chunk_size >= len(sad_vecs):
            print(f"  Matched {min(i+chunk_size, len(sad_vecs)):,}/{len(sad_vecs):,}")

    # 결과 저장
    match_df = pd.DataFrame(matches)
    out_path = "/home/ajy/AU-RegionFormer/data/label_quality/sad_matching.csv"
    match_df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}min")
    print(f"Saved: {out_path}")
    print(f"\nSimilarity stats:")
    print(f"  Mean: {match_df['similarity'].mean():.4f}")
    print(f"  >0.9: {(match_df['similarity']>0.9).sum():,} ({(match_df['similarity']>0.9).mean()*100:.1f}%)")
    print(f"  >0.8: {(match_df['similarity']>0.8).sum():,} ({(match_df['similarity']>0.8).mean()*100:.1f}%)")
    print(f"  >0.5: {(match_df['similarity']>0.5).sum():,} ({(match_df['similarity']>0.5).mean()*100:.1f}%)")
    print(f"  <0.3: {(match_df['similarity']<0.3).sum():,} (매칭 실패 의심)")


if __name__ == "__main__":
    main()
