"""
Label Quality Dataset Preparation
===================================
연세대 검증 데이터(298명 평가) → label quality classifier 학습용 데이터 생성.

Output:
  - train.csv: 학습용 (1명 평가 사진)
  - val_consensus.csv: 검증용 (2명 이상 평가, 합의된 사진만)

Columns:
  path, emotion, is_selected, n_evals, n_selected
"""

import pandas as pd
import os
from pathlib import Path


def prepare(
    csv_path: str = "/home/ajy/shared_ssd2/shared_ssd2/한국인 감정인식을 위한 복합 영상/python_code/final_result.csv",
    output_dir: str = "/home/ajy/AU-RegionFormer/data/label_quality",
    prefix_old: str = "/home/user/다운로드/한국인 감정인식을 위한 복합 영상",
    prefix_new: str = "/home/ajy/shared_ssd2/shared_ssd2/한국인 감정인식을 위한 복합 영상",
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Raw responses: {len(df):,}")

    # 경로 변환 — merged_files 경로를 원천 폴더에서 탐색
    df["emotion"] = df["presented_photo_path"].str.extract(r"Training/([^/]+)/")
    df["fname"] = df["presented_photo_path"].apply(os.path.basename)

    # 원천 폴더별 파일 인덱스 구축
    print("Building file index from source folders...")
    file_index = {}  # filename → local_path
    training_base = os.path.join(prefix_new, "Training")
    for emo_folder in os.listdir(training_base):
        emo_path = os.path.join(training_base, emo_folder)
        if not os.path.isdir(emo_path):
            continue
        for sub in os.listdir(emo_path):
            sub_path = os.path.join(emo_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for f in os.listdir(sub_path):
                if f.endswith((".jpg", ".jpeg", ".png")):
                    file_index[f] = os.path.join(sub_path, f)
    print(f"  Indexed {len(file_index):,} files")

    # 파일명으로 매칭
    df["local_path"] = df["fname"].map(file_index)
    df["exists"] = df["local_path"].notna()
    n_before = len(df)
    df = df[df["exists"]].copy()
    print(f"Accessible: {len(df):,} / {n_before:,} ({len(df)/n_before*100:.1f}%)")

    # 극단적 평가자 제거 (원본 기준 선택률 2% 미만)
    # exists 필터 전의 전체 데이터로 평가자 품질 판단
    df_full = pd.read_csv(csv_path, low_memory=False)
    eval_stats = df_full.groupby("student_id")["is_selected"].agg(["count", "sum"]).reset_index()
    eval_stats["rate"] = eval_stats["sum"] / eval_stats["count"]
    bad_evals = set(eval_stats[eval_stats["rate"] < 0.02]["student_id"])
    df = df[~df["student_id"].isin(bad_evals)]
    print(f"After evaluator filter: {len(df):,} (removed {len(bad_evals)} evaluators)")

    # 사진별 집계
    photo_agg = df.groupby("local_path").agg(
        emotion=("emotion", "first"),
        n_evals=("is_selected", "count"),
        n_selected=("is_selected", "sum"),
    ).reset_index()
    photo_agg.rename(columns={"local_path": "path"}, inplace=True)
    photo_agg["is_selected"] = (photo_agg["n_selected"] > 0).astype(int)

    print(f"\nTotal unique photos: {len(photo_agg):,}")
    print(f"  Selected (is_selected=1): {photo_agg['is_selected'].sum():,}")
    print(f"  Not selected: {(1 - photo_agg['is_selected']).sum():,}")

    # Split: 2명+ 평가 → val_consensus, 1명 → train
    multi = photo_agg[photo_agg["n_evals"] >= 2].copy()
    single = photo_agg[photo_agg["n_evals"] == 1].copy()

    # Consensus: 2명 이상 전원 동의한 것만 (불일치 제거)
    consensus = multi[
        (multi["n_selected"] == 0) | (multi["n_selected"] == multi["n_evals"])
    ].copy()

    print(f"\n=== Split ===")
    print(f"Train (1-eval): {len(single):,}")
    print(f"Val consensus (2+ eval, agreed): {len(consensus):,}")
    print(f"  - Consensus selected: {consensus['is_selected'].sum():,}")
    print(f"  - Consensus not-selected: {(1 - consensus['is_selected']).sum():,}")
    print(f"Discarded (2+ eval, disagreed): {len(multi) - len(consensus):,}")

    # 감정별 분포
    print(f"\n=== Per-emotion ===")
    for emo in sorted(photo_agg["emotion"].dropna().unique()):
        tr = single[single["emotion"] == emo]
        va = consensus[consensus["emotion"] == emo]
        print(f"  {emo}: train={len(tr):,} (sel={tr['is_selected'].sum():,}), "
              f"val={len(va):,} (sel={va['is_selected'].sum():,})")

    # Save
    single.to_csv(f"{output_dir}/train.csv", index=False)
    consensus.to_csv(f"{output_dir}/val_consensus.csv", index=False)
    photo_agg.to_csv(f"{output_dir}/all_photos.csv", index=False)

    print(f"\nSaved to {output_dir}/")
    return single, consensus


if __name__ == "__main__":
    prepare()
