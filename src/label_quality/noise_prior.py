"""
Phase 1: Metadata-based Noise Prior
=====================================
원본 annotator 3인 라벨 + 연세대 검증 → 사진별 P(noise) 계산.

3개 신호:
  1. annotator 3인 완전 일치 여부
  2. majority label = uploader label 여부
  3. 피험자 유형 (전문인/일반인)

각 조건 조합의 경험적 noise율을 lookup table로 적용.
413K AU-RegionFormer 데이터에 hash 매칭으로 전파.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path


# 3중 교차 검증에서 확인된 경험적 noise율 (is_selected=1 비율)
# key: (full_agree, majority_match, isProf)
NOISE_RATE_TABLE = {
    (True,  True,  "전문인"): 0.087,
    (True,  True,  "일반인"): 0.109,
    (True,  False, "전문인"): 0.108,
    (True,  False, "일반인"): 0.215,
    (False, True,  "전문인"): 0.125,
    (False, True,  "일반인"): 0.120,
    (False, False, "전문인"): 0.124,
    (False, False, "일반인"): 0.157,
}

# 감정별 baseline noise율 (조건 매칭 불가 시 fallback)
EMOTION_NOISE_BASELINE = {
    "happy": 0.065, "angry": 0.167, "sad": 0.164, "neutral": 0.069,
    "anxious": 0.167, "hurt": 0.164, "surprised": 0.065,  # proxy
}


def load_annotator_data(
    training_base: str = "/home/ajy/shared_ssd2/shared_ssd2/한국인 감정인식을 위한 복합 영상/Training",
) -> dict:
    """원본 JSON에서 annotator 메타데이터 추출. hash64 → info dict."""
    data = {}
    for folder in os.listdir(training_base):
        fp = os.path.join(training_base, folder)
        if not os.path.isdir(fp):
            continue
        for sub in os.listdir(fp):
            sp = os.path.join(fp, sub)
            if not os.path.isdir(sp):
                continue
            for f in os.listdir(sp):
                if not f.endswith(".json"):
                    continue
                with open(os.path.join(sp, f), "rb") as jf:
                    items = json.loads(jf.read())
                for item in items:
                    labels = [item["annot_A"]["faceExp"],
                              item["annot_B"]["faceExp"],
                              item["annot_C"]["faceExp"]]
                    n_agree = max(Counter(labels).values())
                    majority = Counter(labels).most_common(1)[0][0]
                    h = item["filename"][:64]
                    data[h] = {
                        "full_agree": labels[0] == labels[1] == labels[2],
                        "majority_match": majority == item["faceExp_uploader"],
                        "isProf": item["isProf"],
                        "annot_labels": labels,
                        "uploader_label": item["faceExp_uploader"],
                        "majority_label": majority,
                    }
    return data


def compute_noise_prior(
    train_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train.csv",
    output_csv: str = "/mnt/hdd/ajy_25/au_csv/index_train_with_prior.csv",
):
    """413K 학습 데이터에 noise prior 부여."""

    print("Loading annotator data...")
    annot_data = load_annotator_data()
    print(f"  Annotator records: {len(annot_data):,}")

    print("Loading training data...")
    df = pd.read_csv(train_csv)
    df["hash64"] = df["path"].apply(lambda p: os.path.basename(p)[:64])
    print(f"  Training samples: {len(df):,}")

    # 매칭
    matched = 0
    noise_priors = []
    soft_labels_list = []

    emotion_kr2en = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral",
                     "당황": "surprised", "불안": "anxious", "상처": "hurt"}
    emotion_all = sorted(df["label"].unique())

    for _, row in df.iterrows():
        h = row["hash64"]
        label = row["label"]

        if h in annot_data:
            info = annot_data[h]
            key = (info["full_agree"], info["majority_match"], info["isProf"])
            noise_p = NOISE_RATE_TABLE.get(key, EMOTION_NOISE_BASELINE.get(label, 0.12))
            matched += 1

            # Soft label: annotator 3인 라벨 분포
            annot_labels_en = []
            for al in info["annot_labels"]:
                en = emotion_kr2en.get(al)
                if en and en in emotion_all:
                    annot_labels_en.append(en)
            if annot_labels_en:
                dist = Counter(annot_labels_en)
                total = sum(dist.values())
                soft = {e: dist.get(e, 0) / total for e in emotion_all}
            else:
                soft = {e: (1.0 if e == label else 0.0) for e in emotion_all}
        else:
            noise_p = EMOTION_NOISE_BASELINE.get(label, 0.12)
            soft = {e: (1.0 if e == label else 0.0) for e in emotion_all}

        noise_priors.append(noise_p)
        soft_labels_list.append(soft)

    df["noise_prior"] = noise_priors
    df["sample_weight"] = 1.0 - df["noise_prior"]

    # Soft label columns
    for e in emotion_all:
        df[f"soft_{e}"] = [s[e] for s in soft_labels_list]

    # Save
    df.to_csv(output_csv, index=False)

    print(f"\nResults:")
    print(f"  Matched with annotator: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    print(f"  Mean noise prior: {df['noise_prior'].mean():.4f}")
    print(f"  Mean sample weight: {df['sample_weight'].mean():.4f}")

    print(f"\nPer-emotion noise prior:")
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        m = sub[sub["hash64"].isin(annot_data)]
        print(f"  {label}: prior={sub['noise_prior'].mean():.4f}, "
              f"weight={sub['sample_weight'].mean():.4f}, "
              f"matched={len(m):,}/{len(sub):,}")

    # Soft label entropy 확인
    soft_cols = [f"soft_{e}" for e in emotion_all]
    soft_arr = df[soft_cols].values
    entropy = -np.sum(soft_arr * np.log(soft_arr + 1e-10), axis=1)
    df["label_entropy"] = entropy
    print(f"\nLabel entropy: mean={entropy.mean():.4f}, "
          f"pure(0.0)={(entropy < 0.01).sum():,}, "
          f"mixed(>0.5)={(entropy > 0.5).sum():,}")

    # Re-save with entropy
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    return df


if __name__ == "__main__":
    compute_noise_prior()
