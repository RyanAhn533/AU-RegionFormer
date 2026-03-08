"""
V4-lite Dataset — Audio + Bio Raw Signals
===========================================
K-EmoCon 전처리 데이터에서 audio logmel + raw bio 시계열 로딩.

Segment npz 구조 (5초, 32Hz = 160 samples):
  bvp:  (160, 2) → col0=value, col1=timestamp
  eda:  (160, 2) → col0=value, col1=timestamp
  temp: (160, 2) → col0=value, col1=timestamp
  acc:  (160, 4) → col0=x, col1=y, col2=z, col3=timestamp
  hr:   (160, 2) → col0=value, col1=timestamp

→ 값만 추출: 7ch × 160 timesteps

Audio logmel: audio_mel/pid_X/seg_XXXXX.npz → logmel key
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _load_bio_raw(npz_path: Path, target_len: int = 160) -> np.ndarray:
    """
    Segment npz → (7, T) bio signal array.

    Channels:
      0: bvp (Blood Volume Pulse)
      1: eda (Electrodermal Activity)
      2: temp (Skin Temperature)
      3: acc_x (Accelerometer X)
      4: acc_y (Accelerometer Y)
      5: acc_z (Accelerometer Z)
      6: hr (Heart Rate)

    모든 채널은 이미 32Hz로 리샘플되어 160 samples.
    """
    data = np.load(npz_path, allow_pickle=True)

    channels = []

    # bvp, eda, temp, hr: col0 = value
    for key in ["bvp", "eda", "temp"]:
        if key in data:
            vals = data[key][:, 0].astype(np.float32)
            channels.append(vals)
        else:
            channels.append(np.zeros(target_len, dtype=np.float32))

    # acc: col0=x, col1=y, col2=z
    if "acc" in data:
        acc = data["acc"][:, :3].astype(np.float32)
        channels.append(acc[:, 0])  # x
        channels.append(acc[:, 1])  # y
        channels.append(acc[:, 2])  # z
    else:
        for _ in range(3):
            channels.append(np.zeros(target_len, dtype=np.float32))

    # hr
    if "hr" in data:
        channels.append(data["hr"][:, 0].astype(np.float32))
    else:
        channels.append(np.zeros(target_len, dtype=np.float32))

    # Stack: (7, T)
    bio = np.stack(channels, axis=0)

    # Ensure target length
    if bio.shape[1] != target_len:
        # Simple linear interpolation
        bio_t = torch.from_numpy(bio).unsqueeze(0)  # (1, 7, T_orig)
        bio_t = F.interpolate(bio_t, size=target_len, mode="linear", align_corners=False)
        bio = bio_t.squeeze(0).numpy()

    # Replace NaN/Inf with 0
    bio = np.nan_to_num(bio, nan=0.0, posinf=0.0, neginf=0.0)

    return bio


def _load_logmel(npz_path: Path, target_time_bins: int = 128) -> np.ndarray:
    """
    Audio mel npz → (1, 64, T) logmel.
    """
    data = np.load(npz_path, allow_pickle=True)
    logmel = data["logmel"].astype(np.float32)  # (64, T_orig)

    # Resize time to target
    mel_t = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, T)
    mel_t = F.interpolate(mel_t, size=(64, target_time_bins),
                          mode="bilinear", align_corners=False)
    return mel_t.squeeze(0).numpy()  # (1, 64, target_time_bins)


def _infer_conversation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """K-EmoCon 대화 쌍 추론 (pid 정렬 후 연속 쌍)."""
    df = df.copy()
    all_pids = sorted(df["pid"].unique())

    pid_pairs = {}
    for i in range(0, len(all_pids) - 1, 2):
        p1, p2 = all_pids[i], all_pids[i + 1]
        pid_pairs[p1] = p2
        pid_pairs[p2] = p1
    for p in all_pids:
        if p not in pid_pairs:
            pid_pairs[p] = p

    df["conversation_id"] = df["pid"].apply(lambda p: min(p, pid_pairs.get(p, p)))
    df["speaker_id"] = 0
    for conv_id in df["conversation_id"].unique():
        mask = df["conversation_id"] == conv_id
        pids_sorted = sorted(df.loc[mask, "pid"].unique())
        if len(pids_sorted) >= 2:
            df.loc[mask & (df["pid"] == pids_sorted[1]), "speaker_id"] = 1

    return df


class AVLiteDataset(Dataset):
    """
    V4-lite Dataset: audio logmel + raw bio signals → A/V binary labels.

    Args:
        df: segments_index DataFrame (filtered)
        base_dir: preprocessed data root
        bio_len: bio signal length (default 160 = 5sec × 32Hz)
        mel_time_bins: logmel time bins after resize
        av_threshold: binarization threshold (0.0 for normalized labels)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str,
        bio_len: int = 160,
        mel_time_bins: int = 128,
        av_threshold: float = 0.0,
    ):
        self.base_dir = Path(base_dir)
        self.bio_len = bio_len
        self.mel_time_bins = mel_time_bins
        self.threshold = av_threshold

        # Filter: need audio_mel + bio segment
        df = df.copy()
        if "has_audio_mel" in df.columns:
            df = df[df["has_audio_mel"] == 1]
        df = df.dropna(subset=["label_ext_A_norm", "label_ext_V_norm"])

        # Add speaker info
        df = _infer_conversation_pairs(df)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = int(row["pid"])
        seg_idx = int(row.get("seg_idx", 0))

        # ── Bio raw signals ──
        seg_path = self.base_dir / row["path"]  # segments/pid_X/seg_XXXXX.npz
        bio_raw = _load_bio_raw(seg_path, self.bio_len)  # (7, 160)
        bio_raw = torch.from_numpy(bio_raw).float()

        # ── Audio logmel ──
        mel_path = self.base_dir / row["audio_mel_path"]
        logmel = _load_logmel(mel_path, self.mel_time_bins)  # (1, 64, T)
        logmel = torch.from_numpy(logmel).float()

        # ── Labels (binary) ──
        a_norm = float(row["label_ext_A_norm"])
        v_norm = float(row["label_ext_V_norm"])
        label_A = 1.0 if a_norm >= self.threshold else 0.0
        label_V = 1.0 if v_norm >= self.threshold else 0.0

        return {
            "bio_raw": bio_raw,           # (7, 160)
            "logmel": logmel,             # (1, 64, T)
            "label_A": torch.tensor(label_A, dtype=torch.float32),
            "label_V": torch.tensor(label_V, dtype=torch.float32),
            "a_norm": torch.tensor(a_norm, dtype=torch.float32),
            "v_norm": torch.tensor(v_norm, dtype=torch.float32),
            "pid": pid,
            "seg_idx": seg_idx,
            "speaker_id": int(row.get("speaker_id", 0)),
        }


def collate_av_lite(batch):
    """V4-lite collate function."""
    return {
        "bio_raw": torch.stack([b["bio_raw"] for b in batch]),
        "logmel": torch.stack([b["logmel"] for b in batch]),
        "label_A": torch.stack([b["label_A"] for b in batch]),
        "label_V": torch.stack([b["label_V"] for b in batch]),
        "a_norm": torch.stack([b["a_norm"] for b in batch]),
        "v_norm": torch.stack([b["v_norm"] for b in batch]),
        "pids": torch.tensor([b["pid"] for b in batch], dtype=torch.long),
        "speaker_ids": torch.tensor([b["speaker_id"] for b in batch], dtype=torch.long),
    }
