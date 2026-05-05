#!/usr/bin/env python3
"""Noise-aware training entry point (v5: soft label + sample weight)."""

import sys
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from training.noise_aware_trainer import train_noise_aware


def main():
    ap = argparse.ArgumentParser(description="Train AU-FER Model (Noise-Aware)")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    train_noise_aware(args.config, args.resume)


if __name__ == "__main__":
    main()
