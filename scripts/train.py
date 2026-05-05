#!/usr/bin/env python3
"""Training entry point."""

import sys
import argparse
from pathlib import Path

# Add src to path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from training.trainer import train


def main():
    ap = argparse.ArgumentParser(description="Train AU-FER Model")
    ap.add_argument("--config", type=str, required=True,
                    help="Path to config YAML")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--init_from", type=str, default=None,
                    help="Load only model weights (filter shape mismatches), reset optimizer/scheduler/epoch.")
    args = ap.parse_args()
    train(args.config, args.resume, args.init_from)


if __name__ == "__main__":
    main()
