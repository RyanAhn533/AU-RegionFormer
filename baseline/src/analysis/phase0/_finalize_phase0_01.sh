#!/bin/bash
# Phase 0.1 완료 후 후처리:
#   1. 한글 폰트로 centroid heatmap 재렌더
#   2. summary.md 최종 확인 출력
set -e
cd /home/ajy/AU-RegionFormer

echo "=== [1/2] Re-rendering Korean labels ==="
python -u src/analysis/phase0/01b_rerender_korean.py

echo ""
echo "=== [2/2] Summary ==="
cat outputs/phase0/01_au_embedding_diag/summary.md
