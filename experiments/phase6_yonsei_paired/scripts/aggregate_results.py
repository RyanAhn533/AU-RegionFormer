#!/usr/bin/env python3
"""
Aggregate all Phase 6 results across Stages 1-13 into a single comparison table.

Reads:
  - state/leaderboard.jsonl (training F1)
  - results/stage9_*/stage9_summary.json (alignment metrics)
  - results/xeval_*/crossdataset_summary.json (cross-cultural F1)

Outputs:
  - results/AGGREGATED_RESULTS_YYYYMMDD.md (markdown table for paper draft)
  - results/aggregated.json (machine-readable)
"""
import json
from pathlib import Path
from datetime import datetime

PHASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = PHASE / "results"
LB = PHASE / "state" / "leaderboard.jsonl"


def load_lb():
    out = {}
    if not LB.exists(): return out
    for line in open(LB):
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
            stage = d.get("stage", "")
            if stage and stage not in out:
                out[stage] = d
            elif stage in out:
                # keep the later entry if newer
                out[stage] = d
        except Exception:
            pass
    return out


def load_stage9():
    out = {}
    for d in sorted(RESULTS_DIR.glob("stage9_*")):
        if not d.is_dir(): continue
        f = d / "stage9_summary.json"
        if not f.exists(): continue
        name = d.name.replace("stage9_", "")
        try:
            out[name] = json.load(open(f))
        except Exception:
            pass
    return out


def load_xeval():
    out = {}
    for d in sorted(RESULTS_DIR.glob("xeval_*")):
        if not d.is_dir(): continue
        f = d / "crossdataset_summary.json"
        if not f.exists(): continue
        try:
            out[d.name] = json.load(open(f))
        except Exception:
            pass
    return out


def main():
    lb = load_lb()
    s9 = load_stage9()
    xe = load_xeval()

    today = datetime.now().strftime("%Y-%m-%d")
    md = [f"# Phase 6 Aggregated Results — {today}\n"]

    md.append("## 1. Training F1 (Korean val, master_val.csv)\n")
    md.append("| Stage | Best F1 | Best Acc | Time | Status |")
    md.append("|---|---|---|---|---|")
    for stage in sorted(lb.keys()):
        d = lb[stage]
        m = d.get("metrics", {}) or {}
        md.append(f"| {stage} | {m.get('best_f1','')} | {m.get('best_val_acc', m.get('best_acc',''))} | "
                  f"{m.get('total_time_hr','')}h | {d.get('status','')} |")

    md.append("\n## 2. Stage 9 Alignment (Korean perception calibration)\n")
    md.append("| Ckpt | C1 Spearman (per-class) | C2 Pearson (per-image) | C5 calibration |")
    md.append("|---|---|---|---|")
    for name, s in s9.items():
        c1 = s.get("C1_per_class_spearman", {}).get("spearman_rho", "")
        c2 = s.get("C2_per_image_pearson", {}).get("pearson_r", "")
        c5 = s.get("C5_human_accept_vs_softmax", {}).get("pearson_r", "")
        md.append(f"| {name} | {c1} | {c2} | {c5} |")

    md.append("\n## 3. Cross-cultural Zero-shot (Korean → Western)\n")
    md.append("| Ckpt | AffectNet F1 | AffectNet Acc | SFEW F1 | SFEW Acc |")
    md.append("|---|---|---|---|---|")
    # group by ckpt
    by_ckpt = {}
    for name, s in xe.items():
        # name format: xeval_{tag}_{ckpt_name}
        parts = name.split("_", 2)
        if len(parts) < 3: continue
        tag = parts[1]
        ckpt = parts[2]
        by_ckpt.setdefault(ckpt, {})[tag] = s
    for ckpt in sorted(by_ckpt):
        s_aff = by_ckpt[ckpt].get("affectnet", {})
        s_sfw = by_ckpt[ckpt].get("sfew", {})
        md.append(f"| {ckpt} | {s_aff.get('f1_macro','')} | {s_aff.get('accuracy','')} | "
                  f"{s_sfw.get('f1_macro','')} | {s_sfw.get('accuracy','')} |")

    out_md = RESULTS_DIR / f"AGGREGATED_RESULTS_{today}.md"
    out_json = RESULTS_DIR / "aggregated.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    json.dump({"leaderboard": lb, "stage9": s9, "xeval": xe},
              open(out_json, "w"), indent=2, ensure_ascii=False)
    print(f"saved {out_md}\nsaved {out_json}")


if __name__ == "__main__":
    main()
