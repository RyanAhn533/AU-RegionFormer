"""
Phase 0.3 — Per-AU Linear Probe

AU 단위로 감정 4-class 구분력 측정:
  - Per-AU 단독 ranking (41 AU × scalar feature)
  - All-AU 41d
  - Eye-AU group vs Mouth-AU group (Jack 2012 재검증)
  - FACS canonical subset (Ekman 재현도)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INPUT = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path("/home/ajy/AU-RegionFormer/outputs/phase0/03_per_au_linear_probe_v2")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# 41 AU columns in the parquet
AU_NAMES = [
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU11","AU12",
    "AU13","AU14","AU15","AU16","AU17","AU18","AU19","AU20","AU22","AU23",
    "AU24","AU25","AU26","AU27","AU32","AU38","AU39",
    "AUL1","AUR1","AUL2","AUR2","AUL4","AUR4","AUL6","AUR6",
    "AUL10","AUR10","AUL12","AUR12","AUL14","AUR14",
]

# Jack 2012 ↔ AU mapping
#  Eye region AU (Jack 2012가 East Asian core로 본 영역)
EYE_AU = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7",
          "AUL1", "AUR1", "AUL2", "AUR2", "AUL4", "AUR4", "AUL6", "AUR6"]
#  Mouth region AU (AI가 top으로 본 영역 — Phase 0.1c 결과)
MOUTH_AU = ["AU10", "AU12", "AU14", "AU15", "AU17", "AU18", "AU20",
            "AU22", "AU23", "AU24", "AU25", "AU26", "AU27",
            "AUL10", "AUR10", "AUL12", "AUR12", "AUL14", "AUR14"]

# Ekman canonical mapping
EKMAN_MAP = {
    "기쁨 (Happy, AU6+12)":   ["AU6", "AU12"],
    "슬픔 (Sad, AU1+4+15)":   ["AU1", "AU4", "AU15"],
    "분노 (Anger, AU4+5+7+23)": ["AU4", "AU5", "AU7", "AU23"],
}


def subsample_stratified(df, n_sub, y_col, seed=SEED):
    rng = np.random.default_rng(seed)
    per_class = n_sub // df[y_col].nunique()
    parts = []
    for c in df[y_col].unique():
        sub = df[df[y_col] == c]
        if len(sub) > per_class:
            sub = sub.sample(n=per_class, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def linear_probe(X, y, n_folds=3):
    """No PCA — features already low-dim enough."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = StandardScaler().fit_transform(X.astype(np.float32))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs, f1s, cms = [], [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat, average="macro"))
        cms.append(confusion_matrix(y[te], yhat, labels=np.unique(y)))
    cm_mean = np.mean(cms, axis=0)
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "cm_normalized": cm_norm.tolist(),
    }


def main():
    print(f"[info] loading {INPUT}")
    df = pd.read_parquet(INPUT)
    print(f"[info] total: {len(df)}")
    print(f"[info] emotion dist:\n{df['emotion'].value_counts()}")

    # Subsample 30K stratified (Phase 0.1c와 동일)
    df_s = subsample_stratified(df, n_sub=30000, y_col="emotion")
    print(f"[info] subsampled: {len(df_s)}")

    le = LabelEncoder()
    y = le.fit_transform(df_s["emotion"].values)
    classes_ko = le.classes_.tolist()

    results = {}

    # === 3.1 Per-AU 단독 ===
    print("\n=== 3.1 Per-AU 단독 linear probe ===")
    per_au_rows = []
    for au in AU_NAMES:
        if au not in df_s.columns:
            continue
        X = df_s[au].values
        res = linear_probe(X, y, n_folds=3)
        per_au_rows.append({
            "AU": au,
            "acc": res["acc_mean"] * 100,
            "acc_std": res["acc_std"] * 100,
            "f1": res["f1_mean"],
        })
        print(f"  {au:10s}  acc={res['acc_mean']*100:.1f}% F1={res['f1_mean']:.3f}")
    per_au_df = pd.DataFrame(per_au_rows).sort_values("acc", ascending=False)
    per_au_df.to_csv(OUT / "per_au_ranking.csv", index=False)
    results["per_au"] = per_au_rows

    # === 3.2 All-AU 41d ===
    print("\n=== 3.2 All-AU (41d) linear probe ===")
    X_all = df_s[[c for c in AU_NAMES if c in df_s.columns]].values
    res = linear_probe(X_all, y, n_folds=3)
    results["all_au"] = res
    print(f"  All-41AU: acc={res['acc_mean']*100:.1f}% F1={res['f1_mean']:.3f}")

    # === 3.3 Eye-AU vs Mouth-AU ===
    print("\n=== 3.3 Eye-AU vs Mouth-AU (Jack 2012 재검증) ===")
    eye_cols = [c for c in EYE_AU if c in df_s.columns]
    mouth_cols = [c for c in MOUTH_AU if c in df_s.columns]
    X_eye = df_s[eye_cols].values
    X_mouth = df_s[mouth_cols].values
    res_eye = linear_probe(X_eye, y, n_folds=3)
    res_mouth = linear_probe(X_mouth, y, n_folds=3)
    results["eye_group"] = {**res_eye, "n_feature": len(eye_cols), "au_list": eye_cols}
    results["mouth_group"] = {**res_mouth, "n_feature": len(mouth_cols), "au_list": mouth_cols}
    print(f"  Eye-AU  ({len(eye_cols)} features): acc={res_eye['acc_mean']*100:.1f}% F1={res_eye['f1_mean']:.3f}")
    print(f"  Mouth-AU ({len(mouth_cols)} features): acc={res_mouth['acc_mean']*100:.1f}% F1={res_mouth['f1_mean']:.3f}")
    diff = (res_mouth['acc_mean'] - res_eye['acc_mean']) * 100
    print(f"  Mouth - Eye = {diff:+.1f}%p  ← {'Mouth 우위 (Jack 2012 반박)' if diff > 0 else 'Eye 우위 (Jack 2012 지지)'}")

    # === 3.4 FACS canonical (Ekman) ===
    print("\n=== 3.4 FACS canonical subset (Ekman) ===")
    for name, aus in EKMAN_MAP.items():
        cols = [a for a in aus if a in df_s.columns]
        X_sub = df_s[cols].values
        res = linear_probe(X_sub, y, n_folds=3)
        results[f"ekman_{name}"] = {**res, "au_list": cols}
        print(f"  {name:40s}  features={cols}  acc={res['acc_mean']*100:.1f}%")

    # Save results
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # === Plots ===
    # Per-AU ranking bar
    fig, ax = plt.subplots(figsize=(10, 11))
    y_pos = np.arange(len(per_au_df))
    ax.barh(y_pos, per_au_df["acc"], xerr=per_au_df["acc_std"],
            color="steelblue", alpha=0.85, capsize=2)
    ax.axvline(25, color="red", linestyle="--", alpha=0.7, label="Random (25%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(per_au_df["AU"], fontsize=8)
    ax.set_xlabel("Linear probe accuracy (%, single AU feature)")
    ax.set_title("Phase 0.3 — Per-AU single-feature classification accuracy\n(41 AU × scalar, 30K stratified, 3-fold CV)")
    ax.set_xlim(20, 55)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    for i, v in enumerate(per_au_df["acc"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "per_au_ranking.png", dpi=130)
    plt.close()

    # Eye vs Mouth comparison bar
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        ("All 41 AU", results["all_au"]["acc_mean"] * 100, results["all_au"]["acc_std"] * 100),
        (f"Mouth-AU ({len(mouth_cols)} features)", res_mouth["acc_mean"] * 100, res_mouth["acc_std"] * 100),
        (f"Eye-AU ({len(eye_cols)} features)", res_eye["acc_mean"] * 100, res_eye["acc_std"] * 100),
    ]
    for name, aus in EKMAN_MAP.items():
        r = results[f"ekman_{name}"]
        groups.append((name, r["acc_mean"] * 100, r["acc_std"] * 100))
    groups.append(("Random baseline", 25.0, 0.0))

    names = [g[0] for g in groups]
    accs = [g[1] for g in groups]
    errs = [g[2] for g in groups]
    colors = ["darkgreen", "steelblue", "indianred"] + ["orange"] * 3 + ["gray"]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, accs, xerr=errs, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Linear probe accuracy (%)")
    ax.set_title("Phase 0.3 — AU group comparison (Jack 2012 재검증 + Ekman)")
    ax.invert_yaxis()
    ax.axvline(25, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(accs):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "au_group_comparison.png", dpi=130)
    plt.close()

    # === Summary markdown ===
    lines = ["# Phase 0.3 — Per-AU Linear Probe Results", "",
             "**측정**: 각 AU intensity (OpenGraphAU) 위에 로지스틱 회귀로 감정 4-class 맞추는 정확도.",
             "Random = 25%, Phase 0.1c region POOLED = 80.2-81.5%.", ""]

    lines.append("## 3.3 Jack 2012 재검증 (Eye-AU vs Mouth-AU)")
    lines.append("| Group | # AU | Accuracy | Macro F1 |")
    lines.append("|-------|------|----------|----------|")
    lines.append(f"| All 41 AU | 41 | **{results['all_au']['acc_mean']*100:.1f}%** ± {results['all_au']['acc_std']*100:.1f} | {results['all_au']['f1_mean']:.3f} |")
    lines.append(f"| Mouth-AU group | {len(mouth_cols)} | **{res_mouth['acc_mean']*100:.1f}%** ± {res_mouth['acc_std']*100:.1f} | {res_mouth['f1_mean']:.3f} |")
    lines.append(f"| Eye-AU group | {len(eye_cols)} | **{res_eye['acc_mean']*100:.1f}%** ± {res_eye['acc_std']*100:.1f} | {res_eye['f1_mean']:.3f} |")
    diff_sign = "우위 (Jack 2012 **반박**)" if diff > 0 else "우위 (Jack 2012 지지)"
    lines.append(f"\n**Mouth - Eye = {diff:+.1f}%p** → Mouth group {diff_sign}")

    lines.append("\n## 3.4 Ekman canonical mapping 재현도")
    lines.append("| Emotion + AU | Accuracy | 해석 |")
    lines.append("|-----|----------|------|")
    for name in EKMAN_MAP:
        r = results[f"ekman_{name}"]
        acc = r['acc_mean']*100
        note = "강하게 재현" if acc > 40 else ("부분 재현" if acc > 32 else "약함")
        lines.append(f"| {name} | **{acc:.1f}%** | {note} |")

    lines.append("\n## 3.1 Top 10 per-AU ranking (single feature)")
    lines.append("| AU | Accuracy | 해석 |")
    lines.append("|-----|----------|------|")
    for _, row in per_au_df.head(10).iterrows():
        au_desc = {
            "AU1":"Inner brow raiser","AU2":"Outer brow raiser","AU4":"Brow lowerer",
            "AU5":"Upper lid raiser","AU6":"Cheek raiser","AU7":"Lid tightener",
            "AU9":"Nose wrinkler","AU10":"Upper lip raiser","AU12":"Lip corner puller",
            "AU15":"Lip corner depressor","AU17":"Chin raiser","AU23":"Lip tightener",
            "AU25":"Lips part","AU26":"Jaw drop","AU27":"Mouth stretch",
        }.get(row["AU"], "(sub-AU variant)")
        lines.append(f"| {row['AU']} ({au_desc}) | {row['acc']:.1f}% | |")

    with open(OUT / "summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n=== Done. Output: {OUT} ===")


if __name__ == "__main__":
    main()
