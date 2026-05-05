"""Re-render plots with Korean font after main run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Set Korean font
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # ttc 안의 KR도 포함
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/home/ajy/AU-RegionFormer/outputs/phase0/01_au_embedding_diag")

with open(OUT / "metrics.json") as f:
    data = json.load(f)

BACKBONES = ["mobilevitv2_150", "convnext_base.fb_in22k_ft_in1k"]

# Re-render centroid heatmaps with Korean labels
for key_prefix in ["per_region", "pooled"]:
    src = data[key_prefix]
    for k, v in src.items():
        if key_prefix == "per_region":
            region, bb = k.split("|")
            name = f"cossim_{region}_{bb}.png"
            title = f"Cosine Sim — {region} / {bb}"
        else:
            bb = k
            name = f"cossim_pooled_{bb}.png"
            title = f"Cosine Sim — POOLED / {bb}"
        classes = v["classes_ko"]
        mat = np.array(v["centroid"]["cosine_sim_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(mat, annot=True, fmt=".3f",
                    xticklabels=classes, yticklabels=classes,
                    cmap="coolwarm", ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(OUT / name, dpi=120)
        plt.close()
print("Centroid heatmaps re-rendered.")
