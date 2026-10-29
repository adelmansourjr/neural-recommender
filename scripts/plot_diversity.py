"""Plot the frozen descriptive genre-diversity report; requires matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project = Path(__file__).resolve().parents[1]
report = json.loads((project / "reports/diversity_tradeoffs.json").read_text())
weights = report["protocol"]["relevance_weights"]
rows = [report["results"][str(weight)] for weight in weights]
positions = np.arange(len(weights))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
for axis, metric, title, color in (
    (axes[0], "ndcg_at_10", "Archived-rating relevance", "#3d4fc5"),
    (axes[1], "genre_intra_list_diversity", "Within-list genre variety", "#087e8b"),
):
    means = np.array([row[metric]["mean"] for row in rows])
    bounds = np.array([row[metric]["ci95"] for row in rows]).T
    axis.errorbar(
        positions,
        means,
        yerr=np.vstack((means - bounds[0], bounds[1] - means)),
        fmt="o",
        capsize=4,
        color=color,
    )
    axis.set_title(title)
    axis.set_ylabel("NDCG@10" if metric == "ndcg_at_10" else "Mean genre Jaccard distance")
axes[2].bar(positions, [100 * row["catalog_coverage"] for row in rows], color="#7a64a7")
axes[2].set_title("Overall catalog exposure")
axes[2].set_ylabel("Catalog coverage (%)")
for axis in axes:
    axis.set_xticks(positions, [f"{weight:g}" for weight in weights])
    axis.set_xlabel("User-selected relevance weight λ")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
figure.suptitle("Genre diversity is a tradeoff · 199 previously inspected users", fontsize=14)
figure.supxlabel("Fixed grid; no winner selected. Error bars: descriptive 95% user bootstrap.")
output = project / "reports/figures"
output.mkdir(exist_ok=True)
for extension in ("png", "svg"):
    figure.savefig(output / f"recommendation-diversity.{extension}", dpi=160)
plt.close(figure)
