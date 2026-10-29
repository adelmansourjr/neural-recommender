"""Plot the completed, frozen validation-only diagnostic; requires matplotlib."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
protocol = project / "reports/convergence_protocol_2026_10_23.json"
report = json.loads((project / "reports/convergence_diagnostic_2026_10_23.json").read_text())
if report["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
    raise ValueError("Convergence figure requires the exact frozen diagnostic protocol")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for seed, color in (("7", "#3d4fc5"), ("19", "#087e8b"), ("42", "#a15a13")):
    rows = report["results"][seed]["learning_curve"]
    epochs = [row["epoch"] for row in rows]
    axes[0].plot(epochs, [row["loss"] for row in rows], color=color, label=f"Seed {seed}")
    axes[1].plot(
        epochs, [row["validation_ndcg_at_10"] for row in rows], color=color, label=f"Seed {seed}"
    )
for axis, title, ylabel in (
    (axes[0], "Sampled training objective", "Cross-entropy loss"),
    (axes[1], "Existing development users", "Validation NDCG@10 · five likes"),
):
    axis.axvline(5, color="#777777", linestyle="--", linewidth=1, label="Original epoch budget")
    axis.set(xlabel="Epoch", ylabel=ylabel, title=title, xlim=(1, 20), xticks=(1, 5, 10, 15, 20))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
axes[1].legend(loc="lower right", fontsize=8)
figure.suptitle("Three-seed convergence diagnostic · no reserve-user scoring", fontsize=14)
figure.supxlabel("Training and reused validation only; fitted models were discarded.", fontsize=9)
for extension in ("png", "svg"):
    figure.savefig(project / f"reports/figures/neural-convergence.{extension}", dpi=160)
plt.close(figure)
