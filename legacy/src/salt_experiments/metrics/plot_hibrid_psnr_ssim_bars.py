from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVEL_ORDER = ["low", "moderate", "medium", "high", "extreme"]
LEVEL_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "moderate": "Moderate",
    "high": "High",
    "extreme": "Extreme",
}

METHODS = [
    ("NLM", "nlm", "#4C78A8"),
    ("GNLM", "gnlm_original", "#F58518"),
    ("GHNLM", "geonlm_hibrid", "#7F7F7F"),
    ("IANLM", "anlm", "#D62728"),
    ("Median", "median", "#B279A2"),
    ("ASWMF", "aswmf", "#72B7B2"),
    ("NLMedian", "nlmedians", "#9D755D"),
]


def ordered_summary(path):
    df = pd.read_excel(path)
    summary = df.groupby("level", as_index=False).mean(numeric_only=True)
    summary["level"] = pd.Categorical(summary["level"], categories=LEVEL_ORDER, ordered=True)
    return summary.sort_values("level").reset_index(drop=True)


def metric_table(summary, metric):
    data = []
    for label, key, color in METHODS:
        column = f"{metric}_{key}"
        if column in summary.columns:
            data.append((label, color, summary[column].to_numpy(dtype=float)))
    return data


def plot_metric(summary, dataset_label, metric, output_dir):
    methods = metric_table(summary, metric)
    labels = [LEVEL_LABELS[level] for level in summary["level"].astype(str)]
    x = np.arange(len(labels))
    width = 0.11

    fig, ax = plt.subplots(figsize=(13, 6.8))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

    for offset, (method_label, color, values) in zip(offsets, methods):
        ax.bar(
            x + offset,
            values,
            width,
            label=method_label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=15,
    )

    if metric == "ssim":
        ax.set_ylim(0, min(1.08, max(1.02, np.nanmax([values for _, _, values in methods]) + 0.06)))
    else:
        ax.set_ylim(0, np.nanmax([values for _, _, values in methods]) * 1.18)

    fig.tight_layout()
    png_path = output_dir / f"{dataset_label.lower()}_hibrid_{metric}_bars.png"
    pdf_path = output_dir / f"{dataset_label.lower()}_hibrid_{metric}_bars.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_dataset(dataset_name, dataset_label):
    summary_dir = Path(f"/workspace/data/output/{dataset_name}Hibrid/summary")
    summary_path = summary_dir / f"results_{dataset_name}_hibrid_all.xlsx"
    output_dir = summary_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = ordered_summary(summary_path)
    outputs = []
    for metric in ["psnr", "ssim"]:
        outputs.extend(plot_metric(summary, dataset_label, metric, output_dir))
    return outputs


if __name__ == "__main__":
    for dataset_name, dataset_label in [("set12", "Set12"), ("set50", "Set50")]:
        for output in plot_dataset(dataset_name, dataset_label):
            print(output)
