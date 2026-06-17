from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVEL_ORDER = ["low", "medium", "moderate", "high", "extreme"]
LEVEL_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "moderate": "Moderate",
    "high": "High",
    "extreme": "Extreme",
}

METHODS = [
    ("NLM", "nlm", "#4C78A8"),
    ("GNLM", "gnlm", "#F58518"),
    ("GHNLM", "geonlm_hibrid", "#E45756"),
    ("ANLM", "aswmf_nlm_h001", "#54A24B"),
    ("Median", "median", "#B279A2"),
    ("ASWMF", "aswmf", "#72B7B2"),
    ("NLMedian", "nlmedians", "#9D755D"),
]


def ordered_summary(path):
    df = pd.read_excel(path)
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    return df.sort_values("level").reset_index(drop=True)


def metric_table(summary, metric):
    data = []
    for label, key, color in METHODS:
        column = f"mean_{metric}_{key}"
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
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method_label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}" if metric == "psnr" else f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="#222222",
                )

    ylabel = "PSNR (dB)" if metric == "psnr" else "SSIM"
    title_metric = "PSNR" if metric == "psnr" else "SSIM"
    ax.set_title(f"{dataset_label} - {title_metric} by Noise Level", fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Salt-and-pepper noise level", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

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
    summary_path = summary_dir / f"results_{dataset_name}_hibrid_summary.xlsx"
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
