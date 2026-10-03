from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


DATASETS = {
    "set12": {
        "label": "Set12",
        "summary_dir": Path("/workspace/data/output/set12Hibrid/summary"),
    },
    "set50": {
        "label": "Set50",
        "summary_dir": Path("/workspace/data/output/set50Hibrid/summary"),
    },
}

LEVEL_ORDER = ["low", "moderate", "medium", "high", "extreme"]
LEVEL_LABELS = {
    "low": "Low (1%)",
    "moderate": "Moderate (3%)",
    "medium": "Medium (5%)",
    "high": "High (10%)",
    "extreme": "Extreme (20%)",
}

METHODS = [
    {
        "label": "NLM",
        "key": "nlm",
        "color": "#0099c8",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "label": "GNLM",
        "key": "gnlm_original",
        "color": "#2ca02c",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "label": "GHNLM",
        "key": "geonlm_hibrid",
        "color": "#7f7f7f",
        "marker": "*",
        "linestyle": "-",
    },
    {
        "label": "IANLM",
        "key": "anlm",
        "color": "#d62728",
        "marker": "P",
        "linestyle": "-",
    },
    {
        "label": "Median",
        "key": "median",
        "color": "#9467bd",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "label": "ASWMF",
        "key": "aswmf",
        "color": "#8c564b",
        "marker": "^",
        "linestyle": "-",
    },
    {
        "label": "NLMedians",
        "key": "nlmedians",
        "color": "#ff7f0e",
        "marker": "v",
        "linestyle": "-",
    },
]

SUMMARY_METHODS = [
    ("NLM", "nlm", "#0099c8"),
    ("GNLM", "gnlm_original", "#2ca02c"),
    ("GHNLM", "geonlm_hibrid", "#7f7f7f"),
    ("IANLM", "anlm", "#d62728"),
    ("Median", "median", "#9467bd"),
    ("ASWMF", "aswmf", "#8c564b"),
    ("NLMedians", "nlmedians", "#ff7f0e"),
]


plt.rcParams.update(
    {
        "font.size": 24,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 24,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.dpi": 150,
    }
)


def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def metric_column(metric, method_key):
    return f"{metric}_{method_key}"


def available_methods(df, metric):
    methods = []
    for method in METHODS:
        column = metric_column(metric, method["key"])
        if column in df.columns:
            methods.append(method)
    return methods


def set_metric_axis(ax, values, metric):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return

    min_value = float(finite.min())
    max_value = float(finite.max())
    data_range = max_value - min_value

    if metric == "ssim":
        margin = max(0.02, data_range * 0.12)
        ymin = max(0.0, min_value - margin)
        ymax = min(1.02, max_value + margin)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    else:
        margin = max(2.0, data_range * 0.12)
        ax.set_ylim(min_value - margin, max_value + margin)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def plot_level_metric(df, level, metric, output_dir, sorted_values=False):
    level_df = df[df["level"] == level].copy()
    level_df = level_df.sort_values("file_name", key=lambda col: col.map(natural_sort_key))
    methods = available_methods(level_df, metric)
    x = np.arange(1, len(level_df) + 1)

    fig, ax = plt.subplots(figsize=(20, 11))
    all_values = []
    for method in methods:
        column = metric_column(metric, method["key"])
        values = level_df[column].to_numpy(dtype=float)
        if sorted_values:
            values = np.sort(values)
        all_values.extend(values)
        ax.plot(
            x,
            values,
            label=method["label"],
            color=method["color"],
            marker=method["marker"],
            linestyle=method["linestyle"],
        )

    ax.set_xlabel("Image Index")
    ax.set_ylabel(metric.upper())
    set_metric_axis(ax, all_values, metric)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.legend(loc="best", frameon=False, ncol=3)
    fig.tight_layout()

    suffix = "sorted" if sorted_values else "no_sorted"
    output = output_dir / f"graph_{metric}_{level}_{suffix}.pdf"
    fig.savefig(output, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output


def plot_summary_bars(summary_df, metric, output_dir):
    summary_df = summary_df.set_index("level").loc[LEVEL_ORDER].reset_index()
    x = np.arange(len(summary_df))
    width = 0.11

    fig, ax = plt.subplots(figsize=(18, 10))
    for offset, (label, key, color) in enumerate(SUMMARY_METHODS):
        column = f"mean_{metric}_{key}"
        if column not in summary_df.columns:
            continue
        ax.bar(
            x + (offset - (len(SUMMARY_METHODS) - 1) / 2) * width,
            summary_df[column],
            width,
            label=label,
            color=color,
        )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel(f"Mean {metric.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[level] for level in summary_df["level"]], rotation=15)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    ax.legend(loc="best", frameon=False, ncol=4)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    fig.tight_layout()

    output = output_dir / f"summary_mean_{metric}.pdf"
    fig.savefig(output, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output


def plot_hybrid_wins(summary_df, output_dir):
    summary_df = summary_df.set_index("level").loc[LEVEL_ORDER].reset_index()
    x = np.arange(len(summary_df))
    width = 0.24

    win_columns = [
        ("vs NLM", "wins_hibrid_vs_nlm", "#0099c8"),
        ("vs Median", "wins_hibrid_vs_median", "#9467bd"),
        ("vs NLMedians", "wins_hibrid_vs_nlmedians", "#ff7f0e"),
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    for offset, (label, column, color) in enumerate(win_columns):
        ax.bar(x + (offset - 1) * width, summary_df[column], width, label=label, color=color)

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Wins out of 50 Images")
    ax.set_ylim(0, 52)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[level] for level in summary_df["level"]], rotation=15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.legend(loc="best", frameon=False, ncol=3)
    ax.grid(False)
    fig.tight_layout()

    output = output_dir / "summary_hybrid_wins.pdf"
    fig.savefig(output, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output


def plot_dataset(dataset_name, config):
    summary_dir = config["summary_dir"]
    output_dir = summary_dir / "graphs"
    input_all = summary_dir / f"results_{dataset_name}_hibrid_all.xlsx"
    input_summary = summary_dir / f"results_{dataset_name}_hibrid_summary.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_all)
    summary_df = pd.read_excel(input_summary)
    levels = [level for level in LEVEL_ORDER if level in set(df["level"])]

    outputs = []
    for level in levels:
        for metric in ["psnr", "ssim"]:
            outputs.append(plot_level_metric(df, level, metric, output_dir, sorted_values=False))
            outputs.append(plot_level_metric(df, level, metric, output_dir, sorted_values=True))

    for metric in ["psnr", "ssim", "score"]:
        outputs.append(plot_summary_bars(summary_df, metric, output_dir))
    if "wins_hibrid_vs_median" in summary_df.columns:
        outputs.append(plot_hybrid_wins(summary_df, output_dir))

    print(f"Generated {len(outputs)} {config['label']} graphs in {output_dir}")
    for output in outputs:
        print(output)


def main():
    for dataset_name, config in DATASETS.items():
        plot_dataset(dataset_name, config)


if __name__ == "__main__":
    main()
