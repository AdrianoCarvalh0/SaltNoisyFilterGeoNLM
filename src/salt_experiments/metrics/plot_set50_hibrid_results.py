from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


SUMMARY_DIR = Path("/workspace/data/output/set50Hibrid/summary")
INPUT_ALL = SUMMARY_DIR / "results_set50_hibrid_all.xlsx"
INPUT_SUMMARY = SUMMARY_DIR / "results_set50_hibrid_summary.xlsx"
OUTPUT_DIR = SUMMARY_DIR / "graphs"

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
        "label": "Median",
        "key": "median",
        "color": "#9467bd",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "label": "NLMedians",
        "key": "nlmedians",
        "color": "#ff7f0e",
        "marker": "v",
        "linestyle": "-",
    },
    {
        "label": "GeoHibNLM",
        "key": "geonlm_hibrid",
        "color": "#d62728",
        "marker": "*",
        "linestyle": "-",
    },
]

SUMMARY_METHODS = [
    ("NLM", "nlm", "#0099c8"),
    ("Median", "median", "#9467bd"),
    ("NLMedians", "nlmedians", "#ff7f0e"),
    ("GeoHibNLM", "geonlm_hibrid", "#d62728"),
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
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    else:
        margin = max(2.0, data_range * 0.12)
        ax.set_ylim(min_value - margin, max_value + margin)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def plot_level_metric(df, level, metric, sorted_values=False):
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
    ax.grid(False)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()

    suffix = "sorted" if sorted_values else "no_sorted"
    output = OUTPUT_DIR / f"graph_{metric}_{level}_{suffix}.pdf"
    fig.savefig(output, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output


def plot_summary_bars(summary_df, metric):
    summary_df = summary_df.set_index("level").loc[LEVEL_ORDER].reset_index()
    x = np.arange(len(summary_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 10))
    for offset, (label, key, color) in enumerate(SUMMARY_METHODS):
        column = f"mean_{metric}_{key}"
        if column not in summary_df.columns:
            continue
        ax.bar(
            x + (offset - 2) * width,
            summary_df[column],
            width,
            label=label,
            color=color,
        )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel(f"Mean {metric.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[level] for level in summary_df["level"]], rotation=15)
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(False)
    fig.tight_layout()

    output = OUTPUT_DIR / f"summary_mean_{metric}.pdf"
    fig.savefig(output, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output


def plot_hybrid_wins(summary_df):
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

    output = OUTPUT_DIR / "summary_hybrid_wins.pdf"
    fig.savefig(output, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_ALL)
    summary_df = pd.read_excel(INPUT_SUMMARY)
    levels = [level for level in LEVEL_ORDER if level in set(df["level"])]

    outputs = []
    for level in levels:
        for metric in ["psnr", "ssim"]:
            outputs.append(plot_level_metric(df, level, metric, sorted_values=False))
            outputs.append(plot_level_metric(df, level, metric, sorted_values=True))

    for metric in ["psnr", "ssim", "score"]:
        outputs.append(plot_summary_bars(summary_df, metric))
    outputs.append(plot_hybrid_wins(summary_df))

    print(f"Generated {len(outputs)} graphs in {OUTPUT_DIR}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
