"""Generate V4 grouped-bar summaries for PSNR and SSIM.

Each chart aggregates the per-image results by density for one dataset and
detector tolerance.  The presentation deliberately has no title or y-axis
label, matching the existing hybrid bar-chart convention.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / 'data/output/unifiedComparisonFinalV4'
OUTPUT = ARCHIVE / 'reports/summary_bars'
METHODS = (
    ('nlm', 'NLM', '#15b9d1', 's'),
    ('gnlm', 'GNLM', '#76d627', 'o'),
    ('ghnlm', 'GHNLM', '#420eff', 'D'),
    ('ianlm', 'IANLM', '#f70707', '*'),
    ('median', 'Median', '#9467bd', 'v'),
    ('aswmf', 'ASWMF', '#999C99', '^'),
    ('nlmedians', 'NLMedians', '#111111', 'P'),
)
LEVEL_LABELS = {
    'low': 'Low',
    'moderate': 'Moderate',
    'medium': 'Medium',
    'high': 'High',
    'extreme': 'Extreme',
}


def plot(summary: pd.DataFrame, dataset: str, tolerance: int,
         metric: str, destination: Path) -> None:
    """Write one grouped bar chart, with its legend below the axes."""
    levels = list(LEVEL_LABELS)
    values = summary.pivot(index='level', columns='method', values=metric)
    values = values.loc[levels, [method for method, *_ in METHODS]]
    x = np.arange(len(levels))
    width = 0.11
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    figure, axis = plt.subplots(figsize=(13, 6.8))
    for offset, (method, label, color, _) in zip(offsets, METHODS):
        axis.bar(
            x + offset, values[method].to_numpy(dtype=float), width,
            label=label, color=color, edgecolor='white', linewidth=0.5,
        )

    axis.set_xticks(x, [LEVEL_LABELS[level] for level in levels], fontsize=15)
    axis.tick_params(axis='y', labelsize=15)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)
    axis.legend(
        ncol=len(METHODS), loc='upper center', bbox_to_anchor=(0.5, -0.08),
        frameon=False, fontsize=15,
    )

    maximum = float(values.to_numpy(dtype=float).max())
    axis.set_ylim(0.0, 1.05 if metric == 'ssim' else maximum * 1.18)
    figure.tight_layout()
    figure.savefig(destination, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    protocol = json.loads((ARCHIVE / 'protocol.json').read_text())
    summary = pd.read_csv(ARCHIVE / 'summary_partial.csv')
    expected = (len(protocol['dataset_images']) * len(protocol['tolerances'])
                * len(protocol['densities']) * len(METHODS))
    if len(summary) != expected:
        raise ValueError(f'Expected {expected} summary records, found {len(summary)}')

    written = []
    for dataset in protocol['dataset_images']:
        for tolerance in protocol['tolerances']:
            group = summary[(summary['dataset'] == dataset)
                            & (summary['tolerance'] == tolerance)]
            for metric in ('psnr', 'ssim'):
                destination = OUTPUT / dataset / f'graph_{metric}_tau{tolerance}.pdf'
                destination.parent.mkdir(parents=True, exist_ok=True)
                plot(group, dataset, tolerance, metric, destination)
                written.append(destination.relative_to(ARCHIVE).as_posix())

    (OUTPUT / 'manifest.json').write_text(json.dumps({
        'source_archive': str(ARCHIVE.relative_to(ROOT)),
        'protocol_version': protocol['version'],
        'metrics': ['psnr', 'ssim'],
        'tolerances': protocol['tolerances'],
        'files': written,
    }, indent=2) + '\n')
    print(f'Wrote {len(written)} PDFs to {OUTPUT}')


if __name__ == '__main__':
    main()
