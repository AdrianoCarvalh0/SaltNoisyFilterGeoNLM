"""Generate per-image V4 PSNR and SSIM curves as vector PDFs.

For every dataset, salt-and-pepper density, and principal tolerance, this
script writes an ``unordered`` curve in the protocol image order and an
``ordered`` curve.  In the ordered curves, each method is independently sorted
by the plotted metric, exactly as in the legacy graph notebooks; the x-axis is
therefore rank, not a shared image identity.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / 'data/output/unifiedComparisonFinalV4'
OUTPUT = ARCHIVE / 'reports/per_image_curves'
METHODS = (
    ('nlm', 'NLM', '#15b9d1', 's'),
    ('gnlm', 'GNLM', "#76d627", 'o'),
    ('ghnlm', 'GHNLM', '#ff7f0e', 'D'),
    ('ianlm', 'IANLM', "#f70707", '*'),
    ('median', 'Median', '#9467bd', 'v'),
    ('aswmf', 'ASWMF', '#2ca02c', '^'),
    ('nlmedians', 'NLMedians', '#111111', 'P'),
)
LEVEL_LABELS = {
    'low': 'low (p=1%)', 'moderate': 'moderate (p=3%)',
    'medium': 'medium (p=5%)', 'high': 'high (p=10%)',
    'extreme': 'extreme (p=20%)',
}


plt.rcParams.update({
    'font.size': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 30,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
})

def set_dynamic_ssim_axis(axis: plt.Axes, values: np.ndarray,
                          step: float = 0.03, margin_ratio: float = 0.20) -> None:
    """Set a readable SSIM range and tick interval from the current group."""
    minimum, maximum = float(values.min()), float(values.max())
    data_range = maximum - minimum
    margin = margin_ratio * data_range if data_range > 0 else step
    ymin = max(0.0, minimum - margin)
    ymax = min(1.05, maximum + margin)
    axis.set_ylim(ymin, ymax)
    ticks = np.arange(
        np.floor(ymin / step) * step,
        np.ceil(ymax / step) * step + 1e-9,
        step,
    )
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


def set_dynamic_psnr_axis(axis: plt.Axes, values: np.ndarray) -> None:
    """Leave comparable breathing room while retaining useful PSNR detail."""
    minimum, maximum = float(values.min()), float(values.max())
    axis.set_ylim(minimum - 5.0, maximum + 5.0)
    axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, min_n_ticks=7))


def set_dynamic_x_ticks(axis: plt.Axes, image_count: int, width: float) -> None:
    """Use all labels for Set12 and as many legible labels as possible for Set50."""
    if image_count <= 12:
        ticks = np.arange(1, image_count + 1)
    else:
        max_ticks = max(12, int(width / 1.7))
        ticks = np.unique(np.rint(
            np.linspace(1, image_count, num=max_ticks)
        ).astype(int))
    axis.set_xticks(ticks)


def plot(group: pd.DataFrame, image_names: list[str], dataset: str, level: str,
         tolerance: int, metric: str, ordered: bool, destination: Path) -> None:
    by_method = group.pivot(index='file_name', columns='method', values=metric)
    by_method = by_method.loc[image_names, [name for name, *_ in METHODS]]
    x = np.arange(1, len(image_names) + 1)
    # Larger collections need more horizontal room; no title is drawn, leaving
    # more vertical area for the curves and the legend.
    figure_width = max(20.0, min(28.0, 0.48 * len(image_names) + 3.0))
    figure, axis = plt.subplots(figsize=(figure_width, 12))
    all_values = by_method.to_numpy(dtype=float)
    for method, label, color, marker in METHODS:
        values = by_method[method].to_numpy(dtype=float)
        if ordered:
            values = np.sort(values)
        axis.plot(x, values, label=label, color=color, marker=marker,
                  linewidth=2.5, markersize=7)
    axis.set_xlabel('Independent rank' if ordered else 'Image index')
    axis.set_ylabel('PSNR (dB)' if metric == 'psnr' else 'SSIM')
    axis.set_xlim(1, len(image_names))
    set_dynamic_x_ticks(axis, len(image_names), figure_width)
    if metric == 'ssim':
        set_dynamic_ssim_axis(axis, all_values)
    else:
        set_dynamic_psnr_axis(axis, all_values)
    axis.grid(False)
    axis.legend(loc='best', frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(destination, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    protocol = json.loads((ARCHIVE / 'protocol.json').read_text())
    data = pd.read_csv(ARCHIVE / 'results_partial.csv', dtype={'file_name': str})
    expected = (sum(len(names) for names in protocol['dataset_images'].values())
                * len(protocol['densities']) * len(protocol['tolerances'])
                * len(protocol['methods']))
    if len(data) != expected:
        raise ValueError(f'Expected {expected} V4 records, found {len(data)}')
    written = []
    for dataset, image_names in protocol['dataset_images'].items():
        image_names = [str(name) for name in image_names]
        directory = OUTPUT / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for tolerance in protocol['tolerances']:
            for level in protocol['densities']:
                group = data[(data['dataset'] == dataset)
                             & (data['tolerance'] == tolerance)
                             & (data['level'] == level)]
                if len(group) != len(image_names) * len(METHODS):
                    raise ValueError(f'Incomplete group: {dataset}, tau={tolerance}, {level}')
                for metric in ('psnr', 'ssim'):
                    for ordered, order_name in ((False, 'unordered'), (True, 'ordered')):
                        path = directory / f'graph_{metric}_{level}_tau{tolerance}_{order_name}.pdf'
                        plot(group, image_names, dataset, level, tolerance, metric, ordered, path)
                        written.append(path.relative_to(ARCHIVE).as_posix())
    manifest = {
        'source_archive': str(ARCHIVE.relative_to(ROOT)),
        'protocol_version': protocol['version'],
        'git_revision': protocol['git_revision'],
        'source_tree_sha256': protocol['source_tree_sha256'],
        'metrics': ['psnr', 'ssim'],
        'orders': {
            'unordered': 'protocol image order',
            'ordered': 'independently sorted ascending per method and metric',
        },
        'files': written,
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(written)} PDFs to {OUTPUT}')


if __name__ == '__main__':
    main()
