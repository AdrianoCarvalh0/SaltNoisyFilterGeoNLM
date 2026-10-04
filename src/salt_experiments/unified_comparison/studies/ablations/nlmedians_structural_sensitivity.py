"""Set12-only structural ablation for the NLMedians baseline.

The grid f in {1,2,3} and t in {2,3,4} is evaluated under medium impulse
noise at both principal tolerances.  It is a configuration-selection study:
Set50 is intentionally excluded and must not be used to select the default.
NLMedians has no graph-connectivity parameter k.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from itertools import product

import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src' / 'salt_experiments' / 'unified_comparison'))

from lib.noisy_functions import add_near_extreme_impulse_noise  # noqa: E402
from lib.nlmedians import run_nlmedians  # noqa: E402
from run import LEVELS, PROTOCOL, calibrate, load_reference  # noqa: E402


IMAGES = tuple(f'{index:02d}' for index in range(1, 12))
TOLERANCES = (0, 4)
VARIANTS = tuple(
    (f'f{f}_t{t}', f'Patch radius {f}; search radius {t}', f, t)
    for f, t in product((1, 2, 3), (2, 3, 4))
)


def run_study(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps({
        'study': 'nlmedians_structural_sensitivity',
        'selection_scope': 'Set12 only; Set50 excluded from configuration selection',
        'dataset': 'set12', 'images': list(IMAGES), 'level': 'medium',
        'density': LEVELS['medium'], 'tolerances': list(TOLERANCES),
        'noise_seed': 42, 'h': '0.005 * image-wise calibrated compact-NLM h',
        'variants': [dict(name=name, description=description, f=f, t=t, k=None)
                     for name, description, f, t in VARIANTS],
        'canonical_protocol': PROTOCOL,
    }, indent=2))
    rows = []
    for tolerance in TOLERANCES:
        for image_name in IMAGES:
            reference = load_reference('set12', image_name)
            noisy, mask = add_near_extreme_impulse_noise(
                reference, salt_prob=LEVELS['medium'] / 2,
                pepper_prob=LEVELS['medium'] / 2,
                impulse_tolerance=tolerance, seed=42, return_mask=True,
            )
            case_output = output / f'tolerance_{tolerance}' / image_name
            case_output.mkdir(parents=True, exist_ok=True)
            _, nlm_info = calibrate(reference, noisy, case_output, 'medium')
            for name, description, f, t in VARIANTS:
                start = time.perf_counter()
                h = nlm_info['h'] * .005
                result = run_nlmedians(reference, noisy, h=h, f=f, t=t)
                row = {
                    'tolerance': tolerance, 'file_name': image_name,
                    'density': LEVELS['medium'], 'variant': name,
                    'variant_description': description, 'f': f, 't': t,
                    'k': None, 'h_nlm': nlm_info['h'], 'h_nlmedians': h,
                    'psnr': result['psnr'], 'ssim': result['ssim'],
                    'score': result['score'], 'elapsed_seconds': time.perf_counter() - start,
                    'assigned_impulses': int(mask.sum()),
                }
                rows.append(row)
                print(
                    f"DONE tau={tolerance} image={image_name} {name} "
                    f"score={row['score']:.6f} time={row['elapsed_seconds']:.2f}s",
                    flush=True,
                )
    results = pd.DataFrame(rows)
    results.to_csv(output / 'results.csv', index=False)
    summary = (
        results.groupby(['tolerance', 'variant', 'variant_description', 'f', 't'], as_index=False)
        .agg(images=('file_name', 'nunique'), mean_psnr=('psnr', 'mean'),
             mean_ssim=('ssim', 'mean'), mean_score=('score', 'mean'),
             mean_elapsed_seconds=('elapsed_seconds', 'mean'))
        .sort_values(['tolerance', 'mean_score'], ascending=[True, False])
    )
    summary.to_csv(output / 'summary.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'data' / 'output' / 'studies' / 'ablations' / 'nlmediansStructuralSensitivitySet12MediumV1')
    args = parser.parse_args()
    print(run_study(args.output).to_string(index=False))


if __name__ == '__main__':
    main()
