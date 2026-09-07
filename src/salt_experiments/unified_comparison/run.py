"""Matched Set12/Set50 protocol; resumable per image and method.

Run with the project's salt environment. Existing experimental outputs are read
only for clean references. All calibration and results go to a new output tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

# Avoid multiplying BLAS threads inside the graph workers.
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '8'
os.environ['NUMBA_NUM_THREADS'] = '8'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/salt_experiments'))
sys.path.insert(0, str(ROOT / 'src/salt_experiments/set12'))

import cupy as cp
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma
from functions.noisy_functions import add_near_extreme_impulse_noise
from functions.nlm_functions import NLM_fast_cuda_global, compute_adaptive_q, mirror_cpu
from functions.anlm_functions import Parallel_Switch_ANLM
from functions.geonlm_functions import run_geonlm_pipeline
from functions.impulse_tolerance_filters import (
    run_ghnlm_impulse_tolerance_pipeline, aswmf_impulse_tolerance_filter,
)
from NLMedians import run_nlmedians

LEVELS = {'low': .01, 'moderate': .03, 'medium': .05, 'high': .10, 'extreme': .20}
METHODS = ('nlm', 'ianlm', 'median', 'aswmf', 'nlmedians', 'ghnlm', 'gnlm')
PROTOCOL = {
    'version': 1, 'seed': 42, 'tolerances': [0, 4],
    'nlm': {'f': 1, 't': 3, 'offsets_by_density': {
        'low': [-120, 120], 'moderate': [10, 170], 'medium': [40, 170],
        'high': [100, 280], 'extreme': [250, 600]},
            'selection': 'maximize 0.5*PSNR + 50*SSIM; uint8; smallest h on ties'},
    'ianlm': {'f': 1, 't': 3, 'h': 1.0},
    'ghnlm': {'f': 1, 't': 3, 'h': 1.0, 'k': 7},
    'gnlm': {'f': 1, 't': 3, 'k': 7,
             'h': 'h_nlm*(1.40 if h_nlm<60 or sigma<10 else 1.55)'},
    'nlmedians': {'f': 2, 't': 3, 'h_multiplier': .005},
    'aswmf': {'radius': 3, 'same_tolerance_as_ianlm': True},
    'median': {'size': 3, 'mode': 'reflect'},
    'spatial_weights': [1., 1., 10.], 'z_alpha': 1.96, 'outlier_alpha': 0.,
    'noise': 'paired_scaled_uniform_v1; sample without replacement',
    'metrics': 'clip to [0,255], cast uint8, data_range=255',
}


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def quantize(image):
    if not np.isfinite(image).all():
        raise ValueError('Non-finite filter output; result not saved.')
    return np.clip(image, 0, 255).astype(np.uint8)


def metrics(reference, output):
    ref, out = quantize(reference), quantize(output)
    psnr = float(peak_signal_noise_ratio(ref, out, data_range=255))
    ssim = float(structural_similarity(ref, out, data_range=255))
    return {'psnr': psnr, 'ssim': ssim, 'score': .5 * psnr + 50 * ssim}


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def nlm_filter(noisy, h):
    out = NLM_fast_cuda_global(cp.asarray(noisy), h=float(h), f=1, t=3)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(out)


def calibrate(reference, noisy, destination, level):
    sigma = float(estimate_sigma(noisy))
    base = float(compute_adaptive_q(sigma))
    rows, best, best_out = [], None, None
    start = time.perf_counter()
    lo, hi = PROTOCOL['nlm']['offsets_by_density'][level]
    sigma_base = int(base)
    for offset in range(lo, hi + 1):
        h = max(1, sigma_base + offset)
        out = nlm_filter(noisy, h)
        row = {'h': h, **metrics(reference, out)}
        rows.append(row)
        if best is None or row['score'] > best['score']:
            best, best_out = row, out.copy()
    pd.DataFrame(rows).to_csv(destination / 'nlm_h_sweep.csv', index=False)
    return best_out, {**best, 'sigma': sigma, 'h_source': 'current_noise_full_reference_sweep',
                      'h_at_grid_boundary': best['h'] in (base + 25, base + 499),
                      'time_calibration_s': time.perf_counter() - start}


def run_case(reference, dataset, level, tolerance, name, output, methods):
    destination = output / dataset / f'tolerance_{tolerance}' / level / Path(name).stem
    destination.mkdir(parents=True, exist_ok=True)
    noisy, mask = add_near_extreme_impulse_noise(
        reference, salt_prob=LEVELS[level]/2, pepper_prob=LEVELS[level]/2,
        impulse_tolerance=tolerance, seed=42, return_mask=True,
    )
    noisy = np.asarray(noisy, dtype=np.float32)
    case = {'dataset': dataset, 'level': level, 'file_name': name, 'tolerance': tolerance,
            'density': LEVELS[level], 'shape': list(reference.shape),
            'reference_sha256': digest(reference), 'noisy_sha256': digest(noisy)}
    path = destination / 'case.json'
    if path.exists() and json.loads(path.read_text()) != case:
        raise ValueError(f'Case identity changed: {destination}')
    atomic_json(path, case)
    np.save(destination / 'noisy.npy', noisy)
    np.save(destination / 'corruption_mask.npy', mask)
    detected = (noisy <= tolerance) | (noisy >= 255-tolerance)
    tp = int(np.count_nonzero(detected & mask))
    atomic_json(destination / 'detector.json', {
        'precision': tp / max(1, int(detected.sum())), 'recall': tp/max(1, int(mask.sum())),
        'false_positives': int(np.count_nonzero(detected & ~mask)),
    })
    common = dict(img_original=reference, img_noisy=noisy, h_base=1., mult=1.,
                  f=1, t=3, impulse_tolerance=tolerance, switch_impulse_only=True,
                  reject_impulse_candidates=True, use_aswmf_spatial_weights=True)
    for method in methods:
        result_path = destination / f'{method}.json'
        if result_path.exists():
            if not (destination / f'{method}.npy').exists():
                raise ValueError(f'Missing checkpoint image: {destination}/{method}')
            continue
        print(f'START {dataset} tau={tolerance} {level} {name} {method}', flush=True)
        info = {}
        if method == 'nlm':
            _, info = calibrate(reference, noisy, destination, level)
        if method in ('gnlm', 'nlmedians'):
            nlm = json.loads((destination/'nlm.json').read_text())
        start = time.perf_counter()
        if method == 'nlm':
            filtered = nlm_filter(noisy, info['h'])
        elif method == 'ianlm':
            filtered, stats = Parallel_Switch_ANLM(mirror_cpu(noisy, 1), f=1, t=3,
                h=1., impulse_tolerance=tolerance, return_stats=True)
            info.update(h=1., h_source='fixed_independent', **stats)
        elif method == 'ghnlm':
            filtered, h, *_ = run_ghnlm_impulse_tolerance_pipeline(**common, nn=7)
            info.update(h=h, h_source='fixed_independent')
        elif method == 'gnlm':
            gamma = 1.4 if nlm['h'] < 60 or nlm['sigma'] < 10 else 1.55
            filtered, h, *_ = run_geonlm_pipeline(reference, nlm['h'], noisy,
                                                 f=1, t=3, mult=gamma, nn=7)
            info.update(h=h, gamma=gamma, h_source='scaled_current_nlm')
        elif method == 'median':
            filtered = median_filter(noisy, size=3, mode='reflect')
        elif method == 'aswmf':
            filtered = aswmf_impulse_tolerance_filter(noisy, impulse_tolerance=tolerance,
                                                       radius=3)
        else:
            h = nlm['h']*.005
            filtered = run_nlmedians(reference=reference, noisy=noisy, h=h, f=2, t=3)['filtered']
            info.update(h=h, h_source='scaled_current_nlm')
        elapsed = time.perf_counter()-start
        if filtered.shape != reference.shape:
            raise ValueError(f'Output shape mismatch for {method}')
        row = {**case, 'method': method, **info, **metrics(reference, filtered),
               'time_filter_call_s': elapsed,
               'cpu_worker_limit': 8, 'numba_thread_limit': 8,
               'nlm_backend': 'NLM_fast_cuda_global'}
        # Result JSON is the completion marker and is committed last.
        np.save(destination / f'{method}.npy', quantize(filtered))
        atomic_json(result_path, row)
        print(f'DONE {method} score={row["score"]:.5f} seconds={elapsed:.2f}', flush=True)


def summarize(output):
    rows = []
    for method in METHODS:
        for path in output.glob(f'set*/tolerance_*/*/*/{method}.json'):
            rows.append(json.loads(path.read_text()))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output/'results_partial.csv', index=False)
        df.groupby(['dataset','tolerance','level','method']).agg(
            n=('score','size'), psnr=('psnr','mean'), ssim=('ssim','mean'),
            score=('score','mean'), time_filter_call_s=('time_filter_call_s','mean'),
        ).to_csv(output/'summary_partial.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['set12','set50'], default=['set12','set50'])
    parser.add_argument('--levels', nargs='+', choices=LEVELS, default=list(LEVELS))
    parser.add_argument('--tolerances', nargs='+', type=int, choices=[0,4], default=[0,4])
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--output', type=Path, default=ROOT/'data/output/unifiedComparisonV1')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    # Always calibrate first: downstream h never comes from a legacy pickle.
    methods = [m for m in METHODS if m in set(args.methods)|{'nlm'}]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != PROTOCOL:
        raise ValueError('Different protocol in output directory; use a new directory.')
    atomic_json(manifest, PROTOCOL)
    # Warm up kernels outside per-image timings; no reference data used here.
    dummy = np.full((24,24),128,dtype=np.float32);dummy[12,12]=0
    nlm_filter(dummy,100.)
    Parallel_Switch_ANLM(mirror_cpu(dummy,1), f=1,t=3,h=1.,return_stats=True)
    try:
        for dataset in args.datasets:
            for tolerance in args.tolerances:
                for level in args.levels:
                    source = ROOT/f'data/output/{dataset}/salt_pepper_{level}/full_512/results/array_nlm_salt_pepper_{level}_filtereds.pkl'
                    with source.open('rb') as stream:items=pickle.load(stream)
                    for item in items[:args.max_images]:
                        reference=np.asarray(item['img_reference_np'],dtype=np.float32)
                        run_case(reference,dataset,level,tolerance,item['file_name'],args.output,methods)
                        summarize(args.output)
    finally:
        summarize(args.output)
    print('Requested run completed; summaries include only completed method records.',flush=True)


if __name__ == '__main__':
    main()
