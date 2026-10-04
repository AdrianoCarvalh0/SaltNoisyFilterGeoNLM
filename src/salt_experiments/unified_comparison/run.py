"""Unified, self-contained Set12/Set50 salt-and-pepper comparison.

Everything needed to reproduce the experiment from scratch is local to this
package: the filters and helpers live in ``lib/`` and the clean references are
read directly from ``data/input/``. Noise is generated on the fly from a fixed
seed, so no previously produced data is required. Runs are resumable per image
and per method.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Avoid multiplying BLAS threads inside the graph workers.
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '8'
os.environ['NUMBA_NUM_THREADS'] = '8'

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
WORKER_LIMIT = 8

import cupy as cp
import numpy as np
import pandas as pd
import numba
import scipy
import skimage
from PIL import Image
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

from lib.noisy_functions import add_near_extreme_impulse_noise
from lib.nlm_functions import NLM_fast_cuda_global, compute_adaptive_q
from lib.anlm_functions import Parallel_Switch_ANLM
from lib.geonlm_functions import run_geonlm_pipeline
from lib.impulse_tolerance_filters import (
    run_ghnlm_impulse_tolerance_pipeline, aswmf_impulse_tolerance_filter,
)
from lib.nlmedians import run_nlmedians

LEVELS = {'low': .01, 'moderate': .03, 'medium': .05, 'high': .10, 'extreme': .20}
METHODS = ('nlm', 'ianlm', 'median', 'aswmf', 'nlmedians', 'ghnlm', 'gnlm')
INPUT_ROOT = ROOT / 'data' / 'input'
# Image basenames per dataset (extensions resolved at load time).
DATASET_IMAGES = {
    'set12': [f'{k:02d}' for k in range(1, 12)],   # 01..11 (11 images)
    'set50': [str(k) for k in range(0, 50)],        # 0..49 (50 images)
}


def source_tree_sha256():
    """Digest the canonical Python sources that define this experiment."""
    package = Path(__file__).resolve().parent
    digest_value = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest_value.update(path.relative_to(package).as_posix().encode())
        digest_value.update(b'\0')
        digest_value.update(path.read_bytes())
        digest_value.update(b'\0')
    return digest_value.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def git_describe():
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:  # pragma: no cover - host dependent
        pass
    return platform.processor() or 'unavailable'


def runtime_manifest():
    manifest = {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'scikit_image': skimage.__version__,
        'numba': numba.__version__,
        'cupy': cp.__version__,
        'cpu_model': cpu_model(),
        'os_cpu_count': os.cpu_count(),
        'cpu_worker_limit': WORKER_LIMIT,
        'numba_thread_limit': int(os.environ['NUMBA_NUM_THREADS']),
        'blas_thread_limits': {
            name: os.environ.get(name)
            for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')
        },
    }
    try:
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties['name']
        manifest['cuda_device'] = name.decode() if isinstance(name, bytes) else str(name)
        manifest['cuda_total_global_memory'] = int(properties['totalGlobalMem'])
        manifest['cuda_runtime'] = int(cp.cuda.runtime.runtimeGetVersion())
        manifest['cuda_driver'] = int(cp.cuda.runtime.driverGetVersion())
        manifest['cuda_compute_capability'] = [
            int(properties['major']), int(properties['minor'])
        ]
    except Exception as exc:  # pragma: no cover - host dependent
        manifest['cuda_device'] = f'unavailable: {exc}'
    return manifest


PROTOCOL = {
    'version': 4, 'seed': 42, 'tolerances': [0, 4],
    'git_revision': git_revision(),
    'git_describe': git_describe(),
    'source_tree_sha256': source_tree_sha256(),
    'dependency_lock_sha256': {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (
            ROOT / '.devcontainer' / 'conda-spec-linux-64.txt',
            ROOT / '.devcontainer' / 'requirements-pip.txt',
        )
    },
    'reference_source': 'data/input/<dataset>; grayscale; float32 in [0,255]',
    'dataset_images': DATASET_IMAGES,
    'densities': LEVELS,
    'methods': list(METHODS),
    'nlm': {'f': 1, 't': 3, 'padding': 't+f', 'offsets_by_density': {
        'low': [-120, 120], 'moderate': [10, 170], 'medium': [40, 170],
        'high': [100, 280], 'extreme': [250, 600]},
            'selection': 'maximize 0.5*PSNR + 50*SSIM; uint8; smallest h on ties'},
    'ianlm': {'f': 1, 't': 3, 'h': 1.0, 'padding': 'f+t',
              'search_grid': 'inclusive (2t+1)^2', 'switching': 'input mask'},
    'ghnlm': {'f': 1, 't': 3, 'h': 1.0, 'k': 7, 'padding': 'f+t',
              'search_grid': 'inclusive (2t+1)^2', 'source': 'target coordinate'},
    'gnlm': {'f': 1, 't': 3, 'k': 7,
             'h': 'h_nlm*(1.40 if h_nlm<60 or sigma<10 else 1.55)',
             'padding': 'f+t', 'search_grid': 'inclusive (2t+1)^2',
             'source': 'target coordinate'},
    'nlmedians': {'f': 2, 't': 2, 'h_multiplier': .005,
                   'selection': 'Set12 medium-noise structural ablation; PSNR/runtime compromise'},
    'aswmf': {'radius': 3, 'same_tolerance_as_ianlm': True},
    'median': {'size': 3, 'mode': 'reflect'},
    'spatial_weights': [1., 1., 10.], 'z_alpha': 1.96, 'outlier_alpha': 0.,
    'noise': {
        'generator': 'add_near_extreme_impulse_noise',
        'assignment': 'sample 2*ceil((density/2)*N) positions without replacement',
        'salt_probability': 'density/2', 'pepper_probability': 'density/2',
        'value_bands': '[0,tolerance] and [255-tolerance,255], discrete uniform',
        'pairing': 'same seed, positions, polarity, and uniform draws across tolerances',
    },
    'metrics': {
        'quantization': 'clip to [0,255], cast uint8', 'data_range': 255,
        'psnr': 'skimage.metrics.peak_signal_noise_ratio',
        'ssim': 'skimage.metrics.structural_similarity defaults; 2D grayscale',
        'score': '0.5*PSNR + 50*SSIM',
    },
    'nlm_backend': 'NLM_fast_cuda_global; symmetric mirror padding f+t (bounds-safe)',
    'execution': {'joblib_workers': WORKER_LIMIT, 'blas_threads': 1, 'numba_threads': 8},
}


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def load_reference(dataset, name):
    """Load a clean grayscale reference (float32, [0,255]) from data/input."""
    matches = sorted((INPUT_ROOT / dataset).glob(f'{name}.*'))
    if not matches:
        raise FileNotFoundError(f'No input image for {dataset}/{name} in {INPUT_ROOT/dataset}')
    with Image.open(matches[0]) as handle:
        image = np.asarray(handle.convert('L'), dtype=np.float32)
    return image


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
    return best_out, {**best, 'sigma': sigma,
                      'h_source': 'current_noise_full_reference_sweep',
                      'time_calibration_s': time.perf_counter() - start}


def run_case(reference, dataset, level, tolerance, name, output, methods):
    destination = output / dataset / f'tolerance_{tolerance}' / level / Path(name).stem
    destination.mkdir(parents=True, exist_ok=True)
    noisy, mask = add_near_extreme_impulse_noise(
        reference, salt_prob=LEVELS[level] / 2, pepper_prob=LEVELS[level] / 2,
        impulse_tolerance=tolerance, seed=PROTOCOL['seed'], return_mask=True,
    )
    noisy = np.asarray(noisy, dtype=np.float32)
    case = {'dataset': dataset, 'level': level, 'file_name': name, 'tolerance': tolerance,
            'density': LEVELS[level], 'shape': list(reference.shape),
            'noise_seed': PROTOCOL['seed'], 'salt_probability': LEVELS[level] / 2,
            'pepper_probability': LEVELS[level] / 2,
            'reference_sha256': digest(reference), 'noisy_sha256': digest(noisy),
            'corruption_mask_sha256': digest(mask),
            'assigned_impulses': int(mask.sum())}
    path = destination / 'case.json'
    if path.exists() and json.loads(path.read_text()) != case:
        raise ValueError(f'Case identity changed: {destination}')
    atomic_json(path, case)
    np.save(destination / 'noisy.npy', noisy)
    np.save(destination / 'corruption_mask.npy', mask)
    detected = (noisy <= tolerance) | (noisy >= 255 - tolerance)
    tp = int(np.count_nonzero(detected & mask))
    atomic_json(destination / 'detector.json', {
        'precision': tp / max(1, int(detected.sum())), 'recall': tp / max(1, int(mask.sum())),
        'true_positives': tp, 'detected_pixels': int(detected.sum()),
        'assigned_impulses': int(mask.sum()),
        'false_positives': int(np.count_nonzero(detected & ~mask)),
    })
    common = dict(img_original=reference, img_noisy=noisy, h_base=1., mult=1.,
                  f=1, t=3, impulse_tolerance=tolerance,
                  reject_impulse_candidates=True, use_aswmf_spatial_weights=True)
    for method in methods:
        result_path = destination / f'{method}.json'
        if result_path.exists():
            if not (destination / f'{method}.npy').exists():
                raise ValueError(f'Missing checkpoint image: {destination}/{method}')
            continue
        print(f'START {dataset} tau={tolerance} {level} {name} {method}', flush=True)
        info = {}
        method_parameters = {}
        if method == 'nlm':
            _, info = calibrate(reference, noisy, destination, level)
        if method in ('gnlm', 'nlmedians'):
            nlm = json.loads((destination / 'nlm.json').read_text())
        start = time.perf_counter()
        if method == 'nlm':
            filtered = nlm_filter(noisy, info['h'])
            method_parameters.update(f=1, t=3, h=info['h'])
        elif method == 'ianlm':
            filtered, stats = Parallel_Switch_ANLM(noisy, f=1, t=3,
                h=1., n_jobs=WORKER_LIMIT, impulse_tolerance=tolerance, return_stats=True)
            info.update(h=1., h_source='fixed_independent', **stats)
            method_parameters.update(f=1, t=3, h=1.)
        elif method == 'ghnlm':
            filtered, h, *_ = run_ghnlm_impulse_tolerance_pipeline(
                **common, nn=7, n_jobs=WORKER_LIMIT
            )
            info.update(h=h, h_source='fixed_independent')
            method_parameters.update(f=1, t=3, k=7, h=h)
        elif method == 'gnlm':
            gamma = 1.4 if nlm['h'] < 60 or nlm['sigma'] < 10 else 1.55
            filtered, h, *_ = run_geonlm_pipeline(reference, nlm['h'], noisy,
                                                  f=1, t=3, mult=gamma, nn=7,
                                                  n_jobs=WORKER_LIMIT)
            info.update(h=h, gamma=gamma, h_source='scaled_current_nlm')
            method_parameters.update(f=1, t=3, k=7, h=h, gamma=gamma)
        elif method == 'median':
            filtered = median_filter(noisy, size=3, mode='reflect')
            method_parameters.update(window_size=3, boundary_mode='reflect')
        elif method == 'aswmf':
            filtered = aswmf_impulse_tolerance_filter(noisy, impulse_tolerance=tolerance,
                                                      radius=3)
            method_parameters.update(radius=3, impulse_tolerance=tolerance)
        else:
            h = nlm['h'] * .005
            filtered = run_nlmedians(reference=reference, noisy=noisy, h=h, f=2, t=2)['filtered']
            info.update(h=h, h_source='scaled_current_nlm')
            method_parameters.update(f=2, t=2, h=h)
        elapsed = time.perf_counter() - start
        if filtered.shape != reference.shape:
            raise ValueError(f'Output shape mismatch for {method}')
        row = {**case, 'method': method, **info, **method_parameters, **metrics(reference, filtered),
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
        df.to_csv(output / 'results_partial.csv', index=False)
        df.groupby(['dataset', 'tolerance', 'level', 'method']).agg(
            n=('score', 'size'), psnr=('psnr', 'mean'), ssim=('ssim', 'mean'),
            score=('score', 'mean'), time_filter_call_s=('time_filter_call_s', 'mean'),
        ).to_csv(output / 'summary_partial.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['set12', 'set50'],
                        default=['set12', 'set50'])
    parser.add_argument('--levels', nargs='+', choices=LEVELS, default=list(LEVELS))
    parser.add_argument('--tolerances', nargs='+', type=int, choices=[0, 4], default=[0, 4])
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--output', type=Path, default=ROOT / 'data/output/unifiedComparison')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    # Always calibrate first: downstream h derives from the current NLM sweep.
    methods = [m for m in METHODS if m in set(args.methods) | {'nlm'}]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != PROTOCOL:
        raise ValueError('Different protocol in output directory; use a new directory.')
    atomic_json(manifest, PROTOCOL)
    atomic_json(args.output / 'runtime.json', runtime_manifest())
    requests_path = args.output / 'run_requests.json'
    requests = json.loads(requests_path.read_text()) if requests_path.exists() else []
    request = {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'argv': sys.argv[1:], 'datasets': args.datasets, 'levels': args.levels,
        'tolerances': args.tolerances, 'max_images': args.max_images,
        'requested_methods': args.methods, 'effective_methods': methods,
    }
    requests.append(request)
    atomic_json(requests_path, requests)
    # Warm up kernels outside per-image timings; no reference data used here.
    dummy = np.full((24, 24), 128, dtype=np.float32); dummy[12, 12] = 0
    nlm_filter(dummy, 100.)
    Parallel_Switch_ANLM(dummy, f=1, t=3, h=1., return_stats=True)
    try:
        for dataset in args.datasets:
            names = DATASET_IMAGES[dataset][:args.max_images]
            for tolerance in args.tolerances:
                for level in args.levels:
                    for name in names:
                        reference = load_reference(dataset, name)
                        run_case(reference, dataset, level, tolerance, name,
                                 args.output, methods)
                        summarize(args.output)
    finally:
        summarize(args.output)
    print('Requested run completed; summaries include only completed method records.',
          flush=True)


if __name__ == '__main__':
    main()
