"""Validate a unified-comparison archive before it is cited as final."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASET_IMAGES, LEVELS, METHODS, PROTOCOL  # noqa: E402


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_reference(dataset, name):
    matches = sorted((ROOT / 'data' / 'input' / dataset).glob(f'{name}.*'))
    if not matches:
        raise AssertionError(f'Missing reference input: {dataset}/{name}')
    with Image.open(matches[0]) as handle:
        return np.asarray(handle.convert('L'), dtype=np.float32)


def validate(output, require_complete=False, require_provenance=False):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    if require_provenance:
        assert protocol['version'] >= 4
        assert protocol['git_revision'] != 'unavailable'
        assert protocol['git_describe'] != 'unavailable'
        assert len(protocol['source_tree_sha256']) == 64
        assert protocol['dataset_images'] == DATASET_IMAGES
        assert protocol['densities'] == LEVELS
        assert protocol['methods'] == list(METHODS)
        runtime = json.loads((output / 'runtime.json').read_text())
        assert {'python', 'platform', 'numpy', 'scikit_image', 'numba', 'cupy',
                'cpu_model', 'cuda_device'} <= runtime.keys()
        requests = json.loads((output / 'run_requests.json').read_text())
        assert requests

    rows = []
    case_paths = sorted(output.glob('set*/tolerance_*/*/*/case.json'))
    for case_path in case_paths:
        case = json.loads(case_path.read_text())
        directory = case_path.parent
        reference = load_reference(case['dataset'], case['file_name'])
        noisy = np.load(directory / 'noisy.npy')
        mask = np.load(directory / 'corruption_mask.npy')
        assert tuple(case['shape']) == reference.shape == noisy.shape == mask.shape
        assert case['reference_sha256'] == digest(reference)
        assert case['noisy_sha256'] == digest(noisy)
        assert case['corruption_mask_sha256'] == digest(mask)
        assert case['assigned_impulses'] == int(mask.sum())

        detected = (noisy <= case['tolerance']) | (noisy >= 255 - case['tolerance'])
        detector = json.loads((directory / 'detector.json').read_text())
        tp = int(np.count_nonzero(detected & mask))
        assert detector['false_positives'] == int(np.count_nonzero(detected & ~mask))
        assert detector['true_positives'] == tp
        assert detector['detected_pixels'] == int(detected.sum())
        assert detector['assigned_impulses'] == int(mask.sum())
        assert detector['precision'] == tp / max(1, int(detected.sum()))
        assert detector['recall'] == tp / max(1, int(mask.sum()))

        method_records = []
        for method in METHODS:
            record_path = directory / f'{method}.json'
            assert record_path.exists(), f'Missing {record_path}'
            record = json.loads(record_path.read_text())
            restored = np.load(directory / f'{method}.npy')
            assert restored.dtype == np.uint8 and restored.shape == reference.shape
            assert np.isfinite(restored).all()
            assert record['reference_sha256'] == case['reference_sha256']
            assert record['noisy_sha256'] == case['noisy_sha256']
            psnr = float(peak_signal_noise_ratio(reference.astype(np.uint8), restored, data_range=255))
            ssim = float(structural_similarity(reference.astype(np.uint8), restored, data_range=255))
            assert abs(record['psnr'] - psnr) < 1e-12
            assert abs(record['ssim'] - ssim) < 1e-12
            method_records.append(record)
        rows.extend(method_records)

        sweep = pd.read_csv(directory / 'nlm_h_sweep.csv')
        selected = next(record for record in method_records if record['method'] == 'nlm')
        assert (sweep['h'] == selected['h']).any()

    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['dataset', 'level', 'file_name', 'tolerance', 'method']).any()
    assert len(frame) == len(case_paths) * len(METHODS)
    # Keep Set12 identifiers such as ``01`` as strings instead of allowing
    # pandas to coerce them to integers while reading the aggregate CSV.
    saved = pd.read_csv(output / 'results_partial.csv', dtype={'file_name': str})
    assert len(saved) == len(frame)
    keys = ['dataset', 'level', 'file_name', 'tolerance', 'method']
    assert set(map(tuple, saved[keys].to_numpy())) == set(map(tuple, frame[keys].to_numpy()))

    expected_summary = frame.groupby(['dataset', 'tolerance', 'level', 'method']).agg(
        n=('score', 'size'), psnr=('psnr', 'mean'), ssim=('ssim', 'mean'),
        score=('score', 'mean'), time_filter_call_s=('time_filter_call_s', 'mean'),
    ).reset_index()
    saved_summary = pd.read_csv(output / 'summary_partial.csv')
    merged = expected_summary.merge(
        saved_summary, on=['dataset', 'tolerance', 'level', 'method'], suffixes=('_expected', '_saved')
    )
    assert len(merged) == len(expected_summary) == len(saved_summary)
    for column in ('n', 'psnr', 'ssim', 'score', 'time_filter_call_s'):
        assert np.allclose(merged[f'{column}_expected'], merged[f'{column}_saved'], rtol=0, atol=1e-12)

    # The generator intentionally pairs tolerance cases by assignment mask.
    for dataset, names in DATASET_IMAGES.items():
        for level in LEVELS:
            for name in names:
                zero = output / dataset / 'tolerance_0' / level / name / 'corruption_mask.npy'
                four = output / dataset / 'tolerance_4' / level / name / 'corruption_mask.npy'
                if zero.exists() and four.exists():
                    np.testing.assert_array_equal(np.load(zero), np.load(four))

    if require_complete:
        expected_cases = sum(len(names) for names in DATASET_IMAGES.values()) * len(LEVELS) * 2
        assert len(case_paths) == expected_cases
        assert len(frame) == expected_cases * len(METHODS)
    return len(case_paths), len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--require-provenance', action='store_true')
    args = parser.parse_args()
    cases, records = validate(args.output, args.require_complete, args.require_provenance)
    print(f'Archive valid: {cases} cases, {records} method records.')


if __name__ == '__main__':
    main()
