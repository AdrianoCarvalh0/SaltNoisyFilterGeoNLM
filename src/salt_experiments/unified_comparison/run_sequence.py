"""Run the unified comparison in two resumable phases.

Phase 1 runs NLM plus the switching baselines; phase 2 runs the expensive GNLM.
Progress is tracked in ``batch_status.json`` inside the output directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

# 11 (Set12) + 50 (Set50) images x 5 levels x 2 tolerances = 610 per method.
EXPECTED_RECORDS = 610
ALL_METHODS = ['nlm', 'ianlm', 'median', 'aswmf', 'nlmedians', 'ghnlm', 'gnlm']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output or root / 'data/output/unifiedComparison'
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'batch_status.json'

    def update(state, **extra):
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(
            {'state': state,
             'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             **extra}, indent=2))
        tmp.replace(status)

    command = [sys.executable, str(Path(__file__).with_name('run.py')),
               '--output', str(output)]
    phases = [('nlm_and_switching_methods',
               ['nlm', 'ianlm', 'median', 'aswmf', 'nlmedians', 'ghnlm']),
              ('gnlm', ['gnlm'])]
    for phase, methods in phases:
        update('running', phase=phase, cpu_worker_limit=8, nlm_backend='CUDA')
        result = subprocess.run(command + ['--methods', *methods], cwd=root)
        if result.returncode:
            update('failed', phase=phase, exit_code=result.returncode)
            raise SystemExit(result.returncode)

    counts = {m: len(list(output.glob(f'set*/tolerance_*/*/*/{m}.json')))
              for m in ALL_METHODS}
    if any(n != EXPECTED_RECORDS for n in counts.values()):
        update('incomplete', counts=counts)
        raise SystemExit('Unexpected number of completed records.')
    validation = subprocess.run([
        sys.executable, str(Path(__file__).with_name('validate_archive.py')),
        '--output', str(output), '--require-complete', '--require-provenance',
    ], cwd=root)
    if validation.returncode:
        update('invalid', counts=counts, exit_code=validation.returncode)
        raise SystemExit(validation.returncode)
    update('complete', counts=counts)


if __name__ == '__main__':
    main()
