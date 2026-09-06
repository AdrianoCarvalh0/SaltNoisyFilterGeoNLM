"""Continue after a pilot and run the shared comparison in two resumable phases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = root/'data/output/unifiedComparisonV1'
    status = output/'batch_status.json'

    def update(state, **extra):
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps({'state': state, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **extra}, indent=2))
        tmp.replace(status)

    if args.wait_pid:
        update('waiting_for_pilot', pilot_pid=args.wait_pid)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
        pilot = output/'set12/tolerance_0/low/01'
        for method in ['nlm','ianlm','median','aswmf','nlmedians','ghnlm','gnlm']:
            if not (pilot/f'{method}.json').exists():
                update('pilot_incomplete', missing_method=method)
                raise SystemExit('Pilot did not finish all methods; batch not started.')

    command=[sys.executable,str(Path(__file__).with_name('run.py'))]
    phases = [('nlm_and_switching_methods',['nlm','ianlm','median','aswmf','nlmedians','ghnlm']),
              ('gnlm',['gnlm'])]
    for phase,methods in phases:
        update('running',phase=phase,cpu_worker_limit=8,nlm_backend='CUDA')
        result=subprocess.run(command+['--methods',*methods],cwd=root)
        if result.returncode:
            update('failed',phase=phase,exit_code=result.returncode)
            raise SystemExit(result.returncode)
    counts={m:len(list(output.glob(f'set*/tolerance_*/*/*/{m}.json')))
            for m in ['nlm','ianlm','median','aswmf','nlmedians','ghnlm','gnlm']}
    if any(n!=610 for n in counts.values()):
        update('incomplete',counts=counts)
        raise SystemExit('Unexpected number of completed records.')
    update('complete',counts=counts)


if __name__=='__main__':
    main()
