"""Cross compact NLM optima with freshly recomputed sigma and adaptive h origin.

Uses saved noisy observations and complete corrected-CUDA sweeps; runs no filters.
"""
from pathlib import Path
import sys,json,csv,hashlib,collections,datetime,argparse
import numpy as np
from skimage.restoration import estimate_sigma
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from functions.nlm_functions import compute_adaptive_q

ROOT=Path(__file__).resolve().parents[3]
COMPACT=ROOT/'data/output/compactNLMRangeV1'
SOURCE=ROOT/'data/output/unifiedComparisonV1'
DEST=COMPACT/'adaptive_offsets'


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    global DEST
    parser=argparse.ArgumentParser()
    parser.add_argument('--additional-root',type=Path)
    parser.add_argument('--dest',type=Path,default=DEST)
    args=parser.parse_args()
    DEST=args.dest
    roots=[COMPACT]+([args.additional_root] if args.additional_root else [])
    paths=[(root,p) for root in roots for p in sorted(root.glob('set12/tolerance_*/*/*/nlm.json'))]
    expected=110 if args.additional_root else 66
    seen=set()
    DEST.mkdir(parents=True,exist_ok=True)
    rows=[]
    for compact_root,path in paths:
        record=json.loads(path.read_text())
        assert record['h_grid_inclusive']==[1,1024]
        case_key=tuple(record[k] for k in ['file_name','density','tolerance'])
        assert case_key not in seen,case_key
        seen.add(case_key)
        source=SOURCE/path.parent.relative_to(compact_root)
        noisy=np.load(source/'noisy.npy')
        assert hashlib.sha256(noisy.tobytes()).hexdigest()==record['noisy_sha256']
        sigma=float(estimate_sigma(noisy));origin=int(compute_adaptive_q(sigma))
        legacy=json.loads((source/'nlm.json').read_text())
        assert legacy['reference_sha256']==record['reference_sha256']
        assert abs(sigma-legacy['sigma'])<1e-10
        with path.with_name('h_sweep.csv').open() as f:
            curve=[{'h':int(r['h']),**{k:float(r[k]) for k in ['psnr','ssim','score']}} for r in csv.DictReader(f)]
        assert [r['h'] for r in curve]==list(range(1,1025))
        row={'file_name':record['file_name'],'density':record['density'],'tolerance':record['tolerance'],
             'sigma':sigma,'h0':origin,'noisy_sha256':record['noisy_sha256'],
             'legacy_offset_lower':25,'legacy_offset_upper':499}
        restricted=[r for r in curve if 25<=r['h']-origin<=499]
        assert len(restricted)==475
        for metric in ['score','psnr','ssim']:
            best=max(curve,key=lambda r:r[metric])
            assert best['h']==record['metric_optima'][metric]['h']
            old_range_best=max(restricted,key=lambda r:r[metric])
            delta=best['h']-origin
            row.update({f'h_{metric}':best['h'],f'delta_{metric}':delta,
                        f'old_grid_contains_{metric}_selected_h':25<=delta<=499,
                        f'old_grid_best_h_{metric}':old_range_best['h'],
                        f'old_grid_gap_{metric}':best[metric]-old_range_best[metric]})
        rows.append(row)
    assert len(rows)==expected
    write_csv(DEST/'per_condition.csv',rows)
    summaries=[]
    for density in sorted({r['density'] for r in rows}):
        for tau in ['both',0,4]:
            group=[r for r in rows if r['density']==density and (tau=='both' or r['tolerance']==tau)]
            summary={'density':density,'tolerance':tau,'n':len(group)}
            for key in ['sigma','h0','h_score','h_psnr','h_ssim','delta_score','delta_psnr','delta_ssim']:
                summary[key+'_min']=min(r[key] for r in group)
                summary[key+'_max']=max(r[key] for r in group)
            for metric in ['score','psnr','ssim']:
                summary['old_grid_coverage_'+metric]=sum(r[f'old_grid_contains_{metric}_selected_h'] for r in group)
                summary['old_grid_gap_'+metric+'_max']=max(r[f'old_grid_gap_{metric}'] for r in group)
            summaries.append(summary)
    write_csv(DEST/'summary.csv',summaries)
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'n':len(rows),'definition':'delta=h_selected-h0, per condition; h0=compute_adaptive_q(estimate_sigma(noisy_float32_0_255))',
            'sigma_verification':f'Recomputed sigma equals archived sigma for all {expected} matched observations',
            'range_coverage_note':'Coverage checks selected lowest-h maximizers; metric gaps also recorded to handle ties',
            'limitations':'Descriptive Set12 f=1,t=3 result; no independent validation and no GNLM transfer',
            'summaries':summaries}
    (DEST/'analysis.json').write_text(json.dumps(report,indent=2))
    for summary in summaries:
        if summary['tolerance']=='both':print(json.dumps(summary))
    print('h0 BELOW:',sum(r['delta_score']<0 for r in rows),'OF',len(rows))

if __name__=='__main__':main()
