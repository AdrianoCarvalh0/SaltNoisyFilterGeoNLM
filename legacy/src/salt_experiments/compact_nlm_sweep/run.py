"""Explore absolute h=1..1024 for CUDA NLM f=1,t=3 on 66 matched Set12 cases."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,csv,pickle,time,hashlib,datetime,argparse,statistics
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from nlm_cuda import prepare,filter_prepared

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'data/output/unifiedComparisonV1'
OUTPUT=ROOT/'data/output/compactNLMRangeV1'
LEVELS={'low':.01,'moderate':.03,'medium':.05,'high':.10,'extreme':.20}
HS=range(1,1025)


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(array):return hashlib.sha256(array.tobytes()).hexdigest()
def quantize(array):return np.clip(array,0,255).astype(np.uint8)


def evaluate(ref,out):
    out=quantize(out)
    psnr=float(peak_signal_noise_ratio(ref,out,data_range=255))
    ssim=float(structural_similarity(ref,out,data_range=255))
    return {'psnr':psnr,'ssim':ssim,'score':.5*psnr+50*ssim}


def summarize(output):
    rows=[json.loads(p.read_text()) for p in sorted(output.glob('set12/tolerance_*/*/*/nlm.json'))]
    rows=[r for r in rows if r.get('h_grid_inclusive')==[1,1024]]
    table=[]
    for row in rows:
        table.append({key:row[key] for key in ['file_name','tolerance','density','h','psnr','ssim','score','time_sweep_s','h_best_psnr','h_best_ssim','any_optimum_at_boundary']})
    if table:
        tmp=output/'results_partial.csv.tmp'
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        tmp.replace(output/'results_partial.csv')
    return rows


def main():
    global OUTPUT
    parser=argparse.ArgumentParser();parser.add_argument('--max-images',type=int,default=11)
    parser.add_argument('--levels',nargs='+',choices=list(LEVELS),default=['low','medium','extreme'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    OUTPUT=args.output
    levels={name:LEVELS[name] for name in dict.fromkeys(args.levels)}
    if not 1<=args.max_images<=11:parser.error('max-images must be 1..11')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protocol={'version':1,'method':'NLM bounded CUDA','f':1,'t':3,'h_grid_inclusive':[1,1024],
              'selection':'max 0.5 PSNR +50 SSIM; smallest h on ties; also record PSNR/SSIM optima',
              'datasets':['set12'],'tolerances':[0,4],'densities':list(levels.values()),
              'image_count':args.max_images,'total_cases':args.max_images*len(levels)*2,
              'source':'unifiedComparisonV1 saved noisy.npy with verified hashes',
              'scale':[0,255],'dtype':'float32','metrics':'clip/cast uint8; data_range=255; skimage0.20.0',
              'search':'inclusive, clipped candidate centers; symmetric patch padding',
              'h_policy':'absolute grid; no adaptive offset and no GNLM multiplier',
              'timing':'sweep includes metrics; prepared GPU image reused; not legacy runtime benchmark',
              'kernel_sha256':hashlib.sha256(Path(__file__).with_name('nlm_cuda.py').read_bytes()).hexdigest()}
    manifest=OUTPUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol mismatch')
    atomic(manifest,protocol)
    sources={}
    for level in levels:
        file=ROOT/f'data/output/set12/salt_pepper_{level}/full_512/results/array_nlm_salt_pepper_{level}_filtereds.pkl'
        with file.open('rb') as f:items=pickle.load(f)
        sources[level]={item['file_name']:np.asarray(item['img_reference_np'],dtype=np.float32) for item in items}
    names=sorted(sources[next(iter(levels))])[:args.max_images]
    state={'state':'running','updated_utc':now(),'total_cases':protocol['total_cases'],'completed':len(summarize(OUTPUT))}
    atomic(OUTPUT/'status.json',state)
    dummy=np.full((24,24),128,dtype=np.float32)
    padded,shape=prepare(dummy);filter_prepared(padded,shape,100)
    started=time.perf_counter()
    try:
        for name in names:
            for tau in [0,4]:
                for level,density in levels.items():
                    source=SOURCE/'set12'/f'tolerance_{tau}'/level/Path(name).stem
                    destination=OUTPUT/'set12'/f'tolerance_{tau}'/level/Path(name).stem
                    destination.mkdir(parents=True,exist_ok=True)
                    result=destination/'nlm.json'
                    previous=json.loads(result.read_text()) if result.exists() else None
                    if previous and previous.get('h_grid_inclusive')==[1,1024]:
                        if not (destination/'nlm.npy').exists():raise ValueError('Missing saved result image')
                        continue
                    ref=sources[level][name];noisy=np.load(source/'noisy.npy')
                    identity=json.loads((source/'case.json').read_text())
                    assert sha(ref)==identity['reference_sha256'] and sha(noisy)==identity['noisy_sha256']
                    atomic(destination/'case.json',identity)
                    reference=quantize(ref);padded,shape=prepare(noisy)
                    print(f'START {name} tau={tau} density={density} h=1..1024',flush=True)
                    state.pop('best_so_far',None)
                    state.update(current={'file_name':name,'tolerance':tau,'density':density},updated_utc=now(),h_completed=0)
                    atomic(OUTPUT/'status.json',state)
                    start=time.perf_counter();rows=[];best={};best_images={}
                    if previous:
                        assert previous['noisy_sha256']==identity['noisy_sha256']
                        assert previous['reference_sha256']==identity['reference_sha256']
                        last_h=previous['h_grid_inclusive'][1]
                        with (destination/'h_sweep.csv').open() as stream:
                            rows=[{'h':int(r['h']),**{k:float(r[k]) for k in ['psnr','ssim','score']}}
                                  for r in csv.DictReader(stream) if int(r['h'])<=last_h]
                        assert [r['h'] for r in rows]==list(range(1,last_h+1))
                        for metric in ['score','psnr','ssim']:
                            best[metric]=max(rows,key=lambda r:r[metric]).copy()
                            best_images[metric]=quantize(filter_prepared(padded,shape,best[metric]['h']))
                    reused=len(rows)
                    with (destination/'h_sweep.csv.tmp').open('w',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=['h','psnr','ssim','score']);writer.writeheader();writer.writerows(rows)
                        for h in HS:
                            if h<=reused:continue
                            output=filter_prepared(padded,shape,h)
                            if output.shape!=ref.shape or not np.isfinite(output).all():raise ValueError('Invalid CUDA output')
                            row={'h':h,**evaluate(reference,output)};writer.writerow(row);rows.append(row)
                            for metric in ['score','psnr','ssim']:
                                if metric not in best or row[metric]>best[metric][metric]:
                                    best[metric]=row.copy();best_images[metric]=quantize(output)
                            if h%32==0:
                                stream.flush();state.update(h_completed=h,best_so_far=best['score'],updated_utc=now())
                                atomic(OUTPUT/'status.json',state)
                    (destination/'h_sweep.csv.tmp').replace(destination/'h_sweep.csv')
                    elapsed=time.perf_counter()-start
                    selected=best['score'];row={**identity,'method':'nlm','f':1,'t':3,**selected,
                         'h_best_psnr':best['psnr']['h'],'h_best_ssim':best['ssim']['h'],
                         'metric_optima':best,'any_optimum_at_boundary':any(v['h'] in (1,1024) for v in best.values()),
                         'time_sweep_s':elapsed+(previous['time_sweep_s'] if previous else 0),'reused_candidates':reused,'cuda_backend':'bounded_cuda_v1','h_grid_inclusive':[1,1024]}
                    for metric,image in best_images.items():np.save(destination/f'best_{metric}.npy',image)
                    np.save(destination/'nlm.npy',best_images['score']);atomic(result,row)
                    completed=summarize(OUTPUT);state.update(completed=len(completed),updated_utc=now(),elapsed_s=time.perf_counter()-started)
                    atomic(OUTPUT/'status.json',state)
                    print(f'DONE {name} tau={tau} density={density} h_score={selected["h"]} h_psnr={best["psnr"]["h"]} h_ssim={best["ssim"]["h"]} psnr={selected["psnr"]:.4f} ssim={selected["ssim"]:.6f} seconds={elapsed:.2f} count={len(completed)}/{protocol["total_cases"]}',flush=True)
        rows=summarize(OUTPUT)
        assert len(rows)==protocol['total_cases']
        state.update(state='complete',updated_utc=now(),completed=len(rows),h_score_min=min(r['h'] for r in rows),h_score_max=max(r['h'] for r in rows),boundary_cases=sum(r['any_optimum_at_boundary'] for r in rows))
        atomic(OUTPUT/'status.json',state)
    except BaseException as error:
        state.update(state='failed',updated_utc=now(),error=repr(error));atomic(OUTPUT/'status.json',state);raise

if __name__=='__main__':main()
