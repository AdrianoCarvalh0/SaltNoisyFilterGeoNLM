"""Uniform NLM(f=1,t=3) and GHNLM(f=1,t=3,k=7) comparison."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys,json,csv,pickle,time,hashlib,datetime,argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src/salt_experiments'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from nlm_cuda import prepare,filter_prepared
from functions.impulse_tolerance_filters import run_ghnlm_impulse_tolerance_pipeline
from functions.nlm_functions import compute_adaptive_q
from skimage.restoration import estimate_sigma
LEVELS={'low':.01,'moderate':.03,'medium':.05,'high':.10,'extreme':.20}
OFFSETS={.01:(-120,120),.03:(10,170),.05:(40,170),.10:(100,280),.20:(250,600)}
SOURCE=ROOT/'data/output/unifiedComparisonV1';OUT=ROOT/'data/output/compactNLMGHNLMV1'

def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def q(x):return np.clip(x,0,255).astype(np.uint8)
def sha(x):return hashlib.sha256(x.tobytes()).hexdigest()
def metrics(ref,out):
 ref=q(ref);out=q(out);p=float(peak_signal_noise_ratio(ref,out,data_range=255));s=float(structural_similarity(ref,out,data_range=255));return {'psnr':p,'ssim':s,'score':.5*p+50*s}
def load_refs(ds):
 out={}
 for level in LEVELS:
  f=ROOT/f'data/output/{ds}/salt_pepper_{level}/full_512/results/array_nlm_salt_pepper_{level}_filtereds.pkl'
  with f.open('rb') as h:items=pickle.load(h)
  out[level]={x['file_name']:np.asarray(x['img_reference_np'],dtype=np.float32) for x in items}
 return out
def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--max-images',type=int);ap.add_argument('--datasets',nargs='+',default=['set12','set50'],choices=['set12','set50']);ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
 names={ds:sorted(load_refs(ds)['low'])[:args.max_images] if args.max_images else sorted(load_refs(ds)['low']) for ds in args.datasets};refs={ds:load_refs(ds) for ds in args.datasets}
 protocol={'version':1,'datasets':args.datasets,'methods':['nlm','ghnlm'],'f':1,'t':3,'ghnlm_k':7,'tolerances':[0,4],'densities':list(LEVELS.values()),'nlm_h':'h0 plus Set12 calibrated density offsets, clipped to positive; score 0.5 PSNR + 50 SSIM','ghnlm_h':1.0,'scale':[0,255],'corrected_cuda':True,'images_per_dataset':{d:len(names[d]) for d in args.datasets}}
 if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text())!=protocol:raise ValueError('protocol mismatch')
 atomic(OUT/'protocol.json',protocol);total=sum(len(v) for v in names.values())*10*2;done=0
 for ds in args.datasets:
  for level,density in LEVELS.items():
   for name in names[ds]:
    ref=refs[ds][level][name]
    for tau in [0,4]:
     source=SOURCE/ds/f'tolerance_{tau}'/level/Path(name).stem;noisy=np.load(source/'noisy.npy');identity=json.loads((source/'case.json').read_text());assert sha(noisy)==identity['noisy_sha256']
     dest=OUT/ds/f'tolerance_{tau}'/level/Path(name).stem;dest.mkdir(parents=True,exist_ok=True)
     padded,shape=prepare(noisy);sigma=float(estimate_sigma(noisy));h0=int(compute_adaptive_q(sigma));lo,hi=OFFSETS[density];candidates=range(max(1,h0+lo),h0+hi+1)
     nr=dest/'nlm.json';gr=dest/'ghnlm.json'
     if not nr.exists():
      best=None;bestout=None;start=time.perf_counter();curve=[]
      with (dest/'nlm_h_sweep.csv').open('w',newline='') as f:
       w=csv.DictWriter(f,fieldnames=['h','psnr','ssim','score']);w.writeheader()
       for h in candidates:
        out=filter_prepared(padded,shape,h,1,3);row={'h':h,**metrics(ref,out)};w.writerow(row);curve.append(row)
        if best is None or row['score']>best['score'] or (row['score']==best['score'] and h<best['h']):best,bestout=row,out
      np.save(dest/'nlm.npy',q(bestout));atomic(nr,{**identity,'method':'nlm','f':1,'t':3,'sigma':sigma,'h0':h0,'offset_range':[lo,hi],'h_candidates':[min(candidates),max(candidates)],**best,'time_s':time.perf_counter()-start,'backend':'bounded_cuda_v1'})
     if not gr.exists():
      start=time.perf_counter();out,h,*_=run_ghnlm_impulse_tolerance_pipeline(img_original=ref,h_base=1.,img_noisy=noisy,f=1,t=3,mult=1.,nn=7,impulse_tolerance=tau,switch_impulse_only=True,reject_impulse_candidates=True,use_aswmf_spatial_weights=True);row={**identity,'method':'ghnlm','f':1,'t':3,'k':7,'h':h,'h_source':'fixed_independent','h0':h0,**metrics(ref,out),'time_s':time.perf_counter()-start,'cpu_worker_limit':8};np.save(dest/'ghnlm.npy',q(out));atomic(gr,row)
     done+=2
     atomic(OUT/'status.json',{'state':'running','completed_records':done,'total_records':total,'current':{'dataset':ds,'level':level,'file_name':name,'tolerance':tau},'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print('complete',done,total)
 atomic(OUT/'status.json',{'state':'complete','completed_records':done,'total_records':total,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':main()
