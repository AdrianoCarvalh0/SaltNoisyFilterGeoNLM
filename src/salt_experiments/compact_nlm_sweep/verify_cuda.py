"""Check the new CUDA implementation against bounded CPU NLM, including edges."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from functions.nlm_functions import NLM_fast_cpu
from nlm_cuda import prepare, filter_prepared


def verify():
    rng=np.random.default_rng(42)
    images=[rng.integers(0,256,(13,17)).astype(np.float32),
            np.full((13,17),128,dtype=np.float32)]
    impulses=np.full((13,17),128,dtype=np.float32)
    impulses[0,0]=0;impulses[-1,-1]=255;impulses[6,8]=0
    images.append(impulses)
    checks=[]
    for image_id,image in enumerate(images):
        for f,t in [(1,3),(4,7)]:
            padded,shape=prepare(image,f)
            for h in [1.,25.,100.,400.]:
                actual=filter_prepared(padded,shape,h,f,t)
                expected=NLM_fast_cpu(image,h,f,t)
                assert np.isfinite(actual).all()
                assert actual.shape==image.shape
                np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-4)
                checks.append({'image_id':image_id,'f':f,'t':t,'h':h,
                               'max_abs_error':float(np.max(np.abs(actual-expected)))})
    return {'passed':True,'cases':len(checks),'checks':checks,
            'reference':'functions.nlm_functions.NLM_fast_cpu; float outputs before quantization',
            'rtol':2e-6,'atol':2e-4}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
