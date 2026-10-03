"""Bounded CUDA NLM for the new compact-window experiment.

Candidate centers are clipped to the image domain, with inclusive search limits.
Patches use symmetric padding. This matches NLM_fast_cpu's boundary convention;
it deliberately does not modify the legacy kernel used by archived experiments.
"""
import cupy as cp

KERNEL = cp.RawKernel(r'''
extern "C" __global__ void nlm_bounded(
    const float* padded, float* output,
    int rows, int cols, int f, int t, float h) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;
    int stride = cols + 2*f;
    int r0 = max(0, i-t), r1 = min(rows-1, i+t);
    int s0 = max(0, j-t), s1 = min(cols-1, j+t);
    float numerator = 0.0f, denominator = 0.0f;
    for (int r=r0; r<=r1; ++r) {
        for (int s=s0; s<=s1; ++s) {
            float d2 = 0.0f;
            for (int u=-f; u<=f; ++u) {
                for (int v=-f; v<=f; ++v) {
                    float delta = padded[(i+f+u)*stride+j+f+v]
                                - padded[(r+f+u)*stride+s+f+v];
                    d2 += delta*delta;
                }
            }
            float weight = expf(-d2/(h*h));
            numerator += weight*padded[(r+f)*stride+s+f];
            denominator += weight;
        }
    }
    output[i*cols+j] = numerator/denominator;
}
''', 'nlm_bounded', options=('--fmad=false',))


def prepare(image, f=1):
    image = cp.asarray(image, dtype=cp.float32)
    if image.ndim != 2 or min(image.shape) < 1 or f < 0:
        raise ValueError('Expected non-empty grayscale image and nonnegative radius')
    return cp.pad(image, ((f, f), (f, f)), mode='symmetric'), image.shape


def filter_prepared(padded, shape, h, f=1, t=3):
    if not 0 < h < float('inf') or t < 0:
        raise ValueError('Expected finite positive h and nonnegative search radius')
    rows, cols = shape
    if padded.shape != (rows+2*f, cols+2*f):
        raise ValueError('Padding shape does not match patch radius')
    output = cp.empty(shape, dtype=cp.float32)
    KERNEL(((cols+15)//16, (rows+15)//16), (16,16),
           (padded, output, cp.int32(rows), cp.int32(cols), cp.int32(f),
            cp.int32(t), cp.float32(h)))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(output)
