"""Export complete first-image h curves; do not plot unfinished grids."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]/'data/output/compactNLMRangeV1'
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
colors=['#2374ab','#e28a24','#b63658']
summary=[]
for level,label,color in zip(['low','medium','extreme'],['1% de ruído','5% de ruído','20% de ruído'],colors):
    path=root/'set12/tolerance_0'/level/'01'
    record=json.loads((path/'nlm.json').read_text())
    assert record['h_grid_inclusive']==[1,1024]
    with (path/'h_sweep.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert [int(r['h']) for r in rows]==list(range(1,1025))
    for ax,metric,title in zip(axes,['psnr','ssim','score'],['PSNR (dB)','SSIM','Score combinado']):
        h=[int(r['h']) for r in rows];values=[float(r[metric]) for r in rows]
        ax.plot(h,values,color=color,label=label,linewidth=1.5)
        best=record['metric_optima'][metric]
        ax.scatter([best['h']],[best[metric]],color=color,s=22,zorder=3)
        ax.set(xlabel='h — escala de intensidade [0,255]',ylabel=title,xlim=(1,1024))
        ax.grid(alpha=.2)
    summary.append({k:record[k] for k in ['density','h','h_best_psnr','h_best_ssim','psnr','ssim','score']})
axes[0].legend(fontsize=8)
fig.suptitle('NLM CUDA corrigido · f=1, t=3 · imagem 01 do Set12 · tolerância 0\nResultados exploratórios de uma imagem; pontos marcam os máximos de cada métrica',fontsize=11)
fig.savefig(root/'first_image_h_curves.png',dpi=160)
fig.savefig(root/'first_image_h_curves.pdf')
print(json.dumps(summary,indent=2))
