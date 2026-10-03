"""Run the missing 3%/10% sweeps, cross sigma, then draft rounded score ranges."""
from pathlib import Path
import sys,subprocess,json,csv,math,datetime,hashlib

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
INITIAL=ROOT/'data/output/compactNLMRangeV1'
EXTRA=ROOT/'data/output/compactNLMRangeAdditionalV1'
ANALYSIS=EXTRA/'all_levels_offsets'


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def status(state,**extra):
    atomic(EXTRA/'workflow_status.json',{'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})


def main():
    EXTRA.mkdir(parents=True,exist_ok=True)
    status('running',phase='additional_nlm_sweeps',additional_cases=44)
    try:
        subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'--levels','moderate','high','--output',str(EXTRA)],cwd=ROOT,check=True)
        status('running',phase='sigma_crosswalk')
        subprocess.run([sys.executable,str(HERE/'analyze_adaptive_offsets.py'),'--additional-root',str(EXTRA),'--dest',str(ANALYSIS)],cwd=ROOT,check=True)
        data=json.loads((ANALYSIS/'analysis.json').read_text())
        with (ANALYSIS/'per_condition.csv').open() as f:individual=list(csv.DictReader(f))
        assert data['n']==110
        summary=[r for r in data['summaries'] if r['tolerance']=='both']
        assert len(summary)==5 and all(r['n']==22 for r in summary)
        fixed={.01:(-120,120),.05:(40,170),.20:(250,600)}
        table=[]
        for r in summary:
            density=r['density'];observed_lo=r['delta_score_min'];observed_hi=r['delta_score_max']
            if density in fixed:
                lo,hi=fixed[density]
                assert lo<=observed_lo<=observed_hi<=hi
                rule='User-rounded range; 5% lower limit adjusted to +40 to include observed +47'
            else:
                lo=10*math.floor(observed_lo/10)-10
                hi=10*math.ceil(observed_hi/10)+10
                rule='Round outward to multiples of ten, then add ten on both sides'
            group=[x for x in individual if float(x['density'])==density]
            coverage=sum(lo<=int(x['delta_score'])<=hi for x in group)
            max_gap=0.;candidates=[];upper_clipped=False
            for x in group:
                tau=int(x['tolerance']);origin=int(x['h0']);stem=Path(x['file_name']).stem
                level={.01:'low',.03:'moderate',.05:'medium',.1:'high',.2:'extreme'}[density]
                base=EXTRA if density in (.03,.1) else INITIAL
                case=base/'set12'/f'tolerance_{tau}'/level/stem
                record=json.loads((case/'nlm.json').read_text())
                assert record['h_grid_inclusive']==[1,1024]
                with (case/'h_sweep.csv').open() as f:curve=list(csv.DictReader(f))
                restricted=[v for v in curve if max(1,origin+lo)<=int(v['h'])<=origin+hi]
                assert restricted
                gap=max(float(v['score']) for v in curve)-max(float(v['score']) for v in restricted)
                max_gap=max(max_gap,gap);candidates.append(origin+hi-max(1,origin+lo)+1)
                upper_clipped |= record['metric_optima']['score']['h']==1024 or origin+hi>1024
            assert coverage==22 and max_gap<1e-10
            table.append({'density':density,'n':22,'observed_offset_min':observed_lo,'observed_offset_max':observed_hi,
                          'offset_min':lo,'offset_max':hi,'score_optimum_coverage':coverage,'max_score_loss_vs_full_grid':max_gap,
                          'candidate_count_min':min(candidates),'candidate_count_max':max(candidates),
                          'needs_upper_range_review':upper_clipped,'rounding_rule':rule})
        atomic(ANALYSIS/'rounded_ranges.json',{'criterion':'0.5 PSNR +50 SSIM','definition':'positive integer h with max(1,h0+offset_min)<=h<=h0+offset_max',
                    'scope':'Set12 NLM f=1,t=3, tolerances 0/4; fitted on the same 110 calibration conditions; not independently validated',
                    'ranges':table})
        lines=['# Faixas arredondadas de h em torno da origem adaptativa','',
               'As cinco densidades foram exploradas usando 1024 candidatos absolutos por condição, com CUDA corrigido. Os 110 casos correspondem a 11 imagens Set12 × 5 densidades × 2 tolerâncias. As 66 condições iniciais foram preservadas e apenas as 44 condições de 3%/10% foram acrescentadas.','',
               'Critério: máximo de 0,5 PSNR +50 SSIM. As faixas abaixo são deslocamentos em relação a h0=compute_adaptive_q(estimate_sigma(I)), não valores absolutos. Usar apenas inteiros positivos: h_min=max(1,h0+delta_min), h_max=h0+delta_max.','',
               '| Ruído | Deslocamentos observados | Faixa arredondada proposta | Cobertura | Candidatos por imagem |',
               '|---|---:|---:|---:|---:|']
        for r in table:
            lines.append(f"| {100*r['density']:g}% | {r['observed_offset_min']:+d} a {r['observed_offset_max']:+d} | {r['offset_min']:+d} a {r['offset_max']:+d} | {r['score_optimum_coverage']}/22 | {r['candidate_count_min']}–{r['candidate_count_max']} |")
        lines+=['','As faixas cobrem os h selecionados nas 110 condições e a restrição não reduz o máximo do score nessas mesmas curvas. Essa verificação é retrospectiva no conjunto usado para definir as faixas; não é validação independente nem demonstração de ótimo global. O limite inferior h=1 em algumas imagens também permanece um limite da exploração. Não transferir as faixas automaticamente ao GNLM, a outras janelas ou a ruído real de densidade desconhecida.','',
                'Para 5%, adotou-se +40 em vez de +50 porque o menor deslocamento observado foi +47. Para 3%/10%, aplicou-se arredondamento para fora em múltiplos de dez e margem adicional de dez em cada lado. Os valores iniciais 1% (-120,+120) e 20% (+250,+600) foram mantidos.','',
                'A escolha cabe em uma subseção de análise de sensibilidade/calibração de h. O protocolo final deve explicitar a seleção pelo score e o uso da densidade sintética conhecida. Após fixar as faixas no Set12, avaliar sua cobertura no Set50 com regra previamente congelada e sinalizar máximos nas bordas. Não chamar a redução da grade de ganho de qualidade: ela busca reduzir o custo da calibração.','',
                'A subseção bilíngue pronta para revisão está em `article_sivp/nlm_h_range_sensitivity_draft.tex`. Ela não foi inserida automaticamente no manuscrito principal. Dados completos e perdas por restrição: `data/output/compactNLMRangeAdditionalV1/all_levels_offsets/`.']
        if any(r['needs_upper_range_review'] for r in table):lines+=['','**Revisão necessária:** há condição ou faixa que toca/excede o limite superior explorado. Ampliar a grade antes de fechar o intervalo.']
        (ROOT/'docs/nlm_density_specific_ranges.md').write_text('\n'.join(lines)+'\n')
        tex=r'''% Draft generated from completed, matched Set12 NLM sweeps. Review before inclusion.
\subsection{Sensitivity to the NLM Calibration Range}
\label{subsec:nlm_h_range_sensitivity}

We explored the NLM filtering parameter with patch radius $f=1$ and search radius $t=3$ on 11 Set12 images, five impulse densities, and tolerances 0 and 4. For each of the 110 conditions, all integer values $h=1,\ldots,1024$ were evaluated using the corrected CUDA implementation, and the auxiliary score selected the lowest-$h$ maximizer. We then computed the displacement $\Delta h=h^*-h_0$ from the adaptive origin $h_0$ defined in Eq.~\eqref{eq:nlm_grid_origin}.
\pt{Exploramos o parâmetro de filtragem do NLM com raio de patch $f=1$ e raio de busca $t=3$ em 11 imagens Set12, cinco densidades de impulsos e tolerâncias 0 e 4. Para cada uma das 110 condições, todos os valores inteiros $h=1,\ldots,1024$ foram avaliados usando a implementação CUDA corrigida, e o score auxiliar selecionou o menor $h$ entre os máximos. Em seguida, calculamos o deslocamento $\Delta h=h^*-h_0$ em relação à origem adaptativa $h_0$ definida na Eq.~\eqref{eq:nlm_grid_origin}.}

The observed displacements motivate rounded, density-specific candidate intervals, restricted to positive $h$. The proposed intervals retain the selected score maximizers in all 110 calibration conditions without reducing the maximum score on the stored curves. This is a retrospective calibration result on Set12, not an independent validation of the ranges or a proof of global optimality. The same intervals have not been established for GNLM or for other patch and search sizes.
\pt{Os deslocamentos observados motivam intervalos de candidatos arredondados e específicos por densidade, restritos a $h$ positivo. Os intervalos propostos preservam os máximos selecionados do score nas 110 condições de calibração sem reduzir o score máximo nas curvas armazenadas. Esse é um resultado retrospectivo de calibração no Set12, não uma validação independente das faixas nem uma prova de otimalidade global. Os mesmos intervalos não foram estabelecidos para o GNLM ou para outros tamanhos de patch e busca.}

\begin{table}[!t]
\centering
\caption{Observed and rounded offsets from $h_0$ for compact NLM, combining tolerances 0 and 4.}
\label{tab:nlm_density_offsets}
\begin{tabular}{ccc}
\toprule
Density & Observed $\Delta h$ & Proposed offsets \\
\midrule
'''
        for r in table:
            tex+=f"{100*r['density']:g}\\% & $[{r['observed_offset_min']},{r['observed_offset_max']}]$ & $[{r['offset_min']},{r['offset_max']}]$ \\\\\n"
        tex+=r'''\bottomrule
\end{tabular}
\pt{Deslocamentos observados e arredondados em relação a $h_0$ para o NLM compacto, reunindo as tolerâncias 0 e 4. Cada densidade inclui 22 condições. As faixas são definidas pelo score combinado.}
\end{table}
'''
        (ROOT/'article_sivp/nlm_h_range_sensitivity_draft.tex').write_text(tex)
        status('complete',additional_cases=44,combined_cases=110,ranges=table)
        print('Completed all five densities and wrote rounded-range analysis.',flush=True)
    except BaseException as error:
        status('failed',error=repr(error));raise

if __name__=='__main__':main()
