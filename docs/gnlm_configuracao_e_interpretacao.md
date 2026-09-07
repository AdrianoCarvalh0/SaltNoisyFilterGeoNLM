# Papel do GNLM no artigo e sensibilidade de configuração

**Atualização posterior:** identificado acesso fora dos limites nas bordas do kernel CUDA NLM usado no lote original. As diferenças numéricas abaixo descrevem os arquivos arquivados, mas suas conclusões ficam pendentes de reavaliação com NLM corrigido e recalibração dos parâmetros derivados. O lote foi pausado com 276/610 GNLM concluídos. Ver `compact_nlm_range.md`.

Análise de 2026-09-07. Fonte: `data/output/unifiedComparisonV1/set12/tolerance_*/*/*/{gnlm,nlm}.json`. Os 110 pares do Set12 estão completos (11 imagens × 5 densidades × 2 tolerâncias); hashes da observação ruidosa e da referência conferidos em cada par. Set50 ainda em processamento, não incluído nestas médias. Dados por densidade: `gnlm_vs_nlm_set12_paired.csv`.

## Evidência pareada do Set12

| Tolerância | NLM PSNR | GNLM PSNR | Delta PSNR | NLM SSIM | GNLM SSIM | Vitórias GNLM PSNR / SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 26.4029 | 27.2690 | +0.8661 | 0.734153 | 0.730676 | 55/55 e 36/55 |
| 4 | 26.4523 | 27.3270 | +0.8747 | 0.736107 | 0.733738 | 54/55 e 36/55 |

GNLM vence em PSNR em 109/110 casos e SSIM em 72/110. O SSIM médio melhora nas densidades 1%, 3% e 5%, mas cai em 10% e sobretudo em 20%; na densidade 20%, perde em SSIM nas 11 imagens de cada tolerância. Não escrever “a métrica geodésica é melhor que a euclidiana” como conclusão universal. A comparação avalia implementações e regras de h diferentes (GNLM usa gamma h_NLM), não uma intervenção isolada na métrica.

## Referência publicada e testes encontrados

A entrada `carvalho2026geo` foi preenchida e a configuração publicada foi citada em Experimental Setup. A ablação do artigo anterior considera C0=(4,7,10), C1=(4,7,5), C2=(4,7,15), C3=(2,7,10), C4=(4,5,10), sob ruído gaussiano sigma=5. Ela não avalia a alternativa conjunta (1,3,7) para sal-e-pimenta.

No workspace foram encontrados testes de patches pequenos e conectividade 7 para variantes robustas/geodésicas com mediana em `src/salt_experiments/set50_hibrid/run_set50_hibrid_param_ablation.py`, `run_set50_hibrid_h_ablation.py` e `src/salt_experiments/set12/sweep_geonlm_medians_test256.py`. Não foi localizado um teste GNLM puro (1,3,7) que possa ser identificado como equivalente ao novo protocolo; não confundir essas variantes com o baseline GNLM.

## Experimento adicional recomendado, ainda não executado

Manter a comparação principal com o GNLM publicado. Como análise de sensibilidade complementar, usar todas as 11 imagens do Set12, densidades 1%, 5% e 20%, tolerâncias 0/4: 66 condições previamente definidas, sem escolher imagens em função do resultado. Reutilizar exatamente as observações e os resultados do GNLM (4,7,10) já disponíveis; acrescentar NLM (1,3) e GNLM (1,3,7). Isso avalia a configuração compacta e também fornece um baseline Euclidiano com a mesma janela.

Calibrar novamente o NLM compacto com a mesma regra da grade e aplicar ao GNLM compacto a regra gamma*h_NLM explicitada no protocolo. Não reutilizar cegamente h da janela grande nem impor h=1 ao GNLM puro. Essa comparação preserva a política de parâmetros entre configurações, mas ainda não constitui busca direta do ótimo do GNLM. Se o objetivo for isolar causalmente a distância, será necessário um controle adicional com as demais operações e a política de h alinhadas; se o objetivo for comparar o melhor desempenho atingível, será necessária calibração própria de h para cada método.

O experimento altera conjuntamente f, t e nn, portanto mede sensibilidade à configuração compacta, sem atribuir o efeito a um parâmetro individual. Reportar todos os casos, inclusive ganhos do GNLM compacto. Planejar a execução após o lote atual para evitar concorrência nos oito processos e não contaminar comparações de duração. Não iniciado nem enfileirado nesta análise.

## Enquadramento sugerido

A contribuição central é a restauração seletiva de impulsos com rejeição de candidatos, robustificação, fallback e tolerância configurável. GNLM versus NLM é uma análise secundária da transferência da abordagem geodésica ao ruído impulsivo. GNLM versus IANLM também muda janelas, h e regras impulsivas; não atribuir toda a diferença apenas à detecção. IANLM versus GHNLM avalia a variante geodésica dentro da configuração impulsiva comum. A ablação com fallback isolado e mesma máscara/janela/pesos continua necessária para quantificar quanto do ganho depende da parte não local.

Texto em inglês para discussão do Set12, a integrar com as tabelas atualizadas:

> On the 110 matched Set12 conditions, GNLM achieved higher PSNR than NLM in 109 cases, with mean gains of 0.87 dB at both tested tolerances. This advantage did not extend uniformly to SSIM: the mean SSIM gain at lower impulse densities reversed at higher densities, particularly at 20%. These results support a PSNR benefit of the evaluated geodesic baseline under the adopted parameter policy, while showing that geodesic aggregation alone does not ensure improved structural similarity across impulse-noise regimes.

Tradução:

> Nas 110 condições correspondentes do Set12, o GNLM alcançou PSNR maior que o NLM em 109 casos, com ganhos médios de 0,87 dB nas duas tolerâncias testadas. Essa vantagem não se estendeu uniformemente ao SSIM: o ganho médio de SSIM nas densidades menores de impulsos se inverteu nas densidades maiores, especialmente em 20%. Esses resultados sustentam um benefício de PSNR do baseline geodésico avaliado sob a política de parâmetros adotada, mostrando também que a agregação geodésica isoladamente não garante melhora da similaridade estrutural em todos os regimes de ruído impulsivo.
