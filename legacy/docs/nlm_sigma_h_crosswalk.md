# Cruzamento de estimate_sigma, origem adaptativa e h selecionado

Análise das 66 condições completas do NLM CUDA corrigido, f=1,t=3, Set12 com 11 imagens, densidades 1%, 5% e 20%, tolerâncias 0 e 4. A varredura de referência avaliou todos os inteiros h=1..1024. Não foi necessário executar novas filtragens.

Para cada condição, estimate_sigma foi recalculado sobre a mesma observação float32 na escala [0,255], com hash conferido. O resultado coincide com o sigma arquivado em todas as 66 condições. Esse reaproveitamento é válido porque sigma é calculado diretamente da observação ruidosa; não são usados os resultados da filtragem CUDA legada com o problema nas bordas.

A função existente determina h0=int(100*clip(0.8+0.5*tanh(0.3*(sigma-1)),0.7,2.2)). Definimos delta=h_selecionado-h0 individualmente antes de resumir; os extremos de delta não foram obtidos subtraindo extremos de grupos distintos.

## Seleção pelo score 0,5 PSNR + 50 SSIM

| Ruído | n | sigma estimado | h0 | h selecionado | delta observado | Cobertura da grade antiga |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 22 | 0.371–4.506 | 70–119 | 1–175 | -118 a +104 | 16/22 |
| 5% | 22 | 3.926–10.828 | 115–129 | 175–277 | +47 a +158 | 22/22 |
| 20% | 22 | 43.219–47.466 | 129–129 | 389–723 | +260 a +594 | 19/22 |

A grade antiga tinha offsets +25..+499. Ela inclui os h selecionados de 57/66 condições: exclui seis valores no regime 1% e três no regime 20%. A análise restrita foi feita sobre as curvas do NLM corrigido, não comparando métricas com resultados do kernel antigo. Os arquivos também registram a perda no máximo de cada métrica ao restringir a grade, para distinguir exclusão de um h escolhido de eventual existência de outro máximo empatado.

## Outros critérios

| Ruído | delta para máximo PSNR | delta para máximo SSIM |
|---|---:|---:|
| 1% | +2 a +91 | -118 a +116 |
| 5% | +48 a +119 | +47 a +176 |
| 20% | +194 a +263 | +266 a +672 |

## Interpretação e limites

O ponto inicial h0 não acompanha toda a variação dos melhores h: a transformação tanh satura, chegando a h0=129 em todas as condições com 20% de ruído, apesar de sigma variar de aproximadamente 43,22 a 47,47. Isso é uma propriedade da transformação usada no código, não evidência de que as imagens tenham o mesmo ruído efetivo. Não confundir sigma estimado, densidade nominal de impulsos e parâmetro de filtragem h.

No critério score, seis condições de 1% selecionaram h=1, abaixo de h0. Esses máximos estão no limite inferior da grade e não demonstram que h=1 seja ótimo sobre todos os reais positivos. Os intervalos observados de delta podem orientar uma futura grade em torno de h0, com candidatos estritamente positivos, mas são descrições do conjunto usado para obtê-los, não validação independente nem recomendação universal. Densidades 3% e 10%, GNLM e outras janelas ainda não estão cobertos por esse cruzamento. Nenhum protocolo de filtragem foi alterado por esta análise.

Arquivos:

- `data/output/compactNLMRangeV1/adaptive_offsets/per_condition.csv`: 66 linhas com sigma, h0, três h selecionados, offsets, cobertura e perdas na grade antiga.
- `data/output/compactNLMRangeV1/adaptive_offsets/summary.csv`: resultados conjuntos e separados por tolerância.
- `data/output/compactNLMRangeV1/adaptive_offsets/analysis.json`: metadados e resumos.
- `src/salt_experiments/compact_nlm_sweep/analyze_adaptive_offsets.py`: reprodução do cruzamento.
