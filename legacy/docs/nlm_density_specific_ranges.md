# Faixas arredondadas de h em torno da origem adaptativa

As cinco densidades foram exploradas usando 1024 candidatos absolutos por condição, com CUDA corrigido. Os 110 casos correspondem a 11 imagens Set12 × 5 densidades × 2 tolerâncias. As 66 condições iniciais foram preservadas e apenas as 44 condições de 3%/10% foram acrescentadas. O Set12 é o conjunto de calibração adotado para essas faixas; não será realizada uma nova varredura no Set50.

Critério: máximo de 0,5 PSNR +50 SSIM. As faixas abaixo são deslocamentos em relação a h0=compute_adaptive_q(estimate_sigma(I)), não valores absolutos. Usar apenas inteiros positivos: h_min=max(1,h0+delta_min), h_max=h0+delta_max.

| Ruído | Deslocamentos observados | Faixa arredondada proposta | Cobertura | Candidatos por imagem |
|---|---:|---:|---:|---:|
| 1% | -118 a +104 | -120 a +120 | 22/22 | 190–239 |
| 3% | +24 a +153 | +10 a +170 | 22/22 | 161–161 |
| 5% | +47 a +158 | +40 a +170 | 22/22 | 131–131 |
| 10% | +119 a +268 | +100 a +280 | 22/22 | 181–181 |
| 20% | +260 a +594 | +250 a +600 | 22/22 | 351–351 |

As faixas cobrem os h selecionados nas 110 condições e a restrição não reduz o máximo do score nessas mesmas curvas. Essa verificação é retrospectiva no conjunto usado para definir as faixas; não é validação independente nem demonstração de ótimo global. O limite inferior h=1 em algumas imagens também permanece um limite da exploração. Não transferir as faixas automaticamente ao GNLM, a outras janelas ou a ruído real de densidade desconhecida.

Para 5%, adotou-se +40 em vez de +50 porque o menor deslocamento observado foi +47. Para 3%/10%, aplicou-se arredondamento para fora em múltiplos de dez e margem adicional de dez em cada lado. Os valores iniciais 1% (-120,+120) e 20% (+250,+600) foram mantidos.

A escolha cabe em uma subseção de análise de sensibilidade/calibração de h. O protocolo final deve explicitar a seleção pelo score, o uso da densidade sintética conhecida e o fato de que as faixas foram calibradas no Set12. Não chamar a redução da grade de ganho de qualidade: ela busca reduzir o custo da calibração.

A subseção bilíngue pronta para revisão está em `article_sivp/nlm_h_range_sensitivity_draft.tex`. Ela não foi inserida automaticamente no manuscrito principal. Dados completos e perdas por restrição: `data/output/compactNLMRangeAdditionalV1/all_levels_offsets/`.
