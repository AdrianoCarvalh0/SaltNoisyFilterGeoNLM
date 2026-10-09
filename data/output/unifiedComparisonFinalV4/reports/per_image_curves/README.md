# Curvas por imagem — arquivo V4

Este diretório contém 80 gráficos vetoriais em PDF, derivados de
`../../results_partial.csv` e do protocolo V4 (`../../protocol.json`). Há 40
PDFs para cada conjunto (`set12/` e `set50/`): cinco densidades de ruído
impulsivo, duas tolerâncias principais (`tau0` e `tau4`), e as métricas PSNR e
SSIM nas duas formas de apresentação abaixo.

- `*_unordered.pdf`: imagens na ordem fixa declarada pelo protocolo.
- `*_ordered.pdf`: valores ordenados de forma crescente **independentemente
  para cada método e para cada métrica**. Portanto, o eixo horizontal é a
  posição/rank e não identifica uma mesma imagem entre as curvas.

Os painéis não têm título; conjunto, intensidade, tolerância, métrica e ordem
estão codificados no nome do PDF. A escala vertical é calculada por gráfico.
Para SSIM, os ticks usam passo de 0,03 e uma margem de 20% da amplitude dos
valores do grupo. A largura e os ticks horizontais também se ajustam ao número
de imagens; a legenda usa posicionamento automático. IANLM é mostrado em
vermelho para destaque.

Os arquivos são produzidos por
`src/salt_experiments/unified_comparison/reports/generate_v4_result_curves.py`.
O `manifest.json` registra a revisão, a hash da árvore-fonte e a lista completa
dos PDFs gerados.
