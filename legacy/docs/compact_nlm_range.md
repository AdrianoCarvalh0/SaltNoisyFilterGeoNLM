# Varredura compacta NLM — em execução

Por solicitação do autor, o GNLM original foi pausado e a prioridade passou à exploração do h do NLM compacto. A pausa usa SIGSTOP no grupo 56591 (supervisor, runner e workers): 276/610 registros GNLM concluídos preservados, além do caso em memória. A continuidade em memória depende de manter esta sessão/processos vivos; os checkpoints em disco permitem retomada posterior mesmo se a sessão terminar. Não há retomada automática programada.

## Correção necessária antes da varredura

O kernel legado `src/salt_experiments/functions/nlm_functions.py` usa padding de apenas f pixels, mas percorre centros candidatos desde im-t até im+t sem restringir os índices à imagem. No primeiro pixel, im=f, r=f-t e u=-f produzem x2=-t, que é negativo para t>0. A leitura `img_n[x2*padded_width+y2]` pode, portanto, sair da alocação; perto das laterais também pode atravessar linhas indevidamente. Não foi necessário executar esse acesso inválido para constatar o problema a partir dos limites do código.

A nova implementação `src/salt_experiments/compact_nlm_sweep/nlm_cuda.py` restringe centros candidatos ao domínio original, mantém limites inclusivos da busca e padding simétrico dos patches, de acordo com a referência `NLM_fast_cpu`. A função legada permanece intacta para preservar a identidade do lote suspenso. As estatísticas antigas de NLM, GNLM e NLMedians com h derivado precisam ser reavaliadas; o impacto quantitativo da correção ainda não foi isolado. Os resultados IANLM/GHNLM de h=1 independente não dependem dessa calibração NLM.

Validação: 24 combinações de imagens aleatórias, constantes e com impulsos nas bordas/centro; f,t=(1,3) e (4,7); h=1,25,100,400. Comparação de saídas float antes da quantização, incluindo todas as bordas. Tolerâncias rtol=2e-6, atol=2e-4; erro absoluto máximo observado 0,00015259. Relatório em `data/output/compactNLMRangeV1/cuda_verification.json`.

## Protocolo atual

- Somente NLM CUDA, f=1, t=3, escala float32 [0,255].
- Grade absoluta comum de 1024 candidatos: h=1,2,...,1024, sem offset adaptativo nem multiplicador geodésico.
- Todas as 11 imagens Set12, densidades 1%, 5%, 20% e tolerâncias 0/4: 66 condições, sem seleção por desempenho.
- Mesmas observações salvas no lote original; hashes de referência e observação conferidos antes de cada caso.
- Melhores h por PSNR, SSIM e score registrados separadamente; empates resolvidos pelo menor h. A saída principal usa score=0,5 PSNR+50 SSIM.
- Avaliação após clip/cast uint8, data_range=255. Imagens das três escolhas salvas, curvas completas em CSV e retomada por caso.
- Ótimo na borda da grade é marcado; não declarar ótimo global nem intervalo adequado a todos os casos antes de analisar essa flag.
- Custos da varredura incluem métricas e reutilizam o array preparado na GPU. Não misturar com benchmark de uma filtragem nem com tempos legados.

O diretório `data/output/compactNLMRangeV1` contém protocol.json, status.json, run.pid, run.log, results_partial.csv e subdiretórios por condição. O primeiro conjunto de seis condições usa 01.png, em todas as combinações de densidade/tolerância, e os demais seguem a mesma regra. O GNLM compacto ainda não foi iniciado: sua política de h será definida com os resultados da exploração.

## Controle necessário para GNLM versus NLM

Usar o mesmo valor nominal de t não garante candidatos idênticos: o GNLM legado usa limites superiores exclusivos nos loops de busca, enquanto o NLM usa inclusivos. O pipeline GNLM também faz padding adicional e recorte. Antes de atribuir diferenças unicamente à distância, alinhar e documentar busca, bordas, posição do nó fonte e política de h em uma implementação experimental separada. Usar h selecionado para NLM em ambos testa um h compartilhado; não equivale a otimizar GNLM individualmente. Seleção individual pelo mesmo critério também pode ser metodologicamente válida, desde que seja descrita como tal.

## Ampliação inicial do intervalo

Na primeira imagem com 20% de ruído e tolerância zero, o melhor score da grade 1..512 ficou em h=512. Uma sondagem adicional (640,768,1024,1536,2048,4096,8192) indicou queda do score após essa região. Para incluir o máximo sem truncamento, a grade comum foi ampliada para todos os inteiros 1..1024. Dez condições já haviam concluído 1..512; seus candidatos são reaproveitados e complementados, sem apresentar o intervalo menor como busca final. O manifesto inicial foi arquivado em protocol_initial_1_512.json. A sinalização de extremos permanece ativa caso outra imagem exija uma nova ampliação.

## Cruzamento com a origem adaptativa

Varredura concluída nas 66 condições. Sigma foi recalculado e cruzado com h0 e os melhores h por score, PSNR e SSIM. Relatório em `nlm_sigma_h_crosswalk.md`; dados em `data/output/compactNLMRangeV1/adaptive_offsets/`. Os offsets observados pelo score são -118..+104 (1%), +47..+158 (5%) e +260..+594 (20%). São intervalos descritivos deste experimento, sem mudança automática do protocolo. O Set12 é suficiente para esta calibração; não será realizada varredura adicional no Set50.
