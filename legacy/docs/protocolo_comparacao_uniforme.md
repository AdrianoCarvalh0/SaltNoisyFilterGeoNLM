# Protocolo uniforme Set12/Set50 — em execução

O objetivo é substituir a mistura entre h_NLM recalibrado no Set12 e h_NLM reaproveitado no Set50. Os arquivos anteriores são preservados; nenhuma tabela anterior deve ser renomeada como resultado deste protocolo.

## Configuração comum

- Set12 sem Lena e Set50, referências em resolução original, escala [0,255].
- Cinco densidades: 0.01, 0.03, 0.05, 0.10 e 0.20; tolerâncias 0 e 4. Total esperado: 610 condições imagem × densidade × tolerância e 4.270 registros para sete métodos.
- Ruído gerado uma vez por condição e compartilhado por todos os métodos, seed 42, amostragem de posições sem reposição e valores uniformes discretos nas faixas. Hash da imagem ruidosa registrado.
- NLM: f=4, t=7, melhor h **dentro da grade** base adaptativa + offsets inteiros 25 a 499, em ambos os conjuntos e todas as densidades. Critério: 0.5 PSNR + 50 SSIM, calculado após clip/cast uint8, com data_range=255; empate resolvido pelo menor h. O intervalo ampliado contém os offsets históricos, com a mesma regra em ambos os conjuntos. Seleção na borda é registrada para revisão.
- IANLM: h=1 fixo, f=1, t=3, tolerância correspondente ao ruído. Essa é a configuração independente destinada ao uso prático.
- GHNLM: h=1 fixo, f=1, t=3, k=7, mesma tolerância do IANLM. Trata-se de uma nova comparação sob h comum, não da variante histórica h=0.001 h_NLM. Não representa otimização individual de h do GHNLM.
- GNLM: f=4, t=7, k=10; h=gamma h_NLM, gamma=1.40 quando h_NLM<60 ou sigma<10, e 1.55 caso contrário. Mantida a regra histórica, aplicada ao h_NLM recém-selecionado.
- NLMedians: f=2, t=3, h=0.005 h_NLM recém-selecionado.
- Mediana 3×3 e ASWMF raio 3, preservando as implementações existentes. No ASWMF, a tolerância controla a rejeição de vizinhos; o detector central existente usa mínimo/máximo da janela. Portanto, não descrever esse baseline como se empregasse exatamente M_tau do IANLM.

Uniformidade refere-se ao mesmo protocolo entre conjuntos e à mesma observação para todos os filtros; não a impor a todos os métodos os mesmos parâmetros. A escolha por imagem usa a referência limpa e deve ser descrita como calibração experimental, não como recurso disponível durante aplicação a uma imagem real sem referência.

## Execução e retomada

```bash
/opt/conda/envs/salt/bin/python src/salt_experiments/unified_comparison/run.py
```

Saída: `data/output/unifiedComparisonV1/`. Checkpoint por método e por imagem: a presença do JSON e do array correspondente permite retomada. A grade e os scores do NLM ficam em `nlm_h_sweep.csv`. As tabelas `results_partial.csv` e `summary_partial.csv` são explicitamente parciais até a conclusão. Cada resumo contém a contagem n; não comparar médias com amostras incompletas.

O primeiro teste usa a primeira imagem do Set12, densidade 0.01, tolerância 0, com todos os sete métodos. A execução completa deve reutilizar esses checkpoints.

Os tempos de calibração NLM e de uma execução com o h escolhido são registrados separadamente. `time_filter_call_s` inclui o escopo das funções existentes, que não é idêntico entre implementações; não usar automaticamente esses valores como benchmark final de tempo. A medição controlada dos tempos será uma etapa separada. A compilação inicial do NLM e do IANLM é aquecida antes da medição; outros métodos podem incluir inicialização na primeira chamada.

## Atualização do artigo

Depois da conclusão, atualizar Experimental Setup, tabelas de qualidade, contagens de vitórias e discussão a partir desta mesma execução. A comparação histórica permanece histórica até ser substituída. O novo protocolo não valida ruído real adquirido nem dispensa a ablação do fallback.

Execução supervisionada iniciada: `data/output/unifiedComparisonV1/batch_status.json` informa o estado; `batch.log` registra o processamento e `batch.pid` identifica o supervisor. O supervisor aguarda o término bem-sucedido dos sete métodos do piloto antes de iniciar as duas fases completas. Se houver erro, ele registra falha e não declara conclusão. Os hashes dos arquivos de implementação foram salvos em `source_hashes.json`.

Verificação inicial: 475 candidatos de h avaliados; máximo da grade conferido contra o registro selecionado; seis métodos concluídos no piloto compartilharam o mesmo hash da observação e produziram saídas com as dimensões corretas. GNLM ainda estava em processamento nesse momento.

## Atualização de recursos

Por solicitação do autor, a execução foi configurada para oito processos de CPU (Ryzen 7 5700X, oito núcleos físicos e 16 threads lógicas), com até oito threads Numba e uma thread BLAS por processo. O NLM permanece na função `NLM_fast_cuda_global`, tanto na calibração quanto na filtragem com h selecionado. A troca de processos é realizada após concluir e salvar uma imagem, retomando os checkpoints.

Os registros anteriores à mudança usaram limite de seis processos. Os novos registros informam `cpu_worker_limit=8`, `numba_thread_limit=8` e `nlm_backend`. Os tempos anteriores e posteriores à troca não devem ser agregados como se viessem da mesma configuração de execução; o benchmark final de tempo precisará usar recursos uniformes. O arquivo `restart_8_workers.json` registra a transição.
