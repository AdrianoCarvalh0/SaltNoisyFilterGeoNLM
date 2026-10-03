# Revisão de Main.tex e roteiro para o Prism

Revisão baseada no manuscrito, código e planilhas locais. O Main.tex não foi alterado. Não foram refeitos os experimentos nem auditadas as referências bibliográficas externas. As linhas abaixo referem-se à versão atual.

## Parecer

A nova configuração permite apresentar IANLM com h absoluto fixo, sem selecionar h_NLM durante sua aplicação. Os resultados disponíveis sustentam desempenho em sal e pimenta e em impulsos sintéticos próximos dos extremos. Ainda não sustentam uma afirmação de validação com ruído real adquirido. O principal cuidado científico é explicar a predominância do fallback: o ganho observado não pode ser atribuído exclusivamente à agregação não local.

## Evidência quantitativa conferida

Médias aritméticas das linhas individuais, sobre as cinco densidades. Set12 sem Lena: 11 imagens × 5 densidades × 5 tolerâncias = 275 casos. Set50: 50 × 5 × 5 = 1.250 casos. São 1.525 condições experimentais, não 1.525 imagens independentes.

| Tolerância τ | Set12 PSNR (dB) | Set12 SSIM | Set50 PSNR (dB) | Set50 SSIM |
|---|---:|---:|---:|---:|
| 0 | 44.7911 | 0.992214 | 39.0086 | 0.976577 |
| 1 | 44.2633 | 0.986349 | 38.7137 | 0.975743 |
| 2 | 44.1304 | 0.984783 | 38.4590 | 0.974670 |
| 3 | 43.9942 | 0.983971 | 38.0173 | 0.971705 |
| 4 | 43.8384 | 0.983215 | 37.6471 | 0.969841 |

Fontes:
- `data/output/set12ImpulseToleranceH1/summary/results_set12_ianlm_h1_all.xlsx`
- `data/output/set50ImpulseToleranceH1/summary/results_set50_tolerance_sweep_all.xlsx`

Não substituir apenas h nas tabelas antigas: os novos números pertencem a outro protocolo de geração de ruído. Mesmo τ=0 não deve ser tratado como comparação pareada com os números históricos sem conferir as observações ruidosas.

## Mudanças por seção

1. **Resumo e contribuições (linhas 62 e 125–135):** acrescentar h fixo e extensão a impulsos próximos dos extremos. Usar “pixels classificados como confiáveis”, pois pixels limpos muito escuros ou claros também podem ser detectados. Evitar atribuir os ganhos principalmente à similaridade não local sem ablação do fallback.
2. **Formulação e algoritmo (354–440):** definir máscara Mτ(i)=1 quando y(i)≤τ ou y(i)≥255−τ, com τ inteiro de 0 a 127 na implementação; apenas 0 a 4 foram estudados nestes resultados. A mesma máscara rege preservação, rejeição de candidatos e fallback. τ=0 recupera a detecção estrita em imagens no intervalo [0,255].
3. **Fallback (409 e pseudocódigo):** corrigir o gatilho. O código usa soma de pesos ≤0, inclusive quando há candidatos válidos mas os pesos exponenciais se tornam numericamente zero. Não se limita a conjunto vazio. Descrever também o fallback local: média espacial dos valores válidos em raio f; se não há peso válido, média de toda a janela.
4. **Agregação (403–405):** o código pode substituir o valor central do candidato por uma mediana local, segundo o critério de `_robust_center_value_numba`. A equação usa apenas y(j). Introduzir o valor efetivamente agregado e seu critério. A rejeição atua no centro do candidato; a distância ainda usa todos os elementos dos patches ruidosos.
5. **Configuração experimental (576–683):** apresentar f=1, t=3, h=1 absoluto e escala [0,255]. Explicar seleção empírica no Set12 e aplicação fixa ao Set50. A independência é da calibração por imagem durante aplicação; não significa ausência de seleção experimental. Descrever seed, densidades, geração por faixas e variantes dos baselines. No Set50, o script reutiliza h_NLM de pickles anteriores para os baselines; não chamar isso de nova otimização no ruído próximo dos extremos.
6. **Sensibilidade e variantes (709–788):** incluir varredura de h absoluto, refinamento e frequência de fallback. Manter a antiga análise do multiplicador como resultado histórico claramente identificado, ou mover para suplemento. A comparação antiga IANLM/GHNLM usa h escalado; os novos resultados H1 não contêm uma comparação GHNLM equivalente. Restringir a conclusão geodésica à configuração efetivamente avaliada.
7. **Resultados (789–1237):** atualizar tabelas, rankings, vitórias e figuras em conjunto caso H1 se torne a configuração principal. A tabela acima é um resumo conferido, não substitui a análise por densidade. Separar τ=0 de τ>0. A queda de desempenho ao aumentar τ envolve mudança simultânea no ruído e no detector.
8. **Tempos (1238–1291):** usar medições da configuração correspondente e explicitar aquecimento JIT e escopo cronometrado. Os dois scripts H1 não têm instrumentação idêntica. Não transportar automaticamente os tempos históricos para h=1.
9. **Limitações e conclusão (1292–1325):** remover a dependência obrigatória de h_NLM como limitação da variante H1. Incluir falsos positivos nos extremos, seleção da tolerância, predominância do fallback e ausência de validação real identificada. Os atuais 44.91/39.03 dB são resultados históricos, não os novos 44.7911/39.0086 dB.
10. **Reprodutibilidade:** a conclusão afirma que já há instruções públicas de execução, mas o README local contém apenas o título. Ajustar a afirmação ou fornecer a documentação. A biblioteca permanece trabalho futuro até o empacotamento e validação da API.

## Como organizar sensibilidade e ablação

### Já disponível: sensibilidade de h

Varredura ampla: {0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512}. Refinamento: {0.25, 0.375, 0.5, 0.625, 0.75, 1, 1.25, 1.5, 2, 3}. As planilhas agregadas mostram h=1 como melhor score médio por tolerância entre os valores refinados. Isso não implica ótimo universal ou melhor h por imagem.

Fontes: `data/output/set12ImpulseTolerance/ianlm_independent_h/`, incluindo `refined/`.

Para τ=0, a varredura registra:

| h | PSNR médio | SSIM médio | Fração média de fallback |
|---|---:|---:|---:|
| 0.125 | 44.7493 | 0.992137 | 96.60% |
| 1 | 44.7911 | 0.992214 | 92.51% |
| 8 | 36.9880 | 0.969232 | 11.10% |
| 16 | 36.5233 | 0.966380 | 0.22% |

Em h=1, a fração média de fallback varia de 92.51% a 93.45% entre as tolerâncias no Set12. O denominador é o número de pixels processados em cada imagem; os percentuais são médias dessas frações, não a fração global ponderada. O Set50 H1 não registra esses contadores.

A frequência é medida; atribuir cada ocorrência especificamente a underflow exige instrumentação adicional. O código confirma que pesos numericamente nulos podem ativar o fallback mesmo com candidatos válidos. Isso também limita a interpretação de empates Euclidiano/geodésico como evidência sobre a utilidade das distâncias.

### Já disponível: robustez em faixas próximas dos extremos

O experimento usa o mesmo τ para gerar ruído em [0,τ] e [255−τ,255] e para detectá-lo. Portanto, é avaliação sob modelo de ruído conhecido, não ablação isolada da tolerância nem prova de que τ=4 é um padrão ideal para qualquer imagem.

O recall registrado é 1 por construção nesse cenário. A precisão média cai de 0.9287 para 0.8894 no Set12 e de 0.8528 para 0.7844 no Set50 entre τ=0 e τ=4, mostrando a relevância dos falsos positivos.

### Ablações a acrescentar, ainda não comprovadas por esta revisão

- Na mesma observação ruidosa, variar separadamente a amplitude do ruído e a tolerância do detector; comparar detecção estrita e tolerante.
- Comparar IANLM completo com fallback isolado, mantendo máscara, janela, pesos e bordas. O ASWMF atual com raio 3 não é esse controle: o fallback IANLM usa raio f=1.
- Isolar substituição robusta do centro e ponderação espacial, mantendo h e todos os demais parâmetros fixos.
- Para atribuir ganhos à distância, comparar Euclidiano e geodésico em condições com participação não local mensurável e relatar fallback de ambos.

Antes de usar a grande vantagem sobre ASWMF em τ>0 como argumento central, auditar a queda abrupta desse baseline: no Set12, PSNR médio passa de 35.3049 (τ=0) para 24.1188 (τ=1), embora o script passe tolerância ao filtro. A revisão não determinou a causa.

## Texto LaTeX sugerido para adaptação

```latex
We additionally evaluate an independent configuration with a fixed
$h=1$ for images represented in the intensity range $[0,255]$.
This configuration does not require image-wise NLM parameter selection
at inference time. Parameter sensitivity was investigated on Set12
excluding Lena, and the fixed value was subsequently used on Set50.

For near-extreme impulses, we define
\begin{equation}
M_\tau(i)=\mathbf{1}\{y(i)\leq\tau\ \lor\ y(i)\geq255-\tau\}.
\end{equation}
Pixels with $M_\tau(i)=0$ are preserved, and candidate centers with
$M_\tau(j)=1$ are excluded from aggregation. The experiments evaluate
$\tau\in\{0,1,2,3,4\}$. The case $\tau=0$ recovers strict endpoint detection.

The local fallback is activated whenever the accumulated weight is
non-positive, including cases in which valid candidates exist but their
exponential weights evaluate to zero in finite-precision arithmetic.
For $h=1$, the mean fallback fraction on Set12 ranges from approximately
$92.51\%$ to $93.45\%$ across the evaluated tolerances. Consequently,
the observed performance characterizes the complete restoration
procedure, with a substantial contribution from its local fallback.

The near-extreme experiments use synthetic corruption with matched
noise-generation bands and detection thresholds. They therefore assess
robustness within this specified impulse model; performance on acquired
real-world noise remains to be established.
```

## Instrução pronta para o Prism

Revisar Main.tex usando este parecer e os arquivos de resultados indicados. Preservar os resultados históricos como históricos; não renomeá-los como H1. Atualizar conjuntamente resumo, formulação, pseudocódigo, configuração, discussão e conclusão. Separar resultados já medidos de ablações propostas. Não inventar referências, significância estatística, testes reais ou comparações GHNLM com h=1. Evidenciar o papel do fallback e os falsos positivos. Manter consistência entre texto inglês e comentários em português. Antes de substituir tabelas comparativas completas, recalculá-las a partir dos registros da configuração selecionada, com rastreabilidade por imagem, densidade e tolerância.
