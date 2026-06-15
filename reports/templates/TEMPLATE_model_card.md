# Model Card: <Nome do Modelo>

> Preencha este template para o modelo principal atribuído ao seu grupo. Substitua os campos `<...>` pelos valores reais. Não deixe campos em branco; use "N/A" quando não aplicável.
>
> Estrutura inspirada em Mitchell et al. (2019), com extensões específicas da disciplina (fatores observados nos quatro regimes do TabArena, análise quantitativa contra baselines e AutoGluon, e seção de avisos e recomendações).

## 1. Detalhes do modelo

- **Nome:** <ex.: TabPFN-2.5>
- **Versão:** <ex.: 2.5.1>
- **Autores originais:** <ex.: Hollmann et al., 2025>
- **Repositório oficial:** <URL>
- **Licença do código:** <ex.: Apache 2.0>
- **Licença dos pesos pré-treinados (se aplicável):** <ex.: não-comercial; uso acadêmico permitido>
- **Família arquitetural:** <ex.: foundation model transformer com in-context learning>
- **Contagem de parâmetros:** <ex.: 25M; reportar treináveis vs. fixos quando aplicável>
- **Complexidade computacional:** <tempo e memória em função de n e p; ex.: O(n^2 p) em treino, O(np) em inferência>
- **Pico de memória observado:** <ex.: 7,2 GB em datasets do regime grande>
- **Toolkit / dependências:** <ex.: tabpfn 2.5, pytorch 2.x, CUDA 12 opcional>
- **Hiperparâmetros principais:** <listar; indicar se foi feita busca via Optuna>

## 2. Uso pretendido

- **Caso de uso primário:** classificação supervisionada em dados tabulares.
- **Casos de uso fora de escopo:** <ex.: dados não-IID, séries temporais, dados de imagem, dados textuais brutos sem tokenização>
- **Usuários pretendidos:** <ex.: pesquisadores e praticantes de ML em problemas tabulares com benchmarks padronizados>
- **Faixa de n suportada:** <ex.: até 50.000 amostras com bom desempenho; subamostragem ou destilação acima disso>
- **Faixa de p suportada:** <ex.: até 2.000 features numéricas; categóricas exigem one-hot ou embedding prévio>
- **Condições operacionais:** <ex.: requer GPU com pelo menos 8 GB de VRAM para inferência rápida em datasets médios>

## 3. Fatores observados

Dimensões em que o desempenho do modelo varia, avaliadas neste projeto sobre os 30 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** <descrever sensibilidade do modelo: pequeno (< 1.000), médio (1.000 a 10.000), grande (> 10.000)>
- **Número de classes:** <binário vs. multiclasse; degradação esperada conforme aumenta o número de classes>
- **Proporção entre features categóricas e numéricas:** <baixa vs. alta; impacto na codificação e no tempo de treino>
- **Presença de valores ausentes:** <com NaN vs. sem NaN; estratégia de imputação adotada>

## 4. Métricas alcançadas

Tabela agregada nos 30 datasets do TabArena. Reportar média, desvio padrão e intervalo de confiança de 95% via bootstrap (1.000 reamostragens).

| Métrica | Média | Desvio | IC 95% (bootstrap) | Ranking médio |
|---|---|---|---|---|
| AUC OvO | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| Accuracy | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| G-Mean | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| Cross-Entropy | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| Tempo total (s) | <0,0> | <0,0> | <[0,0; 0,0]> | <0,0> |

### Resultados por regime

- **Tamanho:** pequeno: AUC=<...>; médio: AUC=<...>; grande: AUC=<...>
- **Número de classes:** binário: AUC=<...>; multiclasse: AUC=<...>
- **Proporção categórica:** baixa: AUC=<...>; alta: AUC=<...>
- **Missing values:** com NaN: AUC=<...>; sem NaN: AUC=<...>

## 5. Dados de avaliação

- **Origem:** 30 datasets do TabArena-v0.1 (NeurIPS 2025), via OpenML.
- **Distribuição por regime:** 10 pequenos + 10 médios + 10 grandes.
- **Estratégia de split:** 70/30 estratificado por classe, seed=<n>.
- **Pré-processamento aplicado:** <descrever imputação, codificação categórica e escalonamento>.
- **Lista dos datasets utilizados:** <preencher com nome, OpenML task ID, n, n_features, n_classes, regime; ver tabela do relatório>.

## 6. Dados de treino e pré-treino

- **Modelo é foundation model pré-treinado, treinado do zero ou híbrido?** <responder>
- **Origem dos dados de pré-treino (se aplicável):** <ex.: 130M de datasets sintéticos gerados pelos autores>
- **Origem dos dados de treino direto (se aplicável):** <ex.: pesos inicializados aleatoriamente; treino direto nos splits do projeto>
- **Possíveis vieses herdados do pré-treino:** <descrever; relevante para foundation models>

## 7. Análise quantitativa

- **Posição no ranking médio entre os 15 sistemas avaliados** (10 modelos atribuíveis + 3 baselines + 2 AutoGluon): <x de 15>
- **Friedman + Nemenyi:** <descrever resultado global e os grupos estatisticamente equivalentes; citar o diagrama de diferença crítica>
- **Bayesian signed-rank com ROPE = 0,01 em AUC:** <descrever pares com p_equivalente acima de 0,95; pares onde o modelo do grupo é claramente melhor ou pior>
- **Comparação com AutoGluon:** <delta de AUC e custo computacional vs. preset default e vs. preset extreme 4h>
- **Quebra por regime:** <em quais regimes o modelo do grupo vence; em quais perde; provável explicação alinhada à arquitetura>

## 8. Considerações éticas

- **Riscos de uso indevido:** <ex.: viés herdado dos dados de pré-treinamento sintético, decisões opacas em domínios sensíveis>
- **Fairness por classe:** <recall e precisão por classe; classes minoritárias com baixo recall>
- **Dependência de licença de pesos pré-treinados:** <ex.: TabPFN-2.5 tem licença não-comercial; uso em produção exige avaliação jurídica>
- **Impacto ambiental:** <energia consumida durante tuning; latência de inferência; trade-off entre qualidade e custo>
- **Recomendações de auditoria:** <ex.: comparar predições com baseline interpretável como EBM antes de deploy>

## 9. Avisos e recomendações

- **Quando usar este modelo:** <regimes em que mostrou melhor desempenho ou melhor custo-benefício>
- **Quando NÃO usar este modelo:** <regimes onde os baselines vencem ou onde restrições operacionais inviabilizam o uso>
- **Alternativas recomendadas em cada caso:** <ex.: para n acima de 50K, usar LightGBM TD; para datasets com cardinalidade categórica alta, usar CatBoost TD; para AutoML genérico, usar AutoGluon default>

## 10. Reprodutibilidade

- **Ambiente:** Python <3.11>, dependências fixadas em `pyproject.toml`.
- **Hardware utilizado:** <CPU/GPU, RAM, tempo total de execução>
- **Comandos para reproduzir:**
  ```bash
  uv sync
  python -m src.pipeline.run_all --include-group-model --seed 42
  ```
- **Hash do commit:** <git rev-parse HEAD>

## 11. Referências

- <citação do paper original do modelo>
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.
- Demsar, J. (2006). Statistical comparisons of classifiers over multiple datasets. JMLR.
- Benavoli, A., Corani, G., Demsar, J., Zaffalon, M. (2017). Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. JMLR.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
