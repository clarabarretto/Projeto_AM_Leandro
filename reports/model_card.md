# Model Card: TabICL v2

Preenchido para o modelo TabICL v2, atribuído ao grupo conforme o template da disciplina de Aprendizagem de Máquina (2026-1). Estrutura inspirada em Mitchell et al. (2019), com extensões específicas da disciplina.

---

## 1. Detalhes do modelo

- **Nome:** TabICL v2 (TabICLv2)
- **Versão:** 2.1.1 (checkpoint `tabicl-classifier-v2-20260212.ckpt`)
- **Autores originais:** Schlegel, V., Zhu, Y., Leite, R., Varoquaux, G. (Soda team, Inria)
- **Repositório oficial:** https://github.com/soda-inria/tabicl
- **Licença do código:** Apache 2.0
- **Licença dos pesos pré-treinados:** Apache 2.0 — uso acadêmico e comercial permitidos sem restrições
- **Família arquitetural:** Foundation model Transformer com In-Context Learning (ICL) para dados tabulares
- **Contagem de parâmetros:** 27.552.258 totais (27.552.250 treináveis; 8 fixos — buffers de normalização)
- **Complexidade computacional:** O(n² · p) na inferência (atenção cruzada entre amostras de treino e teste); mitigada com Query-Aware Scalable Softmax para grandes volumes
- **Pico de memória observado:** < 4 GB em CPU para regimes Small e Medium; pode exceder 50 GB em GPU para datasets com n > 100K e p > 500
- **Toolkit / dependências:** `tabicl==2.1.1`, PyTorch 2.7.1, scikit-learn 1.7.2; CUDA opcional para aceleração GPU
- **Hiperparâmetros principais e faixas de busca (Optuna, TPE, 20 trials):**

| Hiperparâmetro | Tipo / Faixa | Descrição |
|---|---|---|
| `n_estimators` | int / [4, 32] | Número de membros do ensemble de contexto |
| `softmax_temperature` | float / [0.3, 1.5] | Calibração de confiança das probabilidades |
| `outlier_threshold` | float / [2.0, 6.0] | Limiar de clipping de features numéricas |
| `average_logits` | bool / {True, False} | Combinação em logits vs. probabilidades |
| `norm_methods` | categórico / {quantile, robust, ...} | Método de normalização do embedding contínuo |

O modelo opera de forma competitiva com os hiperparâmetros padrão; o tuning via Optuna é opcional.

---

## 2. Uso pretendido

- **Caso de uso primário:** classificação supervisionada em dados tabulares (binária e multiclasse), especialmente em cenários de baixo volume de dados ou sem disponibilidade de infraestrutura para tuning.
- **Casos de uso fora de escopo:** dados não-IID, séries temporais, dados de imagem, dados textuais brutos não tabelados, regressão (suportada pelo modelo mas não avaliada neste projeto), datasets com p > 500 features (degradação esperada).
- **Usuários pretendidos:** pesquisadores e praticantes de ML com problemas de classificação tabular, especialmente onde a simplicidade de uso e a ausência de tuning são prioridades.
- **Faixa de n suportada:** até milhões de amostras com Query-Aware Scalable Softmax; melhor desempenho empírico observado com n < 10.000.
- **Faixa de p suportada:** até centenas de features; features categóricas codificadas internamente.
- **Condições operacionais:** execução em CPU puro é suportada; GPU (NVIDIA com CUDA) recomendada para datasets com n > 10.000, reduzindo tempo de inferência.

---

## 3. Fatores observados

Dimensões avaliadas sobre os 30 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** no regime Small (3 datasets, n < 1.000), o desempenho do TabICL ficou abaixo dos GBDTs com tuning (AUC 0,806 vs. ~0,860 dos Tuned), em parte pela falha no dataset *anneal*. No regime Medium (17 datasets), o TabICL alcançou o melhor AUC médio entre todos os sistemas (0,943). No regime Large (10 datasets), permaneceu competitivo (0,863), próximo ao AutoGluon Extreme (0,864), com alta variabilidade.
- **Número de classes:** em problemas multiclasse, o TabICL obteve AUC de 0,948, liderando entre todos os sistemas. Em problemas binários, o desempenho é mais homogêneo (0,856), com AutoGluon Extreme ligeiramente à frente (0,859).
- **Proporção entre features categóricas e numéricas:** em datasets predominantemente numéricos, o TabICL obteve AUC de 0,922, superior a todos os GBDTs. Em datasets com maioria categórica, o CatBoost Tuned liderou (0,869 vs. 0,857 do TabICL), sugerindo vantagem dos GBDTs na codificação nativa de categorias.
- **Presença de valores ausentes:** em datasets sem NaN (20 dos 30), o TabICL alcançou AUC de 0,922, empatando com o AutoGluon Extreme. Em datasets com NaN (10 dos 30), o desempenho caiu para 0,848, abaixo dos GBDTs com tuning (~0,860–0,872), indicando que o tratamento nativo de valores ausentes pelo TabICL pode ser menos eficaz que a imputação explícita por mediana nesses casos.

---

## 4. Métricas alcançadas

Resultados agregados nos 30 datasets do TabArena-v0.1 (conjunto de teste, seed=42). IC de 95% calculado via bootstrap com 1.000 reamostragens sobre os 29 datasets completos (1 falha imputada).

| Métrica | Média | DP | IC 95% (bootstrap) | Rank médio (9 sistemas) |
|---|---|---|---|---|
| AUC OvO | 0,9056 | 0,1154 | [0,861; 0,944] | 2,57 (2º/9) |
| Accuracy | 0,9144 | 0,1102 | [0,872; 0,952] | 3,08 (2º/9) |
| G-Mean | 0,6786 | 0,3607 | [0,554; 0,804] | 3,73 (1º/9) |
| Cross-Entropy | 0,2071 | 0,2450 | [0,125; 0,300] | 2,05 (1º/9) |
| Tempo total (s) | 48,5 | 164,2 | [8,4; 125,7] | 3,60 (3º/9) |

### Resultados por subgrupo

| Dimensão | Subgrupo | AUC OvO (média ± DP) |
|---|---|---|
| Tamanho | Small (n < 1.000, 3 datasets) | 0,806 ± 0,033 |
| Tamanho | Medium (1.000 ≤ n ≤ 10.000, 17 datasets) | 0,943 ± 0,074 |
| Tamanho | Large (n > 10.000, 10 datasets) | 0,863 ± 0,162 |
| Tipo | Binário (18 datasets) | 0,856 ± 0,124 |
| Tipo | Multiclasse (12 datasets) | 0,948 ± 0,091 |
| Features | Maioria numérica (cat_ratio ≤ 0,5) | 0,922 ± 0,102 |
| Features | Maioria categórica (cat_ratio > 0,5) | 0,857 ± 0,141 |
| Valores ausentes | Com NaN (10 datasets) | 0,848 ± 0,152 |
| Valores ausentes | Sem NaN (20 datasets) | 0,922 ± 0,098 |

---

## 5. Dados de avaliação

- **Origem:** 30 datasets do benchmark TabArena-v0.1 (NeurIPS 2025), carregados via OpenML.
- **Distribuição por regime:** 3 pequenos (n < 1.000) + 17 médios (1.000 ≤ n ≤ 10.000) + 10 grandes (n > 10.000). A lista curada da disciplina continha apenas 3 datasets do regime Small, resultando em distribuição assimétrica em relação ao esquema 10/10/10 previsto no enunciado.
- **Estratégia de split:** 70/30 estratificado por classe, `random_state=42`.
- **Pré-processamento para GBDTs:** `OrdinalEncoder` para features categóricas e `SimpleImputer` (mediana) para valores ausentes, aplicados após o split (sem vazamento). O TabICL realiza pré-processamento interno (normalização quantílica, clipping, codificação), sem imputação prévia.
- **Lista completa dos datasets:** ver Apêndice A do relatório final (nome, OpenML task ID, n, n\_features, n\_classes, regime).

---

## 6. Dados de treino e pré-treino

- **Tipo:** foundation model pré-treinado. Não há atualização de pesos durante `fit()` — os dados de treino são armazenados como contexto e utilizados no mecanismo de atenção durante `predict()`.
- **Origem dos dados de pré-treino:** dados tabulares sintéticos gerados por engine proprietário da equipe Soda/Inria, cobrindo distribuições diversas (lineares, não-lineares, categóricas, valores ausentes) para maximizar a generalização. O engine não é público; detalhes estão em Schlegel et al. (2026), arXiv:2602.11139.
- **Origem dos dados de treino direto:** durante `fit()`, os dados fornecidos são armazenados em memória como contexto de inferência. Nenhum gradiente é calculado.
- **Possíveis vieses herdados do pré-treino:** o modelo pode não capturar padrões de domínios com estruturas muito específicas (ex.: dados médicos com correlações complexas não presentes no corpus sintético). Datasets com características muito distantes das distribuições de pré-treino podem apresentar desempenho inferior.

---

## 7. Análise quantitativa

- **Posição no ranking médio (9 sistemas avaliados):** 2º/9 em AUC-OVO (rank 2,57), 2º/9 em ACC (rank 3,08), **1º/9** em G-Mean (rank 3,73) e **1º/9** em Cross-Entropy (rank 2,05). O sistema que classificou à frente em AUC foi o AutoGluon Extreme (rank 2,35).

- **Friedman + Nemenyi (α = 0,05):** o teste de Friedman detectou diferença estatisticamente significativa entre os 9 sistemas. O diagrama de diferença crítica (CD diagram, ver relatório) mostra o TabICL no mesmo agrupamento de equivalência estatística que o AutoGluon Extreme em AUC-OVO, enquanto os GBDTs com hiperparâmetros padrão (TD) formam um grupo com performance inferior.

- **Bayesian signed-rank com ROPE = 0,01 em AUC (TabICL vs LightGBM Tuned):** P(equivalente) ≈ 82%, P(TabICL vence) ≈ 17%, P(LightGBM Tuned vence) < 1%. Resultado indica que os modelos são praticamente equivalentes, com leve favorecimento ao TabICL. Para demais pares, ver tabelas `bayesian_rope_*.csv` em `cluster_apuana/resultados_estatisticos/`.

- **Comparação com AutoGluon:**

  | Comparador | ΔAUC (TabICL − comparador) | Speedup (tempo) |
  |---|---|---|
  | AutoGluon Default | +0,017 | ~29× mais rápido |
  | AutoGluon Extreme (4h) | −0,005 | ~38× mais rápido |

  O TabICL supera o AutoGluon Default em AUC e é ~29× mais rápido. Frente ao AutoGluon Extreme, perde 0,005 de AUC mas é ~38× mais rápido, operando em média em 48 s por dataset contra ~1.831 s do AutoGluon Extreme.

- **Análise por regime:** vantagem mais expressiva no regime Medium (AUC 0,943 vs. 0,945 do AutoGluon Extreme — diferença de 0,002). No regime Small, o TabICL ficou abaixo dos GBDTs Tuned (AUC 0,806 vs. ~0,860), possivelmente influenciado pela falha no dataset *anneal*. No regime Large, é competitivo com o AutoGluon Extreme (0,863 vs. 0,864) e supera todos os GBDTs.

---

## 8. Considerações éticas

- **Riscos de uso indevido:** como foundation model, o TabICL v2 pode ser aplicado a domínios sensíveis (crédito, saúde, justiça) sem validação adequada. As predições são opacas — o modelo não oferece interpretabilidade nativa.
- **Fairness por classe:** o modelo não incorpora mecanismos explícitos de equidade. Classes minoritárias podem apresentar recall inferior em datasets desbalanceados, como evidenciado pelos valores de G-Mean inferiores à acurácia em vários datasets. Recomenda-se análise de G-Mean por classe antes de qualquer aplicação em produção.
- **Dependência de licença:** Apache 2.0 para código e pesos — uso comercial e acadêmico permitidos sem restrições jurídicas.
- **Impacto ambiental:** o pré-treino consumiu recursos computacionais significativos (cluster GPU, não quantificado publicamente). A inferência é relativamente eficiente, especialmente em CPU para datasets de pequeno e médio porte. O tempo médio de execução nos experimentos foi de 48,5 s por dataset.
- **Recomendações de auditoria:** comparar predições com um baseline interpretável (ex.: EBM — Explainable Boosting Machine) antes de deploy em domínios sensíveis. Utilizar SHAP externo para atribuição de importância de features, dado que o TabICL não fornece importâncias nativamente.

---

## 9. Avisos e recomendações

- **Quando usar este modelo:**
  - Datasets de regime Small e Medium (n < 10.000) sem disponibilidade de infraestrutura para tuning extensivo
  - Prototipagem rápida com interface `fit/predict` sem configuração manual
  - Problemas multiclasse com sobreposição de classes, onde o ICL explora vizinhanças contextuais
  - Datasets com valores ausentes (tratamento nativo, sem necessidade de imputação prévia)
  - Quando a relação custo-benefício em tempo de execução é crítica

- **Quando NÃO usar este modelo:**
  - Datasets com n > 100K amostras ou p > 500 features sem GPU com alta VRAM (A100/H100), pelo risco de Out-Of-Memory decorrente da complexidade O(n² · p)
  - Ambientes com restrição severa de latência de inferência (GBDTs são ordens de magnitude mais rápidos)
  - Quando interpretabilidade é requisito regulatório ou operacional (usar EBM ou LightGBM + SHAP)
  - Datasets com alta proporção de features categóricas e alta cardinalidade, onde o CatBoost tem vantagem

- **Observação sobre falhas de execução:** durante os experimentos, o TabICL falhou em 1 dos 30 datasets (`anneal`, n=898, p=39, 5 classes, regime Small), com resultado imputado pela média do regime. Os demais 29 datasets foram concluídos com êxito.

- **Alternativas recomendadas:**
  - Para n > 50K: LightGBM TD ou CatBoost TD (eficiência de memória e velocidade)
  - Para alta cardinalidade categórica: CatBoost TD ou CatBoost Tuned
  - Para AutoML sem restrição de tempo: AutoGluon Extreme (4h)
  - Para interpretabilidade: EBM (Explainable Boosting Machine)
  - Para equilíbrio velocidade/performance sem tuning: AutoGluon Default

---

## 10. Reprodutibilidade

- **Ambiente:** Python 3.11.15; dependências fixadas em `pyproject.toml` e `uv.lock` (gerenciador `uv`).
- **Hardware utilizado nos experimentos principais:** cluster HPC Apuana (SLURM), GPU NVIDIA, com scripts em `experiments/`. Execução local de testes: Apple Silicon (aarch64), CPU only.
- **Tempo total de execução:** aproximadamente 4 horas por membro do grupo para o AutoGluon Extreme (execução distribuída); experimentos com TabICL e GBDTs completados em ~2 horas no cluster.
- **Seed utilizada:** 42, fixa em split, tuning (Optuna TPESampler) e backend determinístico do PyTorch.
- **Hash do commit:** `56b6c935ef274cec5415811ee5d252ba2b62a021`
- **Comandos para reproduzir:**
  ```bash
  uv sync
  uv run pytest tests/test_pipeline.py -v               # smoke test
  uv run python experiments/run_experiment.py            # experimento completo
  uv run python experiments/gerar_graficos_e_tabelas.py # gráficos e tabelas LaTeX
  ```
- **Versões-chave:**
  - `tabicl==2.1.1`
  - `pytabkit==1.7.3`
  - `scikit-learn==1.7.2`
  - `torch==2.7.1`
  - `optuna==4.8.0`
  - `autogluon.tabular==1.4.0`
  - `autorank==1.1.1`
  - `baycomp==1.0.3`

---

## 11. Referências

- Schlegel, V., Zhu, Y., Leite, R., Varoquaux, G. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model*. arXiv:2602.11139.
- Schlegel, V., Leite, R., Varoquaux, G. (2025). *TabICL: A Tabular Foundation Model for In-Context Learning*. ICML 2025.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. Proceedings of FAT* 2019.
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple datasets*. JMLR 7, pp. 1–30.
- Benavoli, A., Corani, G., Demšar, J., Zaffalon, M. (2017). *Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis*. JMLR 18, pp. 1–36.
- Hölzmuller, D. et al. (2024). *Better default hyperparameters for tabular models (pytabkit)*. NeurIPS 2024.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
- TabICL GitHub: https://github.com/soda-inria/tabicl
