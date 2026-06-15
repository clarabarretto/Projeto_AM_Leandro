# TabICL v2 — Avaliação em Benchmark TabArena-v0.1

**Projeto Final — Disciplina de Aprendizagem de Máquina (2026-1)**  
**Prof. Leandro Almeida — CIn/UFPE**

| Integrante | E-mail |
|---|---|
| João Pedro Miranda da Silva | jpms5@cin.ufpe.br |
| Maria Clara Falcão Guerra Barretto | mcfgb@cin.ufpe.br |
| Vinicius Limeira Valença | vlv3@cin.ufpe.br |

---

## Sobre o projeto

Avaliação empírica do modelo fundacional **TabICL v2** (v2.1.1, Soda/Inria) em classificação supervisionada tabular. O modelo aplica *In-Context Learning* com arquitetura Transformer: os dados de treino são fornecidos como contexto de atenção na inferência, sem atualização de pesos.

O estudo cobre **30 datasets** do benchmark TabArena-v0.1 (NeurIPS 2025) e compara o TabICL v2 com 8 sistemas:

- **Baselines padrão:** LightGBM TD, XGBoost TD, CatBoost TD
- **Baselines otimizados:** LightGBM Tuned, XGBoost Tuned, CatBoost Tuned (Optuna, 20 trials)
- **AutoML:** AutoGluon Default e AutoGluon Extreme 4h

A validação estatística usa teste de Friedman–Nemenyi (frequentista) e Bayesian Signed-Rank com ROPE = 0,01 (biblioteca *baycomp*).

---

## Resultados principais

| Modelo | AUC-OVO | Rank médio (9 sistemas) | Tempo médio (s) |
|---|:---:|:---:|:---:|
| **TabICL v2** | **0,906** | **2,57** | **48,5** |
| AutoGluon Extreme 4h | 0,911 | 2,35 | ~1.831 |
| LightGBM Tuned | 0,899 | 3,98 | ~240 |
| CatBoost Tuned | 0,898 | 4,12 | ~190 |
| AutoGluon Default | 0,889 | 4,10 | ~120 |

**Teste bayesiano TabICL vs AutoGluon Extreme 4h:** P(equivalente) = 99,6% — indistinguíveis dentro de 1% de AUC, com o TabICL sendo **~38× mais rápido**.

---

## Quickstart

```bash
# 1. Instalar dependências (Python 3.11 obrigatório)
uv sync

# 2. Smoke test — deve passar 7/7
uv run pytest tests/ -v

# 3. Rodar experimento completo (GPU recomendada para datasets Large)
uv run python experiments/run_experiment.py

# 4. Gerar figuras e tabelas LaTeX
uv run python experiments/gerar_graficos_e_tabelas.py
```

Via Docker (reprodução completa em qualquer máquina):

```bash
docker build -t tabicl-eval .
docker run --gpus all tabicl-eval
```

---

## Estrutura do repositório

```
.
├── reports/                        # Entregáveis
│   ├── relatorio_final.tex         # Relatório completo em LaTeX
│   ├── model_card.md               # Model card do TabICL v2 (11 seções)
│   └── templates/                  # Templates fornecidos pelo professor
│       ├── rubrica.pdf
│       ├── relatorio-template.pdf
│       └── slides-template.pdf
│
├── results/                        # Outputs dos experimentos
│   ├── plots/                      # Figuras (CD diagrams, Bayesian simplex, scatter)
│   ├── tables/                     # Tabelas LaTeX (tabelas_resultados.tex)
│   ├── data/                       # CSVs — resultados por dataset × modelo
│   │   ├── final_run_results_v2.csv
│   │   └── dataset_metadata.csv
│   └── statistical_analysis/       # Testes de Friedman e ROPE bayesiana (CSV por métrica)
│
├── src/                            # Código do pipeline
│   ├── models/                     # TabICL v2, LightGBM, XGBoost, CatBoost, AutoGluon
│   ├── pipeline/                   # Split, tuning Optuna, avaliação, análise estatística
│   └── reports/                    # Geração de tabelas-resumo
│
├── data/                           # Carregador TabArena-v0.1 via OpenML (30 datasets)
├── notebooks/                      # EDA e demonstrações (01_eda … 04_demo_stats)
├── tests/                          # Smoke test end-to-end (7/7 passing)
│
├── experiments/                    # Scripts para rodar e analisar os experimentos
│   ├── run_experiment.py           # Pipeline completo: split → tune → eval → stats
│   ├── gerar_graficos_e_tabelas.py # Gera todos os plots e tabelas LaTeX
│   ├── gerar_optuna_exemplos.py    # Curvas de convergência do Optuna
│   └── cluster/
│       └── job_apuana.slurm        # Job SLURM para cluster HPC (CIn-Apuana)
│
├── docs/                           # Documentação
│   ├── INFRAESTRUTURA.md           # Setup detalhado e dependências
│   ├── CLUSTER.md                  # Guia de comandos SLURM
│   ├── datasets_selecionados.csv   # Metadados dos 30 datasets escolhidos
│   ├── tabarena_datasets.csv       # Lista oficial TabArena-v0.1
│   └── enunciado.pdf               # Enunciado da disciplina
│
├── Dockerfile                      # Reprodução em container
├── pyproject.toml                  # Dependências (Python 3.11, uv)
└── uv.lock                         # Lockfile para reprodução exata
```

---

## Reprodutibilidade

| Aspecto | Detalhe |
|---|---|
| Seed | `random_state=42` em split, Optuna (TPESampler) e PyTorch (`deterministic=True`) |
| Ambiente | Python 3.11, dependências travadas em `pyproject.toml` + `uv.lock` |
| Container | `Dockerfile` na raiz (Python 3.11-slim, CUDA opcional) |
| Cluster | Cluster HPC Apuana (CIn-UFPE), SLURM, GPU NVIDIA RTX 3090 |
| Commit dos experimentos | `56b6c935ef274cec5415811ee5d252ba2b62a021` |

---

## Referências

- Schlegel, V., Zhu, Y., Leite, R., Varoquaux, G. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model*. arXiv:2602.11139.
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple datasets*. JMLR 7, 1–30.
- Benavoli, A., Corani, G., Demšar, J., Zaffalon, M. (2017). *Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis*. JMLR 18, 1–36.
- Hölzmuller, D. et al. (2024). *Better default hyperparameters for tabular models (pytabkit)*. NeurIPS 2024.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
- TabICL GitHub: https://github.com/soda-inria/tabicl
