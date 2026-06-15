# Infraestrutura e ferramentas, guia prático

Este documento é um tutorial passo a passo das ferramentas que o projeto usa. Comece pelo topo se você nunca configurou um ambiente Python para um projeto científico; pule para a seção específica se você já tem o básico.

Sumário:

1. [O que você precisa instalado na máquina](#1-o-que-voce-precisa-instalado-na-maquina)
2. [Por que ambiente virtual e qual usar](#2-por-que-ambiente-virtual-e-qual-usar)
3. [Caminho A, com `uv` (recomendado)](#3-caminho-a-com-uv-recomendado)
4. [Caminho B, com `pip` e `venv`](#4-caminho-b-com-pip-e-venv)
5. [Caminho C, com `poetry`](#5-caminho-c-com-poetry)
6. [Como o `pyproject.toml` funciona](#6-como-o-pyprojecttoml-funciona)
7. [Rodar o smoke test](#7-rodar-o-smoke-test)
8. [Rodar os notebooks](#8-rodar-os-notebooks)
9. [Containerizar com Docker (opcional)](#9-containerizar-com-docker-opcional)
10. [Reprodutibilidade e seeds](#10-reprodutibilidade-e-seeds)
11. [Troubleshooting comum](#11-troubleshooting-comum)

---

## 1. O que você precisa instalado na máquina

| Ferramenta | Versão recomendada | Como verificar |
|---|---|---|
| Python | 3.11 ou 3.12 | `python --version` |
| `pip` | qualquer recente | `pip --version` |
| `git` | qualquer recente | `git --version` |
| `uv` (opcional, recomendado) | 0.4 ou superior | `uv --version` |
| Docker (opcional) | 24 ou superior | `docker --version` |

Se Python não estiver instalado:

- **Windows**: baixar do site oficial (`python.org/downloads`) e marcar a opção "Add Python to PATH".
- **macOS**: `brew install python@3.11` (Homebrew) ou baixar do site oficial.
- **Linux**: `sudo apt install python3.11 python3.11-venv` (Ubuntu/Debian) ou equivalente.

---

## 2. Por que ambiente virtual e qual usar

Um ambiente virtual é um diretório isolado com sua própria cópia do Python e suas próprias bibliotecas. Sem isolamento, instalar uma versão específica de `numpy` para este projeto poderia quebrar outro projeto que precisa de versão diferente.

Três opções principais:

- **`uv`** (recomendado): rápido, moderno (2024-2025), gerencia tudo via `pyproject.toml`.
- **`pip` + `venv`**: tradicional, está em qualquer instalação Python.
- **`poetry`**: alternativa madura, popular desde 2018.

Os três caminhos abaixo levam ao mesmo resultado, escolha um.

---

## 3. Caminho A, com `uv` (recomendado)

Instalar o `uv` (uma única vez):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No diretório do template:

```bash
# clonar o template
git clone <url-do-template>
cd projeto-final-AM-template

# o uv lê pyproject.toml, cria .venv e instala tudo:
uv sync

# rodar qualquer comando dentro do ambiente:
uv run pytest tests/test_pipeline.py -v
uv run python -m src.pipeline.run_all --seed 42
uv run jupyter lab
```

O arquivo `uv.lock` é gerado automaticamente, fixando versões exatas. Comite-o junto com o código para garantir reprodutibilidade total.

---

## 4. Caminho B, com `pip` e `venv`

```bash
cd projeto-final-AM-template

# criar o ambiente virtual
python -m venv .venv

# ativar
# Windows (cmd.exe)
.venv\Scripts\activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# instalar o projeto e suas dependências
pip install --upgrade pip
pip install -e .

# rodar
pytest tests/test_pipeline.py -v
python -m src.pipeline.run_all --seed 42
jupyter lab
```

Para sair do ambiente: `deactivate`.

---

## 5. Caminho C, com `poetry`

Instalar `poetry` (uma única vez): `pipx install poetry` ou conforme `python-poetry.org`.

```bash
cd projeto-final-AM-template
poetry install
poetry run pytest tests/test_pipeline.py -v
poetry run python -m src.pipeline.run_all --seed 42
poetry shell  # entra no ambiente
```

Observação: o `pyproject.toml` deste template segue o padrão PEP 621 (compatível com `uv` e `pip install -e .`). Para usar com `poetry`, talvez seja necessário adaptar a seção `[build-system]`.

---

## 6. Como o `pyproject.toml` funciona

É um arquivo de texto que descreve:

1. **Metadados** do projeto (nome, versão, autores).
2. **Dependências** obrigatórias (`dependencies`).
3. **Dependências opcionais** (`[project.optional-dependencies]`), por exemplo um grupo `dev` com `pytest`, `ruff`, `jupyter`.
4. **Versão mínima do Python** (`requires-python`).
5. **Configuração de ferramentas** (pytest, ruff, etc.) na seção `[tool.*]`.

Exemplo simplificado:

```toml
[project]
name = "meu-projeto"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=1.26,<3.0",
    "pandas>=2.2,<4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0,<10.0"]
```

Os símbolos `>=` e `<` definem ranges de versão compatíveis. Ferramentas como `uv`, `pip` e `poetry` resolvem essas restrições e instalam as versões adequadas.

---

## 7. Rodar o smoke test

O smoke test (`tests/test_pipeline.py`) valida que cada baseline (LightGBM, XGBoost, CatBoost) executa de ponta a ponta no dataset `breast_cancer`. Roda em menos de 30 segundos.

```bash
# com uv
uv run pytest tests/test_pipeline.py -v

# com venv ativo
pytest tests/test_pipeline.py -v
```

Saída esperada: `7 passed`. Se algum teste falhar, normalmente é por dependência faltando ou versão incompatível; ver seção 11.

---

## 8. Rodar os notebooks

```bash
# com uv
uv run jupyter lab

# com venv ativo
jupyter lab
```

Abra `notebooks/01_eda.ipynb` para EDA inicial. A ordem recomendada: 01 (EDA) -> 02 (baselines) -> 03 (modelo do grupo) -> 04 (estatística e regime).

---

## 9. Containerizar com Docker (opcional)

Se você quer rodar o projeto exatamente igual em qualquer máquina (Windows, Linux, servidor), use Docker. Tudo o que precisa já está no `Dockerfile`:

```bash
# construir a imagem (uma vez)
docker build -t projeto-final-am .

# rodar o smoke test dentro do container
docker run --rm projeto-final-am

# abrir um shell para usar interativamente
docker run --rm -it -v "$(pwd)":/workspace projeto-final-am bash
```

A imagem fica isolada do seu sistema; você pode deletar com `docker image rm projeto-final-am` quando não precisar mais.

---

## 10. Reprodutibilidade e seeds

Reprodutibilidade significa que rodar o mesmo código, com os mesmos dados, gera o mesmo resultado. Para isso:

1. **Seed fixa em todos os pontos aleatórios.** O template usa `seed=42` por default em `split`, `tune` e nos modelos.
2. **Versões fixadas.** O `pyproject.toml` define ranges, e o `uv.lock` (ou `requirements.txt` gerado por `pip freeze`) fixa versões exatas.
3. **Dados imutáveis.** O OpenML versiona os datasets via `task_id`. Use sempre o mesmo `task_id` para garantir o mesmo dado.
4. **Hash do commit.** Inclua no relatório o `git rev-parse HEAD` da versão usada nos experimentos.

Cuidado: alguns frameworks (PyTorch CUDA, GPU em geral) introduzem não-determinismo mesmo com seed; nesses casos, documente isso como limitação.

---

## 11. Troubleshooting comum

**`pip install -e .` falha com "Acesso negado".**
- Você está tentando instalar no Python global. Crie um `venv` (seção 4) e ative antes de instalar.

**`ModuleNotFoundError: No module named 'autogluon'`.**
- Falta instalar `autogluon.tabular`. Rode `uv sync` ou `pip install autogluon.tabular`. Em ambientes leves, use os exemplos comentados em `src/models/group_model.py`.

**`pytest` reclama de import.**
- Verifique que você está no diretório raiz do projeto (`projeto-final-AM-template/`) e que o `venv` está ativo.

**Conflito de versões entre `numpy` e `pandas` ou `scikit-learn`.**
- Apague `.venv/` e recrie: `rm -rf .venv && uv sync` (ou equivalente em pip). Versões muito antigas instaladas globalmente podem interferir.

**MiKTeX no Windows não compila o `.tex` (caso esteja recompilando o documento).**
- Rode `pdflatex` duas vezes para resolver referências do `hyperref`.
- Se aparecer "running on unsupported version of Windows", é apenas aviso, o PDF é gerado normalmente.

**TabPFN ou TabICL pedem GPU e você só tem CPU.**
- TabICL roda em CPU sem problemas. TabPFN roda em CPU para datasets pequenos (< 1000 amostras); para maiores, use Google Colab ou outro ambiente com GPU.

**`uv sync` é muito lento na primeira vez.**
- Normal, a primeira execução baixa todas as dependências (centenas de MB). As próximas são instantâneas pelo cache.

---

## Referências

- Python venv: https://docs.python.org/3/library/venv.html
- pyproject.toml (PEP 621): https://peps.python.org/pep-0621/
- uv: https://docs.astral.sh/uv/
- poetry: https://python-poetry.org/
- Docker: https://docs.docker.com/get-started/
- Optuna: https://optuna.org/
- OpenML: https://www.openml.org/
- Model cards: https://arxiv.org/abs/1810.03993
