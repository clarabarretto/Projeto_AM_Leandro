"""Executa o pipeline completo: para cada dataset, treina e avalia todos os modelos.

Uso:
    python -m src.pipeline.run_all --seed 42 --output results/raw.csv
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task
from src.models.baselines import BASELINE_FACTORIES
from src.models.group_model import build_group_model
from src.pipeline.evaluate import fit_predict_evaluate
from src.pipeline.split import stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw.csv"),
        help="caminho do CSV de saida",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="opcional: lista de task IDs do OpenML; se omitido, usa RECOMMENDED_TASK_IDS",
    )
    parser.add_argument(
        "--include-group-model",
        action="store_true",
        help="se passado, inclui o modelo do grupo (build_group_model)",
    )
    return parser.parse_args()


def _preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Imputa NaN e converte categoricas para numerico."""
    # Separar colunas numericas e categoricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Codificar categoricas como inteiros
    for col in cat_cols:
        le = LabelEncoder()
        # Combinar treino e teste para fit, tratando NaN
        combined = pd.concat([X_tr[col], X_te[col]], axis=0).astype(str)
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))

    # Imputar NaN em numericas com mediana
    if X_tr.isna().any().any() or X_te.isna().any().any():
        imputer = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(
            imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
        )
        X_te = pd.DataFrame(
            imputer.transform(X_te), columns=X_te.columns, index=X_te.index
        )

    return X_tr, X_te


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids else RECOMMENDED_TASK_IDS

    # Resume support: load existing results and skip completed task_ids
    rows: list[dict] = []
    done_task_ids: set[int] = set()
    if args.output.exists():
        existing = pd.read_csv(args.output)
        rows = existing.to_dict("records")
        done_task_ids = set(existing["task_id"].unique())
        print(f"Resumindo: {len(done_task_ids)} datasets já completados, pulando...")

    n_total = len(task_ids)

    for i, task_id in enumerate(task_ids, 1):
        if task_id in done_task_ids:
            print(f"\n=== [{i}/{n_total}] task_id={task_id}: JÁ COMPLETADO, pulando ===")
            continue

        try:
            ds = load_task(task_id)
            print(f"\n=== [{i}/{n_total}] Dataset: {ds.name} (task_id={task_id}, "
                  f"n={ds.n_samples}, p={ds.n_features}, classes={ds.n_classes}, "
                  f"missing={ds.has_missing}) ===")
        except Exception as e:
            print(f"\n=== [{i}/{n_total}] ERRO ao carregar task_id={task_id}: {e} ===")
            continue

        # Encode target as integers if needed
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(ds.y)

        X_train, X_test, y_train, y_test = stratified_split(ds.X, y_encoded, seed=args.seed)

        # Pre-process: impute NaN, encode categoricals
        X_train_clean, X_test_clean = _preprocess(X_train, X_test)

        factories: dict[str, callable] = dict(BASELINE_FACTORIES)
        if args.include_group_model:
            factories["group_model"] = build_group_model

        for model_name, factory in factories.items():
            try:
                estimator = factory(args.seed)
                metrics = fit_predict_evaluate(
                    estimator, X_train_clean, y_train, X_test_clean, y_test
                )
                row = {"task_id": task_id, "dataset": ds.name, "model": model_name}
                row.update(metrics.to_dict())
                rows.append(row)
                print(
                    f"  {model_name}: AUC={metrics.auc_ovo:.4f}, "
                    f"ACC={metrics.accuracy:.4f}, "
                    f"time={metrics.fit_time_s + metrics.predict_time_s:.1f}s"
                )
            except Exception as e:
                print(f"  {model_name}: ERRO - {e}")
                traceback.print_exc()
                # Record NaN metrics for failed models
                rows.append({
                    "task_id": task_id,
                    "dataset": ds.name,
                    "model": model_name,
                    "auc_ovo": float("nan"),
                    "accuracy": float("nan"),
                    "g_mean": float("nan"),
                    "cross_entropy": float("nan"),
                    "fit_time_s": float("nan"),
                    "predict_time_s": float("nan"),
                    "total_time_s": float("nan"),
                    "train_auc_ovo": float("nan"),
                    "train_accuracy": float("nan"),
                    "train_g_mean": float("nan"),
                    "train_cross_entropy": float("nan"),
                })

        # Save incrementally after each dataset
        try:
            pd.DataFrame(rows).to_csv(args.output, index=False)
        except PermissionError:
            print(f"\n[AVISO] Nao foi possivel salvar no CSV (o arquivo esta aberto no Excel?). "
                  f"Feche o arquivo para os proximos datasets serem salvos corretamente.")
        except Exception as e:
            print(f"\n[AVISO] Erro ao salvar o arquivo CSV: {e}")

    print(f"\nResultados gravados em {args.output} ({len(rows)} linhas, {n_total} datasets)")


if __name__ == "__main__":
    main()

