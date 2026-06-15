import os
import sys
import time
import signal
import warnings
import logging
import multiprocessing
import numpy as np
import pandas as pd
import random
import openml
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch

# --- CONFIGURAÇÕES GERAIS ---
SEED = 42
N_TRIALS = 20  # Reduzido de 50 para 20 para caber no prazo
AG_EXTREME_TIME_LIMIT = 14400  # 4 horas para a rodada final definitiva
AG_DEFAULT_TIME_LIMIT = 1800   # 30 minutos cravados para o modelo Base

# Determinismo Global
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("run_cluster_v2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

num_cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
NUM_CPUS = int(num_cpus_env) if num_cpus_env else multiprocessing.cpu_count()
logging.info(f"Detectado limite de CPUs: {NUM_CPUS}")

os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_CPUS)

RESULTS_FILE = "results/data/final_run_results_v2.csv"
os.makedirs("results/data", exist_ok=True)

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Tempo limite de segurança atingido!")

# --- DATASETS OFICIAIS ---
DATASETS = [
    {'tid': 1464, 'name': 'blood-transfusion-service-center', 'regime': 'small'},
    {'tid': 37, 'name': 'diabetes', 'regime': 'small'},
    {'tid': 2, 'name': 'anneal', 'regime': 'small'},
    {'tid': 168757, 'name': 'credit-g', 'regime': 'medium'},
    {'tid': 359956, 'name': 'qsar-biodeg', 'regime': 'medium'},
    {'tid': 2077, 'name': 'baseball', 'regime': 'medium'},
    {'tid': 2073, 'name': 'yeast', 'regime': 'medium'},
    {'tid': 45, 'name': 'splice', 'regime': 'medium'},
    {'tid': 359967, 'name': 'Bioresponse', 'regime': 'medium'},
    {'tid': 3011, 'name': 'hypothyroid', 'regime': 'medium'},
    {'tid': 3892, 'name': 'hiva_agnostic', 'regime': 'medium'},
    {'tid': 43, 'name': 'spambase', 'regime': 'medium'},
    {'tid': 58, 'name': 'waveform-5000', 'regime': 'medium'},
    {'tid': 359968, 'name': 'churn', 'regime': 'medium'},
    {'tid': 30, 'name': 'page-blocks', 'regime': 'medium'},
    {'tid': 28, 'name': 'optdigits', 'regime': 'medium'},
    {'tid': 2074, 'name': 'satimage', 'regime': 'medium'},
    {'tid': 3481, 'name': 'isolet', 'regime': 'medium'},
    {'tid': 24, 'name': 'mushroom', 'regime': 'medium'},
    {'tid': 3510, 'name': 'JapaneseVowels', 'regime': 'medium'},
    {'tid': 32, 'name': 'pendigits', 'regime': 'large'},
    {'tid': 26, 'name': 'nursery', 'regime': 'large'},
    {'tid': 6, 'name': 'letter', 'regime': 'large'},
    {'tid': 3688, 'name': 'houses', 'regime': 'large'},
    {'tid': 359979, 'name': 'Amazon_employee_access', 'regime': 'large'},
    {'tid': 3945, 'name': 'KDDCup09_appetency', 'regime': 'large'},
    {'tid': 168868, 'name': 'APSFailure', 'regime': 'large'},
    {'tid': 361329, 'name': 'KDD98', 'regime': 'large'},
    {'tid': 211986, 'name': 'Diabetes130US', 'regime': 'large'},
    {'tid': 360113, 'name': 'porto-seguro', 'regime': 'large'},
]

# --- PRE-PROCESSAMENTO CORRIGIDO (SEM LEAKAGE) ---
def build_preprocessor(cat_indicator, X_columns):
    cat_cols = [col for i, col in enumerate(X_columns) if cat_indicator[i]]
    num_cols = [col for i, col in enumerate(X_columns) if not cat_indicator[i]]
    
    transformers = []
    if cat_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
        
    return ColumnTransformer(transformers, remainder="passthrough")

def compute_auc(y_true, y_proba, n_classes):
    try:
        classes = np.arange(n_classes)
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovo", labels=classes))
    except Exception as e: 
        logging.warning(f"AUC calculation failed: {e}")
        return np.nan

def g_mean_score(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if not mask.any(): continue
        recalls.append(float((y_pred[mask] == c).mean()))
    if not recalls: return 0.0
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0)))))

def compute_cross_entropy(y_true, y_proba, n_classes):
    try: return float(log_loss(y_true, y_proba, labels=np.arange(n_classes)))
    except: return np.nan

# --- BASELINES DEFAULT ---
def run_baseline(model_name, build_fn, X_train, y_train, X_test, y_test, n_classes, **kwargs):
    logging.info(f"   Iniciando {model_name}...")
    t0 = time.perf_counter()
    model = build_fn(**kwargs)
    
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    else:
        # For TabICL
        preds, probs = model(X_train, y_train, X_test)
        
    fit_predict_time = time.perf_counter() - t0
    acc = accuracy_score(y_test, preds)
    gmean = g_mean_score(y_test, preds)
    auc = compute_auc(y_test, probs, n_classes) if probs is not None else np.nan
    ce = compute_cross_entropy(y_test, probs, n_classes) if probs is not None else np.nan
    
    logging.info(f"  {model_name} | ACC={acc:.4f} | AUC={auc:.4f} | Tempo={fit_predict_time:.1f}s")
    return {
        "model": model_name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
        "G_Mean": round(gmean, 6), "CE": round(ce, 6), "total_time_s": round(fit_predict_time, 2)
    }

def build_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, verbose=-1)

def build_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=SEED, eval_metric='logloss', verbosity=0, tree_method='hist', device='cuda')

def build_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=SEED, verbose=0, task_type='GPU')

def build_tabicl():
    from tabicl import TabICLClassifier
    return TabICLClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=SEED)

# --- AUTOGLUON ---
def run_autogluon(preset, X_train, y_train, X_test, y_test, n_classes):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil
    
    name = "AutoGluon_Extreme" if preset == "best_quality" else "AutoGluon_Default"
    time_limit = AG_EXTREME_TIME_LIMIT if preset == "best_quality" else AG_DEFAULT_TIME_LIMIT
    alarm_limit = time_limit + 1800  # Margem de segurança de 30min
    
    logging.info(f"  Iniciando {name} (limite={time_limit//60}min)...")
    
    ag_path = tempfile.mkdtemp()
    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test)
    
    t0 = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(alarm_limit)
        
        ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
        ag_problem_type = "binary" if n_classes == 2 else "multiclass"
        
        # O Random_state global que forçamos acima no python ajuda o AG
        predictor = TabularPredictor(label="target", eval_metric=ag_metric, problem_type=ag_problem_type, path=ag_path, verbosity=0)
        predictor.fit(
            train_data=train_df, presets=preset, time_limit=time_limit,
            num_gpus=1, num_cpus=NUM_CPUS, ag_args_fit={"num_cpus": NUM_CPUS}
        )
        
        signal.alarm(0)
        
        y_pred = predictor.predict(test_df).values
        y_proba = predictor.predict_proba(test_df).values
        
        total_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, y_pred)
        gmean = g_mean_score(y_test, y_pred)
        auc = compute_auc(y_test, y_proba, n_classes)
        ce = compute_cross_entropy(y_test, y_proba, n_classes)
        
        logging.info(f" {name} | ACC={acc:.4f} | AUC={auc:.4f} | Tempo={total_time:.1f}s")
        shutil.rmtree(ag_path, ignore_errors=True)
        
        return {
            "model": name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
            "G_Mean": round(gmean, 6), "CE": round(ce, 6), "total_time_s": round(total_time, 2)
        }
    except Exception as e:
        signal.alarm(0)
        shutil.rmtree(ag_path, ignore_errors=True)
        raise e

# --- OPTUNA TUNING ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def objective_lgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': SEED, 'verbose': -1
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = cross_val_score(LGBMClassifier(**params), X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': SEED, 'eval_metric': 'logloss', 'verbosity': 0, 'tree_method': 'hist', 'device': 'cuda'
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = cross_val_score(XGBClassifier(**params), X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

def objective_cat(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'bootstrap_type': 'Bernoulli',
        'random_state': SEED, 'verbose': 0, 'task_type': 'GPU'
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    # CatBoost cross_val_score sometimes complains without explicit fit_params, but usually fine for simple datasets.
    scores = cross_val_score(CatBoostClassifier(**params), X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.mean()

def run_tuning(model_name, objective_func, model_class, X_train, y_train, X_test, y_test, n_classes):
    logging.info(f"  Tuning {model_name} (Trials={N_TRIALS})...")
    t0 = time.perf_counter()
        
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    # Limita o tuning total a 2 horas (7200s) para segurança
    study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=N_TRIALS, timeout=7200)
    
    best_params = study.best_params
    best_params['random_state'] = SEED
    if "LightGBM" in model_name: best_params['verbose'] = -1
    elif "XGBoost" in model_name: 
        best_params.update({'eval_metric': 'logloss', 'verbosity': 0, 'tree_method': 'hist', 'device': 'cuda'})
    elif "CatBoost" in model_name: 
        best_params.update({'verbose': 0, 'task_type': 'GPU', 'bootstrap_type': 'Bernoulli'})
    
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
    
   
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        
    acc = accuracy_score(y_test, y_pred)
    gmean = g_mean_score(y_test, y_pred)
    auc = compute_auc(y_test, y_proba, n_classes) if y_proba is not None else np.nan
    ce = compute_cross_entropy(y_test, y_proba, n_classes) if y_proba is not None else np.nan
    
    total_time = time.perf_counter() - t0
    logging.info(f"  {model_name} | ACC={acc:.4f} | AUC={auc:.4f} | Tempo={total_time:.1f}s")
    return {
        "model": model_name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
        "G_Mean": round(gmean, 6), "CE": round(ce, 6), "total_time_s": round(total_time, 2)
    }

# --- MAIN ---
def main():
    logging.info(" INICIANDO PIPELINE ")
    all_results = []
    if os.path.exists(RESULTS_FILE):
        all_results = pd.read_csv(RESULTS_FILE).to_dict('records')
        done_pairs = set((r['dataset'], r['model']) for r in all_results)
    else:
        done_pairs = set()

    for ds in DATASETS:
        logging.info(f"\n{'='*65}")
        logging.info(f"📁 {ds['name']} (tid={ds['tid']}, regime={ds['regime']})")
        logging.info(f"{'='*65}")
        
        models_to_run = [
            "TabICL", "LightGBM_TD", "XGBoost_TD", "CatBoost_TD",
            "AutoGluon_Default", "AutoGluon_Extreme",
            "LightGBM_Tuned", "XGBoost_Tuned", "CatBoost_Tuned"
        ]
        
        if all((ds['name'], m) in done_pairs for m in models_to_run):
            logging.info("   All models complete for this dataset, skipping...")
            continue
            
        try:
            try:
                task = openml.tasks.get_task(ds["tid"])
                dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=False, download_features_meta_data=True)
            except openml.exceptions.OpenMLServerException as e:
                if "Unknown task" in str(e):
                    logging.warning(f"   Task {ds['tid']} inexistente. Tentando como Dataset ID direto...")
                    dataset = openml.datasets.get_dataset(ds["tid"], download_data=True, download_qualities=False, download_features_meta_data=True)
                else:
                    raise e

            X, y, cat_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
            
            # Limpeza de Nulos no target e alinhamento de index (Isso não é leakage, é formatação básica)
            y_clean = y.dropna()
            X = X.loc[y_clean.index]
            
            # Label Encoder seguro (Fit apenas no final para labels 0...n)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_clean)
            n_classes = len(le.classes_)
            
            # 1. SPLIT ANTES DO PRE-PROCESSAMENTO 
            try: 
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
                )
            except ValueError: 
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=SEED
                )
            
            # 2. PRE-PROCESSAMENTO (Fit apenas no TREINO)
            preprocessor = build_preprocessor(cat_indicator, X.columns)
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)
            
            # Remover colunas totalmente constantes baseadas no treino
            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()
                X_test = X_test.toarray()
            
            
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)
            
            # --- EXECUÇÃO DOS MODELOS ---
            def execute_model(m_name, run_func, *args):
                if (ds['name'], m_name) not in done_pairs:
                    try:
                        res = run_func(*args)
                        res['dataset'] = ds['name']
                        res['regime'] = ds['regime']
                        all_results.append(res)
                        pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
                    except Exception as e:
                        logging.error(f"  {m_name} FAILED: {e}", exc_info=True)

            # Defaults
            execute_model("TabICL", run_baseline, "TabICL", build_tabicl, X_train, y_train, X_test, y_test, n_classes)
            execute_model("LightGBM_TD", run_baseline, "LightGBM_TD", build_lightgbm, X_train, y_train, X_test, y_test, n_classes)
            execute_model("XGBoost_TD", run_baseline, "XGBoost_TD", build_xgboost, X_train, y_train, X_test, y_test, n_classes)
            execute_model("CatBoost_TD", run_baseline, "CatBoost_TD", build_catboost, X_train, y_train, X_test, y_test, n_classes)
            
            # AutoGluon
            execute_model("AutoGluon_Default", run_autogluon, "medium_quality", X_train, y_train, X_test, y_test, n_classes)
            execute_model("AutoGluon_Extreme", run_autogluon, "best_quality", X_train, y_train, X_test, y_test, n_classes)
            
            # Tuned
            execute_model("LightGBM_Tuned", run_tuning, "LightGBM_Tuned", objective_lgb, LGBMClassifier, X_train, y_train, X_test, y_test, n_classes)
            execute_model("XGBoost_Tuned", run_tuning, "XGBoost_Tuned", objective_xgb, XGBClassifier, X_train, y_train, X_test, y_test, n_classes)
            execute_model("CatBoost_Tuned", run_tuning, "CatBoost_Tuned", objective_cat, CatBoostClassifier, X_train, y_train, X_test, y_test, n_classes)
                    
        except Exception as e: 
            logging.error(f" DATASET {ds['name']} FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    main()
