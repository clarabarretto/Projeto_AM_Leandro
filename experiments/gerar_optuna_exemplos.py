from sklearn.ensemble import RandomForestClassifier
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
import warnings
import openml
import optuna
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42

# Datasets escolhidos para cada regime (TIDs do OpenML)
regimes = {
    'Small': 40,   # Sonar (pequeno)
    'Medium': 31,  # Credit-g (médio)
    'Large': 1461  # Bank-marketing (grande, mas roda em poucos minutos)
}

for regime_name, tid in regimes.items():
    print(f"\n--- Iniciando regime {regime_name} (TID: {tid}) ---")
    dataset = openml.datasets.get_dataset(tid)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    
    # Preprocessing
    for col in X.columns:
        if str(X[col].dtype) == 'category' or str(X[col].dtype) == 'object':
            X[col] = X[col].cat.codes

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': SEED,
            'n_jobs': 1
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=15) # 15 trials para agilizar o grande
    # Gerar gráficos
    ax_hist = plot_optimization_history(study)
    if ax_hist.figure._suptitle:
        ax_hist.figure.suptitle("")
        
    for text_obj in ax_hist.findobj(match=plt.Text):
        if "Optimization History" in text_obj.get_text():
            text_obj.set_text(f"Histórico de Otimização - {regime_name}")
            
    plt.tight_layout()
    plt.savefig(f'results/plots/optuna_history_{regime_name.lower()}.png', dpi=300)
    plt.close()

    ax_imp = plot_param_importances(study)
    if ax_imp.figure._suptitle:
        ax_imp.figure.suptitle("")
        
    for text_obj in ax_imp.findobj(match=plt.Text):
        if "Hyperparameter Importances" in text_obj.get_text():
            text_obj.set_text(f"Importância de Hiperparâmetros - {regime_name}")

    plt.tight_layout()
    plt.savefig(f'results/plots/optuna_importances_{regime_name.lower()}.png', dpi=300)
    plt.close()

print("\n✅ Todos os gráficos de todos os regimes gerados!")
