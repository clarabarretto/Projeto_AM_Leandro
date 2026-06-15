import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('cluster_apuana/resultados_estat_finais/plots', exist_ok=True)
df = pd.read_csv('cluster_apuana/resultados_estat_finais/data/final_run_results_v2.csv')

# --- REMOVER AUTOGLUON EXTREME ANTIGO ---
df = df[df['model'] != 'AutoGluon_Extreme'].copy()

# 1. IMPUTATION OF MISSING VALUES
missing_records = [
    {'model': 'TabICL', 'dataset': 'anneal', 'regime': 'small'},
    {'model': 'AutoGluon_Default', 'dataset': 'houses', 'regime': 'large'},
    {'model': 'XGBoost_Tuned', 'dataset': 'nursery', 'regime': 'large'}
]

for rec in missing_records:
    model = rec['model']
    regime = rec['regime']
    dataset = rec['dataset']
    means = df[(df['model'] == model) & (df['regime'] == regime)].mean(numeric_only=True)
    if means.isna().all():
        means = df[df['model'] == model].mean(numeric_only=True)
    new_row = {
        'model': model, 'dataset': dataset, 'regime': regime,
        'AUC_OVO': means['AUC_OVO']
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# -- MERGE METADATA --
meta = pd.read_csv('cluster_apuana/resultados_estat_finais/data/dataset_metadata.csv')
df = df.merge(meta, on='dataset', how='left')

# Criar colunas formatadas para os eixos do gráfico
df['type'] = df['is_binary'].apply(lambda x: 'Binário' if x else 'Multiclasse')
df['cat_dominante'] = df['cat_ratio'].apply(lambda x: 'Categorico (>0.5)' if x > 0.5 else 'Numerico (<=0.5)')
df['missing_cat'] = df['has_missing'].apply(lambda x: 'Com Missing' if x else 'Sem Missing')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.set_theme(style="whitegrid")

models_order = ['TabICL', 'AutoGluon_Extreme_4h', 'AutoGluon_Default', 'LightGBM_Tuned', 'XGBoost_Tuned', 'CatBoost_Tuned']
# Filtrar df apenas pros top models
df_plot = df[df['model'].isin(models_order)].copy()

# Usar paleta focada em acessibilidade para daltonismo
palette = sns.color_palette("colorblind", len(models_order))

# Plot 1: Regime de Tamanho
sns.barplot(data=df_plot, x='regime', y='AUC_OVO', hue='model', ax=axes[0,0], order=['small', 'medium', 'large'], hue_order=models_order, palette=palette)
axes[0,0].set_title('Eixo 1: AUC-OVO por Tamanho do Dataset', fontsize=16, fontweight='bold')
axes[0,0].set_ylim(0.6, 1.0)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('AUC-OVO (Médio)', fontsize=16)
axes[0,0].tick_params(axis='both', which='major', labelsize=14)
axes[0,0].legend(loc='lower right', fontsize='small')

# Plot 2: Binario vs Multiclasse
sns.barplot(data=df_plot, x='type', y='AUC_OVO', hue='model', ax=axes[0,1], order=['Binário', 'Multiclasse'], hue_order=models_order, palette=palette)
axes[0,1].set_title('Eixo 2: AUC-OVO por N° de Classes', fontsize=16, fontweight='bold')
axes[0,1].set_ylim(0.6, 1.0)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('')
axes[0,1].tick_params(axis='both', which='major', labelsize=14)
axes[0,1].get_legend().remove()

# Plot 3: Categorico vs Numerico
sns.barplot(data=df_plot, x='cat_dominante', y='AUC_OVO', hue='model', ax=axes[1,0], hue_order=models_order, palette=palette)
axes[1,0].set_title('Eixo 3: AUC-OVO por Tipo de Variáveis', fontsize=16, fontweight='bold')
axes[1,0].set_ylim(0.6, 1.0)
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('AUC-OVO (Médio)', fontsize=16)
axes[1,0].tick_params(axis='both', which='major', labelsize=14)
axes[1,0].get_legend().remove()

# Plot 4: Missing Values
sns.barplot(data=df_plot, x='missing_cat', y='AUC_OVO', hue='model', ax=axes[1,1], order=['Sem Missing', 'Com Missing'], hue_order=models_order, palette=palette)
axes[1,1].set_title('Eixo 4: AUC-OVO por Valores Ausentes (NaN)', fontsize=16, fontweight='bold')
axes[1,1].set_ylim(0.6, 1.0)
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('')
axes[1,1].tick_params(axis='both', which='major', labelsize=14)
axes[1,1].get_legend().remove()

plt.tight_layout()
output_path = 'cluster_apuana/resultados_estat_finais/plots/graficos_regime.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Grafico salvo em {output_path}")
