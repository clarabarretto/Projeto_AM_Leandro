import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autorank import autorank, create_report, plot_stats
from baycomp import SignedRankTest
import os

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

df = pd.read_csv('results/data/final_run_results_v2.csv')

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
        'ACC': means['ACC'], 'AUC_OVO': means['AUC_OVO'], 
        'G_Mean': means['G_Mean'], 'CE': means['CE'], 'total_time_s': means['total_time_s']
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

print("✅ Furos imputados com a média do regime correspondente.")

# -- MERGE METADATA --
meta = pd.read_csv('results/data/dataset_metadata.csv')
df = df.merge(meta, on='dataset', how='left')

metrics = ['AUC_OVO', 'ACC', 'G_Mean', 'CE', 'total_time_s']

def gerar_tabela_latex(df_group, titulo, label, f):
    agg = df_group.groupby('model')[metrics].agg(['mean', 'std'])
    tex_table = "\\begin{table}[htbp]\n\\centering\n\\caption{" + titulo + "}\n\\label{" + label + "}\n"
    tex_table += "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l" + "c"*len(metrics) + "}\n\\toprule\n"
    tex_table += "\\textbf{Modelo} & \\textbf{AUC-OVO} & \\textbf{Acurácia} & \\textbf{G-Mean} & \\textbf{CE} & \\textbf{Tempo (s)} \\\\\n\\midrule\n"
    
    ordem_modelos = [
        'TabICL',
        'LightGBM_TD', 'XGBoost_TD', 'CatBoost_TD', 'AutoGluon_Default',
        'LightGBM_Tuned', 'XGBoost_Tuned', 'CatBoost_Tuned',
        'AutoGluon_Extreme_4h'
    ]
    
    # Adicionar coluna de ordenação
    agg = agg.reset_index()
    agg['sort_order'] = agg['model'].map(lambda x: ordem_modelos.index(x) if x in ordem_modelos else 999)
    agg = agg.sort_values(by='sort_order').drop(columns=['sort_order'])
    
    for _, row in agg.iterrows():
        model = row['model'].values[0] if isinstance(row['model'], pd.Series) else row['model']
        if isinstance(model, pd.Series):
            model = model.iloc[0]
        model_clean = str(model).replace('_', '\\_')
        row_str = f"{model_clean}"
        for m in metrics:
            mean_val = row[(m, 'mean')]
            std_val = row[(m, 'std')]
            row_str += f" & {mean_val:.4f} $\\pm$ {std_val:.4f}"
        row_str += " \\\\\n"
        tex_table += row_str
        
    tex_table += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n\n"
    f.write(tex_table)

with open('results/tables/tabelas_resultados.tex', 'w') as f:
    # 2.1 TABELAS POR REGIME
    for reg in ['small', 'medium', 'large']:
        df_sub = df[df['regime'] == reg]
        gerar_tabela_latex(df_sub, f"Resultados Agregados - Regime {reg.capitalize()}", f"tab:res_{reg}", f)
        
    # 2.2 TABELAS POR CLASSES (Binário vs Multiclasse)
    for is_bin, nome in [(True, "Binário"), (False, "Multiclasse")]:
        df_sub = df[df['is_binary'] == is_bin]
        gerar_tabela_latex(df_sub, f"Resultados Agregados - Datasets {nome}", f"tab:res_{nome.lower()}", f)
        
    # 2.3 TABELAS POR TIPO DE FEATURES (Numérico vs Categórico)
    for is_cat, nome in [(True, "Maioria Categórica"), (False, "Maioria Numérica")]:
        df_sub = df[df['is_mostly_categorical'] == is_cat]
        gerar_tabela_latex(df_sub, f"Resultados Agregados - {nome}", f"tab:res_{nome.split()[-1].lower()}", f)
        
    # 2.4 TABELAS POR MISSING VALUES
    for has_nan, nome in [(True, "Com Valores Ausentes (NaNs)"), (False, "Sem Valores Ausentes")]:
        df_sub = df[df['has_missing'] == has_nan]
        gerar_tabela_latex(df_sub, f"Resultados Agregados - {nome}", f"tab:res_nan_{has_nan}", f)

print("✅ Todas as 9 Tabelas LaTeX segmentadas foram geradas.")

# 3. CD DIAGRAM (AUTORANK) PARA TODAS AS MÉTRICAS
for metric in ['AUC_OVO', 'ACC', 'G_Mean', 'CE']:
    pivot_df = df.pivot(index='dataset', columns='model', values=metric)
    
    # Se for CE (Cross-Entropy), valores MENORES são melhores.
    # O autorank assume que maior é melhor por padrão (order='descending' dependendo da versão, ou basta inverter o sinal).
    # Uma forma robusta no autorank é não inverter, mas ao interpretar sabemos o significado.
    # Para garantir consistência no visual:
    try:
        if metric == 'CE':
            res = autorank(pivot_df, alpha=0.05, verbose=False, order='ascending')
        else:
            res = autorank(pivot_df, alpha=0.05, verbose=False)
            
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_stats(res, ax=ax, allow_insignificant=True)
        filename = f"cd_diagram_{metric.lower()}.png"
        plt.savefig(f'results/plots/{filename}', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ CD Diagram para {metric} gerado em {filename}.")
    except Exception as e:
        print(f"⚠️ Erro ao gerar CD Diagram para {metric}: {e}")

# 4. SCATTER PLOT
global_agg = df.groupby('model').agg({'AUC_OVO': 'mean', 'total_time_s': 'mean'}).reset_index()
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.scatterplot(data=global_agg, x='total_time_s', y='AUC_OVO', hue='model', s=200, palette='tab10', edgecolor='black')
plt.xscale('log')
plt.xlim(right=global_agg['total_time_s'].max() * 10)
plt.xlabel('Tempo Médio de Treinamento/Inferência (s) - Log Scale')
plt.ylabel('Média AUC-OVO')
plt.title('Custo vs. Desempenho Geral')
for i in range(global_agg.shape[0]):
    plt.text(global_agg['total_time_s'][i] * 1.1, global_agg['AUC_OVO'][i], global_agg['model'][i], size='small', color='black')
plt.legend([],[], frameon=False)
plt.savefig('results/plots/scatter_cost_perf.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Gráfico de Dispersão gerado.")

# 5. BAYESIAN SIGNED-RANK TEST (Baycomp)
pivot_auc = df.pivot(index='dataset', columns='model', values='AUC_OVO')
tabicl_scores = pivot_auc['TabICL'].values

baselines = ['LightGBM_Tuned', 'AutoGluon_Extreme_4h']

for baseline_name in baselines:
    baseline_scores = pivot_auc[baseline_name].values
    try:
        # rope=0.01 (diferença de 1% em AUC é ignorável / praticamente equivalente)
        test = SignedRankTest(tabicl_scores, baseline_scores, rope=0.01)
        probs = test.probs()
        fig = test.plot(names=('TabICL', baseline_name.replace('_4h', '')))
        fig.suptitle(f'Bayesian Signed-Rank\nTabICL vs {baseline_name} (ROPE=0.01)')
        filename = f'bayesian_plot_auc_{baseline_name.lower()}.png'
        fig.savefig(f'results/plots/{filename}', bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"✅ Gráfico Bayesiano gerado em {filename}")
        print(f"Probabilidades vs {baseline_name}: TabICL ganha: {probs[0]:.3f}, Equivalentes: {probs[1]:.3f}, Baseline ganha: {probs[2]:.3f}")
    except Exception as e:
        print(f"⚠️ Erro ao gerar gráfico bayesiano para {baseline_name}: {e}")
