import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('cluster_apuana/resultados_estat_finais/plots', exist_ok=True)
df = pd.read_csv('cluster_apuana/resultados_estat_finais/data/final_run_results_v2.csv')

# Removendo o AutoGluon Extreme antigo (bugado/duplicado) mantendo apenas a versão 4h
df = df[df['model'] != 'AutoGluon_Extreme'].copy()

# Renomeando o AutoGluon_Extreme_4h para AutoGluon_Extreme para ficar mais limpo no gráfico
df['model'] = df['model'].replace({'AutoGluon_Extreme_4h': 'AutoGluon_Extreme'})

# Usar todos os modelos presentes no CSV para o scatter plot
# Agrupar pela média
means = df.groupby('model')[['total_time_s', 'AUC_OVO']].mean().reset_index()

# Calcular Fronteira de Pareto (Minimizando Tempo, Maximizando AUC)
# Ponto domina o outro se tempo <= tempo_outro E auc >= auc_outro (e pelo menos um é estritamente melhor)
pareto_front = []
for i, row in means.iterrows():
    is_pareto = True
    for j, other_row in means.iterrows():
        if i == j:
            continue
        # Se 'other_row' tem tempo menor ou igual E AUC maior ou igual
        if (other_row['total_time_s'] <= row['total_time_s']) and (other_row['AUC_OVO'] >= row['AUC_OVO']):
            # E pelo menos um é estritamente melhor
            if (other_row['total_time_s'] < row['total_time_s']) or (other_row['AUC_OVO'] > row['AUC_OVO']):
                is_pareto = False
                break
    if is_pareto:
        pareto_front.append(row)

pareto_df = pd.DataFrame(pareto_front)
# Ordenar a fronteira pelo tempo
pareto_df = pareto_df.sort_values('total_time_s')

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

sns.scatterplot(data=means, x='total_time_s', y='AUC_OVO', s=150, hue='model', palette='colorblind', style='model', markers=True)

# Desenhar a linha de Pareto
plt.plot(pareto_df['total_time_s'], pareto_df['AUC_OVO'], color='red', linestyle='--', linewidth=2, label='Fronteira de Pareto', alpha=0.7)

# Adicionar textos
for i, row in means.iterrows():
    plt.annotate(row['model'].replace('_', ' '), 
                 (row['total_time_s'], row['AUC_OVO']),
                 xytext=(8, 0), 
                 textcoords='offset points',
                 fontsize=14)

plt.xscale('log')
# Adicionando uma margem gigante à direita no eixo x para o texto não vazar (escala log)
plt.xlim(right=means['total_time_s'].max() * 50)
plt.title('Custo Computacional vs Desempenho (Fronteira de Pareto)', fontsize=18, fontweight='bold')
plt.xlabel('Tempo Total Médio de Treino/Inferência (s) - Escala Log', fontsize=16)
plt.ylabel('AUC-OVO Médio', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

# Remover a caixa da legenda (os rótulos já estão flutuantes)
ax = plt.gca()
if ax.get_legend():
    ax.get_legend().remove()

plt.tight_layout()
output_path = 'cluster_apuana/resultados_estat_finais/plots/scatter_pareto.png'
plt.savefig(output_path, dpi=300)
print(f"Scatter plot salvo em {output_path}")
