""" Segmentação com K-means
❱ Objetivo:
Agrupar clientes por perfil de consumo para orientar campanhas.

❱ Tarefas:
Normalizar variáveis numéricas; one-hot na região.
Treinar K-Means com k∈{2,3,4}.
Escolher kkk com silhouette score e justificar (3–5 linhas).
(Opcional) Nomear segmentos (“Preço-sensível”, “Frequente”, etc.) e ligar ao teu setor.

❱ Sugestões de pesquisa:
"StandardScaler sklearn";
"KMeans sklearn";
"silhouette_score";
"pandas get_dummies".

Sugestão de resolução (Raciocínio)
❱ Passos essenciais:

Carregar e preparar dados
○ Variáveis: idade, gasto_mensal, visitas_mensais (numéricas) e regiao (categórica).
○ Porquê: K-Means usa distância Euclidiana; precisamos pôr as variáveis na mesma escala e codificar categorias.

Codificação e normalização

○ One-hot para regiao (ex.: get_dummies(drop_first=True)).
○ Normalizar numéricas com StandardScaler.
○ Porquê: evita que variáveis com escala maior dominem a distância.

Treinar K-Means e escolher k

○ Testar k∈{2,3,4}k\in\{2,3,4\}k∈{2,3,4}.
○ Calcular silhouette score para cada kkk; escolher o maior.
○ Porquê: silhouette mede coesão e separação dos clusters.

Interpretar os clusters

○ Analisar médias por cluster (idade, gasto, visitas) e sugerir rótulos (“Alto gasto & alta frequência”, etc.).
○ Porquê: transformar números em segmentos de negócio acionáveis.

Entregáveis

○ Tabela k vs. silhouette.
○ Tabela com médias por cluster; 2 recomendações de ação por segmento (campanhas, upgrades, retenção).

 """


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("dados.csv")
X = pd.get_dummies(df[['idade', 'gasto_mensal', 'visitas_mensais', 'regiao']], drop_first=True)
num_cols = ['idade', 'gasto_mensal', 'visitas_mensais']
X[num_cols] = StandardScaler().fit_transform(X[num_cols])

best_k, best_s = None, -1
for k in [2,3,4]:
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    s = silhouette_score(X, km.labels_)
    if s > best_s: best_k, best_s, best_labels = k, s, km.labels_

df['cluster'] = best_labels
print("Melhor k:", best_k, "silhouette:", round(best_s,3))
print(df.groupby('cluster')[['idade', 'gasto_mensal', 'visitas_mensais']].mean())
        