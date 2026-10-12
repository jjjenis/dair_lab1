import itertools
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.algorithms import community as nxc

from common import DATA, FIG, RES

sns.set_theme(style="white", context="talk")
MIN_FREQ = 3
MIN_EDGE_W = 2 
papers = json.loads((DATA / "keywords.json").read_text())

freq = Counter(k for p in papers for k in set(p["keywords"]))
nodes = {k for k, c in freq.items() if c >= MIN_FREQ}
G = nx.Graph()
for k in nodes:
    G.add_node(k, freq=freq[k])
for p in papers:
    ks = sorted(set(p["keywords"]) & nodes)
    for a, b in itertools.combinations(ks, 2):
        w = G.get_edge_data(a, b, {"weight": 0})["weight"]
        G.add_edge(a, b, weight=w + 1)
full_Q = nxc.modularity(G, nxc.louvain_communities(G, weight="weight", seed=42), weight="weight")
full_info = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "louvain_modularity": full_Q}
G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] < MIN_EDGE_W])
G.remove_nodes_from(list(nx.isolates(G)))
comps = sorted(nx.connected_components(G), key=len, reverse=True)
H = G.subgraph(comps[0]).copy()  
info = {"full_graph": full_info, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "components": len(comps), "largest_component_nodes": H.number_of_nodes(),
        "largest_component_edges": H.number_of_edges(),
        "density": nx.density(H), "avg_clustering": nx.average_clustering(H),
        "avg_degree": 2 * H.number_of_edges() / H.number_of_nodes()}
print(info)

algos = {
    "Louvain": lambda g: nxc.louvain_communities(g, weight="weight", resolution=1.0, seed=42),
    "Greedy modularity": lambda g: nxc.greedy_modularity_communities(g, weight="weight"),
    "Label propagation": lambda g: list(nxc.asyn_lpa_communities(g, weight="weight", seed=42)),
}
quality = {}
partitions = {}
for name, f in algos.items():
    parts = [set(c) for c in f(H)]
    partitions[name] = parts
    quality[name] = {"modularity": nxc.modularity(H, parts, weight="weight"),
                     "n_clusters": len(parts),
                     "coverage": nxc.partition_quality(H, parts)[0],
                     "largest_cluster": max(len(c) for c in parts),
                     "n_clusters_ge5": sum(len(c) >= 5 for c in parts)}

null_q = []
deg = [d for _, d in H.degree()]
for s in range(5):
    R = nx.Graph(nx.configuration_model(deg, seed=s))
    R.remove_edges_from(nx.selfloop_edges(R))
    null_q.append(nxc.modularity(R, nxc.louvain_communities(R, seed=s)))
quality["null_model_modularity_mean"] = float(np.mean(null_q))
print(json.dumps(quality, indent=1))

best = partitions["Louvain"]
best = sorted(best, key=len, reverse=True)
node2c = {n: i for i, c in enumerate(best) for n in c}
nx.set_node_attributes(H, node2c, "cluster")


Hw = H.copy()
for u, v, d in Hw.edges(data=True):
    d["dist"] = 1.0 / d["weight"]
cent = pd.DataFrame({
    "freq": pd.Series(nx.get_node_attributes(H, "freq")),
    "degree": pd.Series(nx.degree_centrality(H)),
    "weighted_degree": pd.Series(dict(H.degree(weight="weight"))),
    "betweenness": pd.Series(nx.betweenness_centrality(Hw, weight="dist")),
    "closeness": pd.Series(nx.closeness_centrality(Hw, distance="dist")),
    "eigenvector": pd.Series(nx.eigenvector_centrality_numpy(H, weight="weight")),
    "pagerank": pd.Series(nx.pagerank(H, weight="weight")),
    "cluster": pd.Series(node2c),
})
cent.index.name = "keyword"
cent.to_csv(RES / "keyword_centralities.csv")
print(cent.sort_values("degree", ascending=False).head(15).round(3))


cluster_rows = []
for i, c in enumerate(best):
    sub = cent.loc[list(c)]
    top = sub.sort_values("weighted_degree", ascending=False).head(10).index.tolist()
    cluster_rows.append({"cluster": i, "size": len(c), "top_keywords": top,
                         "internal_weight": int(sum(d["weight"] for u, v, d in H.subgraph(c).edges(data=True)))})
(RES / "clusters.json").write_text(json.dumps(cluster_rows, ensure_ascii=False, indent=1))
(RES / "graph_stats.json").write_text(json.dumps({"graph": info, "quality": quality}, indent=1))
for r in cluster_rows[:12]:
    print(r["cluster"], r["size"], r["top_keywords"])



fig, ax = plt.subplots(figsize=(18, 14))
pos = nx.spring_layout(H, k=0.35, seed=7, weight="weight", iterations=150)
palette = sns.color_palette("tab20", n_colors=max(len(best), 3))
sizes = [40 + 12 * cent.loc[n, "weighted_degree"] for n in H]
nx.draw_networkx_edges(H, pos, alpha=0.15, width=[0.4 + 0.5 * d["weight"] for *_, d in H.edges(data=True)], ax=ax)
nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=[palette[node2c[n] % 20] for n in H],
                       alpha=0.9, linewidths=0.3, edgecolors="k", ax=ax)

labels = {}
for i, c in enumerate(best[:16]):
    labels[cent.loc[list(c)].sort_values("weighted_degree", ascending=False).index[0]] = f"{i}: "
labels = {n: l + n for n, l in labels.items()}
nx.draw_networkx_labels(H, pos, labels, font_size=12, font_weight="bold", ax=ax,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
ax.set_title(f"Граф ключевых слов: {H.number_of_nodes()} вершин, {H.number_of_edges()} рёбер, "
             f"Louvain: {len(best)} кластеров, Q = {quality['Louvain']['modularity']:.3f}")
ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / "07_keyword_graph.png", dpi=150); plt.close()


fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for a, col in zip(axes.ravel(), ["degree", "betweenness", "closeness", "eigenvector"]):
    s = cent[col].sort_values().tail(12)
    a.barh(s.index, s.values, color=sns.color_palette("viridis", 12))
    a.set_title(f"Топ-12: {col}")
plt.tight_layout(); plt.savefig(FIG / "08_centralities.png", dpi=150); plt.close()


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
dseq = np.array(sorted([d for _, d in H.degree()], reverse=True))
cnt = Counter(dseq)
ax[0].loglog(list(cnt.keys()), list(cnt.values()), "o", alpha=.7)
ax[0].set(title="Распределение степеней вершин", xlabel="Степень", ylabel="Число вершин")
ax[1].bar(range(len(best)), [len(c) for c in best], color="#4C72B0")
ax[1].set(title="Размеры кластеров (Louvain)", xlabel="Кластер", ylabel="Ключевых слов")
plt.tight_layout(); plt.savefig(FIG / "09_degree_clusters.png", dpi=150); plt.close()


plt.figure(figsize=(8, 6.5))
sns.heatmap(cent[["degree", "betweenness", "closeness", "eigenvector", "pagerank", "freq"]]
            .corr(method="spearman"), annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
plt.title("Корреляция центральностей (Спирмен)")
plt.tight_layout(); plt.savefig(FIG / "10_centrality_corr.png", dpi=150); plt.close()

nx.write_gexf(H, RES / "keyword_graph.gexf")
