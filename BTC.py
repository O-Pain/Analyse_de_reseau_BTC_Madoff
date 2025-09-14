import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx.algorithms.community as nxcom

links_df = pd.read_csv('bitcoin.links.csv', quoting=3, on_bad_lines='skip')
vertices_df = pd.read_csv('bitcoin.vertices.csv', quoting=3, on_bad_lines='skip')
print(links_df.head())
print(vertices_df.head())
G = nx.DiGraph()
G.add_weighted_edges_from(
    links_df[['src_id', 'dst_id', 'count']].values
)
print("Nombre de noeuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())

attributes = pd.Series(vertices_df.first_transaction_date.values, index=vertices_df.vid).to_dict()
nx.set_node_attributes(G, attributes, 'first_transaction_date')
sample_nodes = list(G.nodes())[:100]
subG = G.subgraph(sample_nodes)
pos = nx.spring_layout(subG, seed=42)
plt.figure(figsize=(10,7))
nx.draw(subG, pos, with_labels=False, node_size=30, arrowsize=10)
plt.title("Sous-graphe (100 premiers nœuds) Bitcoin Transactions")
plt.show()

print("Densité :", nx.density(G))
print("Le graphe est-il fortement connecté ?", nx.is_strongly_connected(G))
print("Le graphe est-il faiblement connecté ?", nx.is_weakly_connected(G))

giant_component = max(nx.weakly_connected_components(G), key=len)
G_giant = G.subgraph(giant_component).copy()
print("Taille de la composante géante :", len(G_giant.nodes))

sample_nodes = list(G_giant.nodes())[:100]
subG_giant = G_giant.subgraph(sample_nodes)
pos = nx.spring_layout(subG_giant, seed=42)
plt.figure(figsize=(10,7))
nx.draw(subG_giant, pos, with_labels=False, node_size=30, arrowsize=10)
plt.title("Sous-graphe (100 premiers nœuds) composante géante Bitcoin")
plt.show()

in_strength = dict(G_giant.in_degree(weight='weight'))
out_strength = dict(G_giant.out_degree(weight='weight'))

import random

sample_nodes = random.sample(list(G_giant.nodes()), 2000)

G_sample = G_giant.subgraph(sample_nodes).copy()

degree_centrality_sample = nx.degree_centrality(G_sample)
closeness_centrality_sample = nx.closeness_centrality(G_sample)
betweenness_centrality_sample = nx.betweenness_centrality(G_sample)
eigenvector_centrality_sample = nx.eigenvector_centrality(G_sample, max_iter=1000)

print("Top 5 Centralité de degré:", sorted(degree_centrality_sample.items(), key=lambda x: x[1], reverse=True)[:5])
print("Top 5 Centralité de Eigenvectors:", sorted(eigenvector_centrality_sample.items(), key=lambda x: x[1], reverse=True)[:5])

top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 (centralité de degré) :", top_degree)

top_betweenness = sorted(betweenness_centrality_sample.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 (centralité d’intermédiarité) :", top_betweenness)

weights = [d['weight'] for u,v,d in G_giant.edges(data=True)]
plt.hist(weights, bins=50, log=True)
plt.title("Distribution des poids des transactions")
plt.xlabel("Nombre de transactions (poids)")
plt.ylabel("Nombre de liens")
plt.show()

assortativity = nx.degree_assortativity_coefficient(G_giant)
print("Assortativité (degré) :", assortativity)

sample_nodes = random.sample(list(G_giant.nodes()), 2000)
G_sample = G_giant.subgraph(sample_nodes).copy()

clustering_sample = nx.clustering(G_sample)

avg_clustering_sample = nx.average_clustering(G_sample)

print("Coefficient de clustering moyen (échantillon) :", avg_clustering_sample)

largest_scc = max(nx.strongly_connected_components(G_giant), key=len)
G_scc = G_giant.subgraph(largest_scc).copy()

print("Longueur moyenne des chemins (APL) :", nx.average_shortest_path_length(G_scc))
print("Diamètre de la composante géante :", nx.diameter(G_scc.to_undirected()))

from copy import deepcopy

def robustness_test(G, removal_fraction=0.1):
    G_copy = deepcopy(G)
    num_remove = int(removal_fraction * G_copy.number_of_nodes())
    nodes_sorted = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
    to_remove = [n for n, d in nodes_sorted[:num_remove]]
    G_copy.remove_nodes_from(to_remove)
    largest_cc = max(nx.weakly_connected_components(G_copy), key=len)
    return len(largest_cc) / G.number_of_nodes()
print("Taille relative après suppression de 10% des hubs :", robustness_test(G_giant, 0.1))

sample_nodes = random.sample(list(G_giant.nodes()), 2000)
G_sample = G_giant.subgraph(sample_nodes).copy()
G_sample_undirected = G_sample.to_undirected()
communities_sample = nxcom.greedy_modularity_communities(G_sample_undirected)
print("Nombre de communautés détectées (échantillon) :", len(communities_sample))

color_map = {}
for c_idx, community in enumerate(communities_sample):
    for name in community:
        color_map[name] = c_idx

colors = [color_map.get(n, 0) for n in G_giant.nodes()]
sample_nodes = list(G_giant.nodes())[:2000]
subG_giant = G_giant.subgraph(sample_nodes)
pos = nx.spring_layout(subG_giant, seed=42)

plt.figure(figsize=(12,8))
nx.draw(subG_giant, pos, node_color=[color_map.get(n, 0) for n in sample_nodes],
        with_labels=False, node_size=20, cmap=plt.cm.tab20)
plt.title('Communautés Bitcoin (échantillon)')
plt.show()

sample_nodes = random.sample(list(G_giant.nodes()), 2000)
G_sample = G_giant.subgraph(sample_nodes).copy()
G_sample_undirected = G_sample.to_undirected()

core_numbers_sample = nx.core_number(G_sample_undirected)
print("Noyau le plus dense (k-core max, échantillon) :", max(core_numbers_sample.values()))

A_sample = nx.adjacency_matrix(G_sample)
print("Matrice d'adjacence du sous-graphe (échantillon) :")
print(A_sample.todense())

!pip install networkx
import networkx as nx
import numpy as np
knn_sample = nx.average_neighbor_degree(G_sample_undirected)
print("Degré moyen des voisins (échantillon) :", knn_sample)

triangles_sample = sum(nx.triangles(G_sample_undirected).values()) // 3
print("Nombre de triangles (échantillon) :", triangles_sample)

