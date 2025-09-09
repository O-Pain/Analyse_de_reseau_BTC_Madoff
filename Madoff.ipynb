import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
madoff_df = pd.read_csv('MADOFF.csv', index_col=0)

madoff_df = madoff_df.fillna(0)

G = nx.from_pandas_adjacency(madoff_df)

print("Nombre de noeuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())
print("Densité :", nx.density(G))
print("Liste des noeuds :", list(G.nodes)[:10])
print("Liste des arêtes :", list(G.edges)[:10])
degree_sequence = [d for n, d in G.degree()]
plt.hist(degree_sequence, bins=range(max(degree_sequence)+1))
plt.title("Distribution du degré")
plt.xlabel("Degré")
plt.ylabel("Nombre de nœuds")
plt.show()
avg_degree = sum(degree_sequence) / G.number_of_nodes()
print("Degré moyen :", avg_degree)
assortativity = nx.degree_assortativity_coefficient(G)
print("Assortativité par degré :", assortativity)
deg_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
sorted_degree = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 hubs (centralité de degré) :", sorted_degree[:5])
sizes = [5000 * deg_centrality[n] for n in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=sizes, font_size=6)
plt.show()
print("Le réseau est-il connecté :", nx.is_connected(G))
print("Nombre de composantes connectées :", nx.number_connected_components(G))
largest_cc = max(nx.connected_components(G), key=len)
print("Taille de la plus grande composante :", len(largest_cc))
G_giant = G.subgraph(largest_cc)
plt.figure(figsize=(12,8))
nx.draw(G_giant, pos, with_labels=True, node_size=50, font_size=6)
plt.title('Composante géante du réseau Madoff')
plt.show()
print("Longueur moyenne des chemins (APL) :", nx.average_shortest_path_length(G_giant))
print("Diamètre :", nx.diameter(G_giant))
print("Coefficient de clustering moyen :", nx.average_clustering(G))
print("Nombre de triangles dans le réseau :", triangles)
import networkx.algorithms.community as nxcom
print("Nombre de communautés trouvées :", len(communities))

color_map = {}
for c_idx, community in enumerate(communities):
    for name in community:
        color_map[name] = c_idx
colors = [color_map[n] for n in G.nodes()]
plt.figure(figsize=(12,8))
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.tab20, node_size=50, font_size=6)
plt.title('Réseau Madoff - Communautés détectées')
plt.show()

G.remove_edges_from(nx.selfloop_edges(G))

core_numbers = nx.core_number(G)
print("Core number max :", max(core_numbers.values()))

clustering = nx.clustering(G)
sorted_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
print("Top 5 noeuds avec le plus fort clustering :", sorted_clustering[:5])

edge_betweenness = nx.edge_betweenness_centrality(G)
sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
print("Top 5 liens avec la plus forte intermédiarité :", sorted_edges[:5])

path_lengths = dict(nx.all_pairs_shortest_path_length(G_giant))
all_lengths = [length for target_dict in path_lengths.values() for length in target_dict.values()]
plt.hist(all_lengths, bins=range(max(all_lengths)+1))
plt.title("Distribution des longueurs des plus courts chemins")
plt.xlabel("Longueur")
plt.ylabel("Fréquence")
plt.show()

from copy import deepcopy

def robustness_test(G, removal_fraction=0.1):
    G_copy = deepcopy(G)
    num_remove = int(removal_fraction * G_copy.number_of_nodes())
    nodes_sorted = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
    to_remove = [n for n, d in nodes_sorted[:num_remove]]
    G_copy.remove_nodes_from(to_remove)
    largest_cc = max(nx.connected_components(G_copy), key=len)
    return len(largest_cc) / G.number_of_nodes()

print("Taille relative après suppression de 10% des hubs :", robustness_test(G, 0.1))

import pandas as pd

deg_df = pd.DataFrame(sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True), columns=['Node', 'Degree Centrality'])
deg_df[:10].plot(x='Node', y='Degree Centrality', kind='bar', legend=False)
plt.title("Top 10 noeuds - Centralité de degré")
plt.show()

edge_color = [edge_betweenness.get(edge, 0) for edge in G.edges()]
plt.figure(figsize=(12,8))
nx.draw(G, pos, edge_color=edge_color, edge_cmap=plt.cm.plasma, with_labels=False, node_size=30)
plt.title('Liens colorés par leur intermédiarité')
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma))
plt.show()

community_sizes = [len(c) for c in communities]
plt.pie(community_sizes, labels=[f"Com {i}" for i in range(len(community_sizes))], autopct='%1.1f%%')
plt.title("Répartition des communautés")
plt.show()

A = nx.adjacency_matrix(G)
print("Matrice d'adjacence :")
print(A.todense())

strength = dict(G.degree(weight='weight'))
print("Force des nœuds :", strength)

A = nx.adjacency_matrix(G)
print("Matrice d'adjacence :")
print(A.todense())

strength = dict(G.degree(weight='weight'))
print("Force des nœuds :", strength)

knn = {node: sum(G.degree(neighbor) for neighbor in G.neighbors(node)) / G.degree(node) if G.degree(node) > 0 else 0 for node in G}
print("Degré moyen des voisins :", knn)
