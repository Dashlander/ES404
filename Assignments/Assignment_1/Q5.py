import networkx as nx
import random

def snobbynet(N, p, q):
    G = nx.Graph()
    blue_nodes = range(N)
    red_nodes = range(N, 2*N)
    G.add_nodes_from(blue_nodes, color='blue')
    G.add_nodes_from(red_nodes, color='red')

    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if G.nodes[u]['color'] == G.nodes[v]['color']:
                    if G.nodes[u]['color'] == 'blue':
                        if random.random() < p:
                            G.add_edge(u, v)
                    else:
                        if random.random() < p:
                            G.add_edge(u, v)
                else:
                    if random.random() < q:
                        G.add_edge(u, v)

    return G

# Example usage:
N = 1000
p = 0.1
q = 0.002
G = snobbynet(N, p, q)
apl = nx.average_shortest_path_length(G)
acc = nx.average_clustering(G)
print("Average path length:", apl)
print("Average clustering coefficient:", acc)
