# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 08:57:39 2016

@author: minori
"""

import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_nodes_from([3,4])
G.add_nodes_from(['a', 's'])
print(G.nodes())

G.add_edge(1,3)
G.add_edge('a', 's')
G.add_edges_from([(1,2), (1,5), (1,6), (1,8)])
print(G.edges())

G.remove_node(2)
print(G.nodes())
G.remove_nodes_from([4,5])
print(G.nodes())

G.remove_edge(1,3)
print(G.edges())
G.remove_edges_from([(1,3), ('a', 's')])
print(G.edges())

print(G.number_of_nodes())
print(G.number_of_edges())


G = nx.karate_club_graph()
import pylab as plt
plt.figure()
nx.draw(G, with_labels = True, node_color = 'lightblue', edge_color = 'gray')
plt.savefig('karate_graph.pdf')
print(G.degree())
print(G.degree()[3])
print(G.degree(3))


from scipy.stats import bernoulli
print(bernoulli.rvs(p = 0.2))

N = 20
p = 0.2
def er_graph(N, p):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p = p):
                G.add_edge(node1, node2)
    return G

G = er_graph(N = 50, p = 0.08)    
print(G.number_of_nodes())
plt.figure()
nx.draw(G, with_labels = False, node_size = 40, 
        node_color = 'gray', edge_color = 'gray')
plt.savefig('er1.pdf')



def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()), histtype = 'step')
    plt.xlabel('Degree $K$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution')


plt.figure()
G1 = er_graph(N = 500, p = 0.08)    
plot_degree_distribution(G1)
G2 = er_graph(N = 500, p = 0.08)    
plot_degree_distribution(G2)
G3 = er_graph(N = 500, p = 0.08)    
plot_degree_distribution(G3)
plt.savefig('er_hist_graph3.pdf')

import numpy as np
A1 = np.loadtxt('adj_allVillageRelationships_vilno_1.csv', delimiter = ',')
A2 = np.loadtxt('adj_allVillageRelationships_vilno_2.csv', delimiter = ',')

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print('Number of nodes: %d' % G.number_of_nodes())
    print('Number of edges: %d' % G.number_of_edges())
    print('Averqge degree: %.2f' % np.mean(list(G.degree().values())))
    
basic_net_stats(G1)
basic_net_stats(G2)

plt.figure()
plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig('village_hist.pdf')


gen = nx.connected_component_subgraphs(G1)
g = gen.__next__()
print(type(g))
print(g.number_of_nodes())
print(len(gen.__next__()))

print(G1.number_of_nodes())
print(len(G1))

G1_LCC = max(nx.connected_component_subgraphs(G1), key = len)
G2_LCC = max(nx.connected_component_subgraphs(G2), key = len)
print(len(G1_LCC))
print(G2_LCC.number_of_nodes())

print(len(G1_LCC)/len(G1))
print(len(G2_LCC)/len(G2))

plt.figure()
nx.draw(G1_LCC, node_color = 'red', edge_color = 'gray', node_size = 20)
plt.savefig('village1.pdf')

plt.figure()
nx.draw(G2_LCC, node_color = 'green', edge_color = 'gray', node_size = 20)
plt.savefig('village2.pdf')












