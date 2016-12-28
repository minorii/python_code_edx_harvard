# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 21:18:45 2016

@author: minori
"""

import pandas as pd
import numpy as np
df  = pd.read_stata("individual_characteristics.dta")
df1 = df[df.village == 1]# Enter code here!
df2 = df[df.village == 2]# Enter code here!

# Enter code here!
print(df1.head())

# Enter code here!
pid1 = pd.read_csv('key_vilno_1.csv', header=None)
pid2 = pd.read_csv('key_vilno_2.csv', header=None)



sex1      = dict(zip(df1['pid'], df1['resp_gend']))# Enter code here!
caste1    = dict(zip(df1['pid'], df1['caste']))# Enter code here!
religion1 = dict(zip(df1['pid'], df1['religion']))# Enter code here!

# Continue for df2 as well.
sex2      = dict(zip(df2['pid'], df2['resp_gend']))# Enter code here!
caste2    = dict(zip(df2['pid'], df2['caste']))# Enter code here!
religion2 = dict(zip(df2['pid'], df2['religion']))# Enter code here!

'''            
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()

sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()                     
'''


from collections import Counter
def chance_homophily(chars):
    """
    Computes the chance homophily of a characteristic,
    specified as a dictionary, chars.
    """
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props  = chars_counts / sum(chars_counts)
    return sum(chars_props**2)

favorite_colors = {
    "ankit":  "red",
    "xiaoyu": "blue",
    "mary":   "blue"
}

print(chance_homophily(favorite_colors))


print("Village 1 chance of same sex:", chance_homophily(sex1))
print("Village 1 chance of same caste:", chance_homophily(caste1))
print("Village 1 chance of same religion:", chance_homophily(religion1))

print("Village 2 chance of same sex:", chance_homophily(sex2))
print("Village 2 chance of same caste:", chance_homophily(caste2))
print("Village 2 chance of same religion:", chance_homophily(religion2))

def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties, num_ties = 0, 0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:   # do not double-count edges!
                if IDs[0][n1] in chars and IDs[0][n2] in chars:
                    if G.has_edge(n1, n2):
                        # Should `num_ties` be incremented?  What about `num_same_ties`?
                        num_ties += 1
                        if chars[IDs[0][n1]] == chars[IDs[0][n2]]:
                            # Should `num_ties` be incremented?  What about `num_same_ties`?
                            num_same_ties += 1
    return (num_same_ties / num_ties)

import networkx as nx
# first read in the network of adjacency matrices and construct the networks
A1 = np.loadtxt('adj_allVillageRelationships_vilno_1.csv',delimiter=',')
A2 = np.loadtxt('adj_allVillageRelationships_vilno_2.csv',delimiter=',')

# convert the adjacency matrices to graph objects.
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)     
    
print("Village 1 observed proportion of same sex:", homophily(G1, sex1, pid1))
print("Village 1 observed proportion of same caste:", homophily(G1, caste1, pid1))
print("Village 1 observed proportion of same religion:", homophily(G1, religion1, pid1))

print("Village 2 observed proportion of same sex:", homophily(G2, sex2, pid2))
print("Village 2 observed proportion of same caste:", homophily(G2, caste2, pid2))
print("Village 2 observed proportion of same religion:", homophily(G2, religion2, pid2))















