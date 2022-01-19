import re
import numpy as np
from collections import defaultdict
import networkx as nx
import torch

def load_data(G, feat_dir):
    data = np.genfromtxt(feat_dir, delimiter=',', skip_header=True, dtype=str)
    features = np.array(np.delete(data[:,2:], -3, 1), dtype=float)
    uris = data[:, 1]
    uris = [re.sub('spotify:track:', '', uri) for uri in uris]
    uri_map = {n: i for i,n in enumerate(uris)}

    adj_list = defaultdict(set)    
    for e in G.edges:
        u,v = uri_map[e[0]], uri_map[e[1]]
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    return features, adj_list

def adj_matrix(adj_list):
    row_idx = torch.LongTensor([k for k in range(len(adj_list.keys())) for v in range(len(adj_list[k]))])
    col_idx = torch.LongTensor([v for k in range(len(adj_list.keys())) for v in adj_list[k]]) 

    idx = torch.vstack((row_idx, col_idx))
    
    return torch.sparse_coo_tensor(indices = idx, values = torch.ones(len(row_idx)), 
                                   size=[len(adj_list.keys()), len(adj_list.keys())])

def make_label(batch_nodes, adj_list):
    batch_map = {n:i for i,n in enumerate(batch_nodes)}
    neigh_list = [adj_list[n].intersection(batch_nodes) for n in batch_nodes]
    mask = torch.zeros(len(neigh_list), len(neigh_list)) 
    column_indices = [batch_map[n] for neigh in neigh_list for n in neigh]   
    row_indices = [i for i in range(len(neigh_list)) for j in range(len(neigh_list[i]))]
    mask[row_indices, column_indices] = 1
    
    return mask