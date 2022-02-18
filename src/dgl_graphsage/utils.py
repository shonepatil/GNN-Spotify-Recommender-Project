import re
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.functional import normalize
import torch
import dgl
import networkx as nx

import sys
sys.path.insert(0, './src/features')
from build_features import load_data as graph_from_scratch

def get_gpickle(data_path, dataset, pickle_path):
    G = graph_from_scratch(data_path, dataset, 0, 0, 0)
    nx.write_gpickle(G, pickle_path)

def load_features(feat_dir, normalize=True):
    print('Loading feature data...')
    data = np.genfromtxt(feat_dir, delimiter=',', skip_header=True, dtype=str)
    data = data[np.argsort(data[:, 13])]
    features = np.array(np.delete(data[:,1:], [11, 12, 13, 14, 15], 1), dtype=float)
    if normalize:
        features = F.normalize(torch.Tensor(features), dim=0)
    uris = data[:, 14]
    uris = [re.sub('spotify:track:', '', uri) for uri in uris]
    uri_map = {n: i for i,n in enumerate(uris)}
    print('Feature data shape: ' + str(features.shape))

    return features, uri_map

def load_graph(gpickle_dir, create_graph_from_scratch, uri_map):
    print('Loading graph data...')
    if create_graph_from_scratch:
        get_gpickle('./data/playlists', 'Spotify Playlist', gpickle_dir)

    G = nx.read_gpickle(gpickle_dir)
    print('Graph Info:\n', nx.info(G))

    src, dest = [], []
    weights = []
    for e in G.edges.data('weight'):
        uri_u, uri_v, w = e
        u, v = uri_map[uri_u], uri_map[uri_v]
        src.append(u)
        dest.append(v)
        w = G[uri_u][uri_v]['weight']
        weights.append(w)
  
    #make double edges
    src, dest = torch.tensor(src), torch.tensor(dest)
    src, dest = torch.cat([src, dest]), torch.cat([dest, src])
    dgl_G = dgl.graph((src, dest), num_nodes=len(G.nodes))
    
    #store edge weights in graph
    weights = torch.FloatTensor(weights+weights)
    dgl_G.edata['weights'] = weights
    
    return dgl_G, weights

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

def compute_loss(pos_score, neg_score, cuda):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    if cuda:
        labels = labels.to('cuda:0')
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def edge_coordinate(batch_nodes, adj_list, neg=False):
    if not neg:
        neigh_dict = {n:adj_list[n] for n in batch_nodes}
    else:
        neigh_dict = {n:adj_list[n]^set(batch_nodes) for n in batch_nodes}
    src = [k for k in neigh_dict.keys() for n in neigh_dict[k]]
    dest = [n for v in neigh_dict.values() for n in v]
    
    return src, dest