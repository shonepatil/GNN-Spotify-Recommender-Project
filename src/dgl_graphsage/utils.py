import re
import random
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.functional import normalize
import torch
import dgl
import networkx as nx
import json
from recommend import recommend_no_repeat
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score

import sys
sys.path.insert(0, './src/features')
from features.build_features import load_data as graph_from_scratch
sys.path.insert(1, '..')
sys.path.insert(0, './src/api')
from api.spotifyAPI_script import pull_audio_features
from api.songset_processor import build_songset_csv

def get_gpickle(data_path, dataset_name, pickle_path, playlist_num):
    G = graph_from_scratch(data_path, dataset_name, playlist_num)
    nx.write_gpickle(G, pickle_path)
    return G.number_of_nodes()

def load_features(feat_dir, gpickle_dir, create_graph_from_scratch, playlist_num=10000, normalize=True):

    if create_graph_from_scratch:
        num_nodes = get_gpickle('./data/playlists/', 'Spotify Playlist', gpickle_dir, playlist_num)

        # Run spotify API script to create features
        print('Pulling spotify song data using SpotifyAPI')
        pull_audio_features(num_nodes)

        print('Building songset features csv')
        build_songset_csv(feat_dir, num_nodes)

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

def load_graph(gpickle_dir, uri_map):
    print('Loading graph data...')

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

def metrics(dgl_G, target_set, model, pred, feat_data):
    weights = dgl_G.edata['weights']
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k=5)
    
    g = dgl.node_subgraph(dgl_G, target_set)
    embeddings = model(g, feat_data[g.ndata[dgl.NID]], weights)
    
    s_neg, d_neg = eid_neg_sampling(g, neg_sampler)
    s_neg, d_neg = torch.cat([s_neg, d_neg]), torch.cat([d_neg, s_neg])
    neg_g = dgl.graph((s_neg, d_neg), num_nodes=g.number_of_nodes())
    
    pos = pred(g, embeddings)
    neg = pred(neg_g, embeddings)

    scores = torch.cat([pos, neg]).detach().numpy()
    labels = torch.cat(
                [torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])])
    prediction = scores >= 0

    auc = roc_auc_score(labels, scores, average='weighted')
    report = classification_report(labels, prediction, zero_division=1)
    precision, recall, f_score = precision_recall_fscore_support(labels, prediction, average='weighted')
    
    return auc, precision, recall, f_score, report

def r_precision(tracks, dgl_G, z, pred, neigh, feat_data, uri_map):
    seed_len = len(tracks)//2
    seeds = tracks[:seed_len]
    masked = tracks[seed_len:]
    rec_uris = recommend_no_repeat(seeds, dgl_G, z, pred, neigh, feat_data, uri_map)
    relevant = len(set(masked).intersection(set(uri_recs)))
    r = relevant / len(masked)
    return r


def r_precision_analysis(total, start, end, dgl_G, z, pred, neigh, feat_data, uri_map, data_dir):
    start, end = 20000, start+1000
    total = 50000
    rs = []
    seed_size = []
    ks = []
    avg_deg = []
        
    while end != total+1000:
        slice_path = data_dir+'/mpd.slice.%s.json'%(str(start)+'-'+str(end-1))
        with open(slice_path, 'r') as f:
            mpd_slice = json.load(f)
        playlists = mpd_slice['playlists']
        print('playlist %s'%(str(start)+'-'+str(end-1)))
        for i in range(50):
            k = random.randint(0, len(playlists))
            ks.append(k)
            tracks = pd.DataFrame(playlists[k]['tracks'])
            track_uris = tracks['track_uri'].apply(lambda x: re.sub('spotify:track:', '', x))

            seed_len = len(track_uris)//2
            seeds = track_uris[:seed_len]
            masked = track_uris[seed_len:]
            seed_ids = [uri_map[i] for i in seeds]
            avg_deg.append(np.mean([dgl_G.out_degrees(i) for i in seed_ids]))

            uri_recs = recommend_no_repeat(seeds, dgl_G, z, pred, neigh, feat_data, uri_map)
            relevant = len(set(masked).intersection(set(uri_recs)))
            r = relevant / len(masked)
            print('k={}, seed_size={}, r={}'.format(k, seed_len, r))
            rs.append(r)
            seed_size.append(seed_len)
        start+=1000
        end+=1000
        print()