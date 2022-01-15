from operator import xor
import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from fastnode2vec import Graph, Node2Vec

import sys
sys.path.insert(0, 'src/model')
from src.model.utils import encode_onehot, frac_mat_power

def load_data(path, dataset, train, val, test, include_ad_hoc_feat=False, include_node2vec=False):
    """Load network dataset"""
    print('Loading {} dataset...'.format(dataset))

    # Construct graph
    G = nx.Graph(name = 'G')
    
    # Load data
    # Choose how many playlists to load in
    n = 10000
    count = 0
    print('Loading {} playlists'.format(n))
    for i in range(999, n, 1000):
        with open(path + 'mpd.slice.{}-{}.json'.format(i - 999, i)) as f:
            data = json.load(f)
            playlists = data['playlists']
            count += create_nodes_edges(G, playlists)

    print('Playlist co-occurences: ' + str(count))

    # See top co-occurence pairs of songs
    # print(sorted(G.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)[:5])

    # See graph info
    print('Graph Info:\n', nx.info(G))
    # A = nx.to_scipy_sparse_matrix(G)
    
    # Get Node Features Matrix (X)
    X_start = pd.read_csv('./data/a13group1/features/merged_features.csv')
    uris = [i[14:] for i in X_start['track_uri']]
    X = X_start.iloc[:, 2:].drop(['type'], axis=1).values

    # Set up Node2Vec Embeddings
    if include_node2vec:
        g = []
        for edge in nx.convert.to_edgelist(G):
            last = (edge[0], edge[1], edge[2]['weight'])
            g.append(last)

        graph = Graph(g, directed=False, weighted=True)
        n2v = Node2Vec(graph, dim=32, walk_length=10, context=10, p=2.0, q=0.5, workers=4, seed=10)
        n2v.train(epochs=200, progress_bar=True)
        emb = [n2v.wv[node] for node in uris]
        X = np.c_[X, np.array(emb)].astype(float)

    # Standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # X = torch.FloatTensor(X)

    # Show stats
    print('\nShape of X: ', X.shape)
    print('\nNode Features Matrix (X):\n', X)

    # Dump to csv
    X_df = pd.DataFrame(X)
    X_df['track_uri'] = uris
    X_df.to_csv('./data/node2vec_merged_features.csv')


# Create graph nodes and edges
def create_nodes_edges(G, playlists):
    count = 0
    for p in playlists:
        tracks = p['tracks']

        # keep track of pairs in current playlist
        seen = set()

        for i in range(len(tracks)): # create nodes
            first = tracks[i]['track_uri'][14:]

            for j in range(i + 1, len(tracks)):
                second = tracks[j]['track_uri'][14:]

                if first != second: # create edges
                    # Ensures co-occurence increased only if edge pair is found in new playlist
                    if not G.has_edge(first, second):
                        G.add_edge(first, second, weight=1)
                        seen.add(frozenset([first, second]))
                    elif frozenset([first, second]) not in seen:
                        G[first][second]['weight'] += 1
                        seen.add(frozenset([first, second]))
                        count += 1

    return count
