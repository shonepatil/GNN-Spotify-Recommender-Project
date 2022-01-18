from operator import xor
import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fastnode2vec import Graph, Node2Vec
from itertools import combinations, islice
import random

import sys
sys.path.insert(0, 'src/model')
from src.model.utils import encode_onehot, frac_mat_power

def load_data(path, dataset, train, val, test, node2vec_from_scratch=False, graph_from_scratch=False):
    """Load network dataset"""
    print('Loading {} dataset...'.format(dataset))

    # Load graph
    if graph_from_scratch:
        G = build_graph_from_scratch(path, n = 10000)
    else:
        # Load from edgelist
        print('Loading graph from file')
        G = nx.read_edgelist("./data/a13group1/node2vec_features/170k_edgelist.csv", nodetype=str, data=(("weight", int),))

    # See graph info
    print('Graph Info:\n', nx.info(G))

    # Load node2vec features
    if node2vec_from_scratch:
        n2v_features = create_node2vec_embeddings(G)
    else:
        # Load from file
        print('Loading node2vec features from file')
        n2v_features = pd.read_csv('./data/a13group1/node2vec_features/node2vec_merged_features.csv', index_col='track_uri')\
            .drop(['Unnamed: 0'], axis=1)
    n2v_features = n2v_features.T.to_dict('list')

    # get node pairs which don't have an edge
    print('Creating negative edges')
    non_existing_edges = set()
    i = 0
    while i < (G.number_of_edges() // 3):
        u, v = random.sample(G.nodes(), 2)
        
        if not G.has_edge(u, v) and not G.has_edge(v, u) and (u, v) not in non_existing_edges:
            non_existing_edges.add((u, v))
            i += 1

    non_existing_edges = list(non_existing_edges)

    df1 = pd.DataFrame(data = non_existing_edges, columns =['Node 1', 'Node 2'])
 
    # create a column 'Connection' with default 0 (no-connection)
    df1['Connection'] = 0
    
        # get node pairs in fb dataframe with indices in removable_edges_indices
    df2 = pd.DataFrame(data = list(G.edges), columns =['Node 1', 'Node 2'])
    
    # create a column 'Connection' and assign default value of 1 (connected nodes)
    df2['Connection'] = 1

    total = df1.append(df2[['Node 1', 'Node 2', 'Connection']],
                ignore_index=True)

    print(total.head())

    y = total['Connection']

    X = np.array([(n2v_features[i]+n2v_features[j]) for i,j in zip(total['Node 1'], total['Node 2'])])

    print(len(y))
    print(len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train, test_size = test, random_state = 0)

    print(len(X_train))
    print(len(X_test))

    return X_train, X_test, y_train, y_test


# Create graph from raw data
def build_graph_from_scratch(path, n = 10000):
    # Construct graph
    G = nx.Graph(name = 'G')
    
    # Load data
    # Choose how many playlists to load in
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

    # Dump to csv
    nx.write_edgelist(G, "./data/10k_edgelist.csv", data=["weight"])

    return G

# Create node2vec embeddings
def create_node2vec_embeddings(G):
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

    # Show stats
    print('\nShape of X: ', X.shape)
    print('\nNode Features Matrix (X):\n', X)

    # Dump to csv
    X_df = pd.DataFrame(X)
    X_df['track_uri'] = uris
    X_df.to_csv('./data/node2vec_merged_features.csv')

    return X_df

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
