import os
import networkx as nx
import re
import torch
import torch.nn as nn
from utils import load_data
from train import train


def main():
    G = nx.read_gpickle('graph_170k.gpickle')
    print('Graph loaded')
    num_nodes = len(G.nodes)
    
    feat_data, adj_list = load_data(G, './features/merged_features.csv')
    #labels = adj_matrix(adj_list)
    features = nn.Embedding(num_nodes, feat_data.shape[1]) #features as look-up table
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    print('Features loaded')
    
    train(14, 10, G, features, adj_list)

if __name__ == "__main__":
    os.chdir('..')
    main()