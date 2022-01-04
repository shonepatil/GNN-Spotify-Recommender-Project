from operator import xor
import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, 'src/model')
from src.model.utils import encode_onehot, frac_mat_power

def load_data(path, dataset, train, val, test, include_ad_hoc_feat=False, include_node2vec=False):
    """Load network dataset"""
    print('Loading {} dataset...'.format(dataset))
    
    # Load data

    # Construct graph
    G = nx.Graph(name = 'G')

    # Create nodes

    # Create edges

    #See graph info
    print('Graph Info:\n', nx.info(G))
    
    #Get the Adjacency Matrix (A) and Node Features Matrix (X) as numpy array

    # Include Ad-Hoc graph variables

    # Standardize X
