import sys
import time
import json
import os
import recommend as r
from train_updated import train
from api.spotifyAPI import SpotifyAPI
from utils import load_graph
from utils import load_features
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from dgl import load_graphs
import matplotlib.pyplot as plt
import dgl
import torch
import numpy as np

# Loading Graph ~ 3min
feat_dir = "../../data/a13group1/460k_songset_features.csv"
scratch_pickle_dir = "../../data/a13group1/"
feat_data, uri_map = load_features(feat_dir, scratch_pickle_dir, False, playlist_num=100000)
graph_dir = ("../../data/a13group1/graph_460k.gpickle")
dgl_G, weights = load_graph(graph_dir, uri_map)

# Training the Model. GPU ~ 00:00:40. CPU ~ 00:53:00.
with open('../../config/model-params.json') as fh:
            model_cfg = json.load(fh)
model, pred, measures = train(dgl_G, weights.to('cpu'), feat_data, cuda=False, feat_dim=13, emb_dim=10, test_data=False)

# Put everything on CPU
model = model.to('cpu')
pred = pred.to('cpu')

torch.save(model, '460k_1epoch_model.pt')
torch.save(pred, '460k_1epoch_pred.pt')

with open("460k_1epoch_measures.json", "w") as out_measures:
    json.dump(measures, out_measures)