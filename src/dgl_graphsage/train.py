import random
import time
import numpy as np
import dgl
import torch
import itertools
from graphsage import GraphSAGE
from pred import DotPredictor
from utils import edge_coordinate, compute_loss, compute_auc

def train(feat_dim, emb_dim, G, features, adj_list):
    np.random.seed(1)
    random.seed(1)
    num_nodes = G.number_of_nodes()

    model = GraphSAGE(feat_dim, emb_dim)
    pred = DotPredictor()
#   model.cuda()

    rand_indices = np.random.permutation(num_nodes)
    test = list(rand_indices[:34000])
    val = list(rand_indices[34000:51000])
    train = list(rand_indices[51000:])
    
    train_g = dgl.remove_edges(G, val+test)
    val_pos_g = dgl.graph(edge_coordinate(val), num_nodes=train_g.number_of_nodes())
    val_neg_g = dgl.graph(edge_coordinate(val,neg=True), num_nodes=train_g.number_of_nodes())
    print('Training starts:')

    optimizer = torch.optim.SGD(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)                   
    losses = []
    for batch in range(100):
        batch_nodes = train[:3000]
        random.shuffle(train)  
        start_time = time.time()
        embed = model(train_g, features)
        
        train_pos_g = dgl.graph(edge_coordinate(batch_nodes), num_nodes=train_g.number_of_nodes())
        train_neg_g = dgl.graph(edge_coordinate(batch_nodes,neg=True), num_nodes=train_g.number_of_nodes())
        pos_score = pred(train_pos_g, embed)
        neg_score = pred(train_neg_g, embed)
        loss = compute_loss(pos_score, neg_score)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            print('In epoch {}, loss: {}'.format(batch, loss))

            with torch.no_grad():
                pos = pred(val_pos_g, embed)
                neg = pred(val_neg_g, embed)
                print('AUC', compute_auc(pos, neg))