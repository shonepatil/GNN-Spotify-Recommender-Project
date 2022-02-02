import random
import time
import numpy as np
import dgl
import torch
import itertools
import gc
from graphsage import GraphSAGE
from pred import MLPPredictor, DotPredictor
from utils import edge_coordinate, compute_loss, compute_auc

def train(G, features, adj_list, cuda, feat_dim, emb_dim):
    np.random.seed(1)
    random.seed(1)
    num_nodes = G.number_of_nodes()

    model = GraphSAGE(feat_dim, emb_dim)
    pred = MLPPredictor(emb_dim)

    rand_indices = np.random.permutation(num_nodes)
    test = list(rand_indices[:34000])
    # val = list(rand_indices[34000:51000])
    val = list(rand_indices[34000:39000])
    train = list(rand_indices[51000:])
    
    train_g = dgl.remove_edges(G, val+test)
    val_pos_g = dgl.graph(edge_coordinate(val, adj_list), num_nodes=train_g.number_of_nodes())
    val_neg_g = dgl.graph(edge_coordinate(val, adj_list, neg=True), num_nodes=train_g.number_of_nodes())

    print('Cuda enabled: ' + str(cuda))
    if cuda:
        model.cuda()
        features = features.to('cuda:0')
        pred = pred.to('cuda:0')
        train_g = train_g.to('cuda:0')

    print('Training starts:')

    optimizer = torch.optim.SGD(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)                   
    losses = []
    for epoch in range(10):
        print("\nStart of epoch %d" % (epoch,))

        for batch in range(101):
            batch_nodes = train[:4000]
            random.shuffle(train)  
            start_time = time.time()
            embed = model(train_g, features)
            
            train_pos_g = dgl.graph(edge_coordinate(batch_nodes, adj_list), num_nodes=train_g.number_of_nodes())
            train_neg_g = dgl.graph(edge_coordinate(batch_nodes, adj_list,neg=True), num_nodes=train_g.number_of_nodes())
            if cuda:
                train_pos_g = train_pos_g.to('cuda:0')
                train_neg_g = train_neg_g.to('cuda:0')
            pos_score = pred(train_pos_g, embed)
            neg_score = pred(train_neg_g, embed)
            loss = compute_loss(pos_score, neg_score, cuda)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            if batch % 20 == 0:
                print('In batch {}, loss: {}'.format(batch, loss))

    with torch.no_grad():
        if cuda:
            pos = pred(val_pos_g.to('cuda:0'), embed).cpu()
            neg = pred(val_neg_g.to('cuda:0'), embed).cpu()
        else:
            pos = pred(val_pos_g, embed).cpu()
            neg = pred(val_neg_g, embed).cpu()

        print('AUC', compute_auc(pos, neg))
