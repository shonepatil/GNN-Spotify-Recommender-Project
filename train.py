import random
import time
import numpy as np
import torch
import torch.nn as nn
from utils import make_label
from aggregators import MeanAggregator
from model import GraphSage
from model import Link_Prediction


def train(feat_dim, emb_dim, G, features, adj_list):
    np.random.seed(1)
    random.seed(1)
    num_nodes = len(G.nodes)

    agg = MeanAggregator(features)
    enc = GraphSage(features, feat_dim, emb_dim, adj_list, agg, gcn=False, cuda=False)
    enc.num_samples = 20

    model = Link_Prediction(enc)
#   model.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:5000]
    val = rand_indices[5000:8000]
    train = list(rand_indices[8000:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:10000]
        random.shuffle(train)  
        start_time = time.time()
        optimizer.zero_grad()
        labels = make_label(batch_nodes, adj_list)
        loss = model.loss(batch_nodes, labels)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print("Batch {} - Training loss: {}".format(batch, loss.item()))

    #val_output = model.forward(val) 
    #print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))