from dgl.dataloading import negative_sampler
import numpy as np
import dgl
from graphsage import GraphSAGE
from pred import MLPPredictor
from pred import DotPredictor
import torch 
import itertools
import time
from utils import compute_loss, compute_auc

def eid_neg_sampling(G, neg_sampler):
    s, d = G.all_edges(form='uv', order='srcdst')
    unique_idx = np.unique(s, return_index=True)[1] #index of unique nodes
    s, d = s[unique_idx], d[unique_idx]
    sample_eids = G.edge_ids(s, d)  
    return neg_sampler(G, sample_eids)

def train(G, features, adj_list, cuda, feat_dim, emb_dim, test_data, k=5):
    np.random.seed(1)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    model = GraphSAGE(feat_dim, emb_dim)
    pred = MLPPredictor(emb_dim)
    #pred = DotPredictor()
    rand_indices = np.random.permutation(num_nodes)
    if test_data:
        test = list(rand_indices[:2])
        val = list(rand_indices[2:3+2])
        train = list(rand_indices[3+2:])
        batch_size = 5
    else:
        test = list(rand_indices[:34000])
        val = list(rand_indices[34000:51000])
        train = list(rand_indices[51000:])
        batch_size = 5000
    
    #Construct train and validation graph
    train_g = dgl.node_subgraph(G, train)
    val_g = dgl.node_subgraph(G, val)
    
    #Construct neg validation graph
    #For each unique src node s in validation graph, sample a dest node d and get the eid of (s,d) 
    #to pass in neg_sampler
    #5 negative dest node for each unique node in graph
    val_s, val_d = val_g.all_edges(form='uv', order='srcdst')
    unique_idx = np.unique(val_s, return_index=True)[1] #index of unique nodes
    val_s, val_d = val_s[unique_idx], val_d[unique_idx]
    sample_eids = val_g.edge_ids(val_s, val_d)  
    val_s_neg, val_d_neg = neg_sampler(val_g, sample_eids)
    val_neg_g = dgl.graph((val_s_neg, val_d_neg), num_nodes=val_g.number_of_nodes())
    
    print('Train pos edge: {}'.format(train_g.number_of_edges()))
    print('Validation pos edge: {}'.format(val_g.number_of_edges()))
    print('Cuda enabled: ' + str(cuda))
    if cuda:
        model.cuda()
        pred = pred.to('cuda:0')
        features = features.to('cuda:0')
        train_g = train_g.to('cuda:0')
    print()
    print('Training starts:')

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    losses = []
    batch_per_epoch = len(train) // batch_size
    for epoch in range(10):
        for batch in range(batch_per_epoch):
            #randomly sample batch size nodes from train graph
            batch_nodes = torch.randperm(len(train))[:batch_size]  
            if cuda:
                batch_nodes = batch_nodes.to('cuda:0')
            start_time = time.time()
            #Use original node ids to extract features for train graph
            embed = model(train_g, features[train_g.ndata[dgl.NID]]) 

            #construct pos and neg graph for batch
            src, dest = train_g.out_edges(batch_nodes, form='uv') 
            train_pos_g = dgl.graph((src, dest), num_nodes=train_g.number_of_nodes())
            
            unique_idx = np.unique(src.cpu(), return_index=True)[1] #index of unique nodes
            sample_eids = train_g.edge_ids(src[unique_idx], dest[unique_idx])  
            src_neg, dest_neg = neg_sampler(train_g, sample_eids)
            train_neg_g = dgl.graph((src_neg, dest_neg), num_nodes=train_g.number_of_nodes())

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
            end_time = time.time()
            
            
            if batch % 5 == 0:
                print('In epoch {} batch {}, loss: {}'.format(epoch+1, batch+1, loss))
        
        print()
        with torch.no_grad():
            if cuda:
                val_g = val_g.to('cuda:0')
                val_neg_g = val_neg_g.to('cuda:0')

            z = model(val_g, features[val_g.ndata[dgl.NID]])
            
            pos = pred(val_g, z).cpu()
            neg = pred(val_neg_g, z).cpu()
            print('Epoch {} AUC: '.format(epoch+1), compute_auc(pos, neg))

    return model, pred, train, val_g, val_neg_g, embed