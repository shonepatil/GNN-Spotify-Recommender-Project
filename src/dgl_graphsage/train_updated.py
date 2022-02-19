from dgl.dataloading import negative_sampler
import numpy as np
import dgl
from graphsage import GraphSAGE
from pred import MLPPredictor
import torch 
import itertools
import time
from utils import compute_loss, compute_auc
from sklearn.metrics import classification_report, roc_auc_score

def eid_neg_sampling(G, neg_sampler):
    s, d = G.all_edges(form='uv', order='srcdst')
    unique_idx = np.unique(s, return_index=True)[1] #index of unique nodes
    s, d = s[unique_idx], d[unique_idx]
    sample_eids = G.edge_ids(s, d)  
   
    return neg_sampler(G, sample_eids)

def train(G, weights, features, cuda, feat_dim, emb_dim, test_data, k=5):
    np.random.seed(1)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    model = GraphSAGE(feat_dim, emb_dim)
    pred = MLPPredictor(emb_dim)

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
    t1 = time.perf_counter()
    print('starting edge sampling')
    val_s_neg, val_d_neg = eid_neg_sampling(val_g, neg_sampler)
    print('finished edges sampling')
    t2 = time.perf_counter()
    print(t2-t1)
    val_s_neg, val_d_neg = torch.cat([val_s_neg, val_d_neg]), torch.cat([val_d_neg, val_s_neg])
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
            t1 = time.perf_counter()
            print('getting batch size nodes from train graph')
            batch_nodes = torch.randperm(len(train))[:batch_size]  
            print('done getting batch size nodes from train graph')
            t2 = time.perf_counter()
            print(t2-t1)
            if cuda:
                batch_nodes = batch_nodes.to('cuda:0')
            start_time = time.time()
            #Use original node ids to extract features for train graph
            embed = model(train_g, features[train_g.ndata[dgl.NID]], weights) 

            #construct pos and neg graph for batch
            t1 = time.perf_counter()
            print('pos and neg graph for batch construction')
            src, dest = train_g.out_edges(batch_nodes, form='uv')
            src, dest = torch.cat([src, dest]), torch.cat([dest, src])
            train_pos_g = dgl.graph((src, dest), num_nodes=train_g.number_of_nodes())
            
            t11 = time.perf_counter()
            print('within batch eid_neg_sampling')
            src_neg, dest_neg = eid_neg_sampling(train_g, neg_sampler)
            t12 = time.perf_counter()
            print(t12 - t11)
            print('within batch eid_neg_sampling done')
            
            src_neg, dest_neg = torch.cat([src_neg, dest_neg]), torch.cat([dest_neg, src_neg])
            train_neg_g = dgl.graph((src_neg, dest_neg), num_nodes=train_g.number_of_nodes())
            print('finished pos and neg graph for batch construction')
            t2 = time.perf_counter()
            print(t2-t1)
            if cuda:
                train_pos_g = train_pos_g.to('cuda:0')
                train_neg_g = train_neg_g.to('cuda:0')
            
            pos_score = pred(train_pos_g, embed, edges=(src,dest))
            
            
            
            neg_score = pred(train_neg_g, embed, edges=(src_neg, dest_neg))
            
            
            
            
            loss = compute_loss(pos_score, neg_score, cuda)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            
            print(f'end batch {batch}')
            print('-----')
            print(t2-t1)
            if batch % 5 == 0:
                print('In epoch {} batch {}, loss: {}'.format(epoch+1, batch+1, loss))
        
        print()
        with torch.no_grad():
            if cuda:
                val_g = val_g.to('cuda:0')
                val_neg_g = val_neg_g.to('cuda:0')
            print('getting embeddings at end of batch')
            z = model(val_g, features[val_g.ndata[dgl.NID]], weights)
            print('got embeddings')
            print('making positive and negative predictions on validation')
            pos = pred(val_g, z)
            neg = pred(val_neg_g, z)
            print('positive and negative predictions made')
            
            scores = torch.cat([pos, neg])
            labels = torch.cat(
                [torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])])
            prediction = scores >= 0
            print('running classification report')
            print(classification_report(labels, prediction))
            print('Epoch {} AUC: '.format(epoch+1), roc_auc_score(labels, scores, average='weighted'))
            print('done running classification report')
    return model, pred, losses