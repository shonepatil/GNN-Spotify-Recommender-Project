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
        if num_nodes > 35000:
            test = list(rand_indices[:34000])
            val = list(rand_indices[34000:51000])
            train = list(rand_indices[51000:])
            batch_size = 5000
        else:
            test = list(rand_indices[:5000])
            val = list(rand_indices[5000:12000])
            train = list(rand_indices[12000:])
            batch_size = 3000
    
    #Construct train and validation graph
    train_g = dgl.node_subgraph(G, train)
    val_g = dgl.node_subgraph(G, val)
    
    #Construct neg validation graph
    #For each unique src node s in validation graph, sample a dest node d and get the eid of (s,d) 
    #to pass in neg_sampler
    #5 negative dest node for each unique node in graph
    val_s_neg, val_d_neg = eid_neg_sampling(val_g, neg_sampler)
    val_s_neg, val_d_neg = eid_neg_sampling(val_g, neg_sampler)

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
        weights = weights.to('cuda:0')
    print()
    print('Training starts:')

   
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    losses = []
    batch_per_epoch = len(train) // batch_size
    
    measures = {'epoch': [], 'loss': [], 'auc': [], 'report': []}
    for epoch in range(2):
        for batch in range(batch_per_epoch):
            #randomly sample batch size nodes from train graph
    
            batch_nodes = torch.randperm(len(train))[:batch_size]  
            
            if cuda:
                batch_nodes = batch_nodes.to('cuda:0')
            start_time = time.time()
            #Use original node ids to extract features for train graph
            embed = model(train_g, features[train_g.ndata[dgl.NID]], weights) 
            #construct pos and neg graph for batch
            embed = model(train_g, features[train_g.ndata[dgl.NID]], weights)
            #construct pos and neg graph for batch
            src, dest = train_g.out_edges(batch_nodes, form='uv')
            src, dest = torch.cat([src, dest]), torch.cat([dest, src])
            train_pos_g = dgl.graph((src, dest), num_nodes=train_g.number_of_nodes())
            
            src_neg, dest_neg = eid_neg_sampling(train_g, neg_sampler)
            src_neg, dest_neg = eid_neg_sampling(train_g, neg_sampler)

            src_neg, dest_neg = torch.cat([src_neg, dest_neg]), torch.cat([dest_neg, src_neg])
            train_neg_g = dgl.graph((src_neg, dest_neg), num_nodes=train_g.number_of_nodes())
            
            if cuda:
                train_pos_g = train_pos_g.to('cuda:0')
                train_neg_g = train_neg_g.to('cuda:0')
            pos_score = pred(train_pos_g, embed, edges=(src,dest))
            
            
            
            neg_score = pred(train_neg_g, embed, edges=(src_neg, dest_neg))
            
            pos_score = pred(train_pos_g, embed, edges=(src,dest))
            neg_score = pred(train_neg_g, embed, edges=(src_neg, dest_neg))

            loss = compute_loss(pos_score, neg_score, cuda)
            losses.append(loss)
            measures['loss'].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            
            print('-----')
            print('In epoch {} batch {}, loss: {}'.format(epoch+1, batch+1, loss))
        
        print()
        with torch.no_grad():
            if cuda:
                val_g = val_g.to('cuda:0')
                val_neg_g = val_neg_g.to('cuda:0')

                z = model(val_g, features[val_g.ndata[dgl.NID]], weights)

                pos = pred(val_g, z)
                neg = pred(val_neg_g, z)


                scores = torch.cat([pos, neg])
                labels = torch.cat(
                    [torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])])
                prediction = scores >= 0
                
                auc = roc_auc_score(labels, scores, average='weighted')
                report = classification_report(labels, prediction)
                print(report)
                print('Epoch {} AUC: '.format(epoch+1), auc)
                measures['epoch'].append(epoch+1)
                measures['auc'].append(auc.item())
                measures['report'].append(report)
            else:
                z = model(val_g, features[val_g.ndata[dgl.NID]], weights)
                pos = pred(val_g, z)
                neg = pred(val_neg_g, z)

                scores = torch.cat([pos, neg]).cpu()
                labels = torch.cat(
                    [torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).cpu()
                prediction = scores >= 0
                
                auc = roc_auc_score(labels, scores, average='weighted')
                report = classification_report(labels, prediction, zero_division=1)
                print(report)
                print('Epoch {} AUC: '.format(epoch+1), auc)
                measures['epoch'].append(epoch+1)
                measures['auc'].append(auc.item())
                measures['report'].append(report)

    return model, pred, measures