import random
import torch
import torch.nn as nn

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, gcn=False, cuda=False): 
        super(MeanAggregator, self).__init__()

        self.features = features #lookup table of features
        self.gcn = gcn  
        self.cuda = cuda
        
    def forward(self, nodes, to_neighs, num_sample=10):
        # Local pointers to functions (speed hack)
        _set = set
        _sample = random.sample
        #sampling
        #If neighbors less than intended num_sample, use all neighbors
        samp_neighs = [_set(_sample(to_neigh, num_sample,)) 
                       if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]

        if self.gcn: #self-loop: add self into samples
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes)) 
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True) 
        mask = mask.div(num_neigh) 
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix) #average sampled features
        return to_feats # num_nodes x feat_dim