import torch
import torch.nn as nn
from torch.nn.functional import normalize

class GraphSage(nn.Module):
    def __init__(self, features, feature_dim, 
            embed_dim, adj_list, aggregator,
            num_sample=10,
            gcn=False, cuda=False, 
            feature_transform=False): 
        super(GraphSage, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_list = adj_list
        self.aggregator = aggregator
        self.num_sample = num_sample

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_list[int(node)] for node in nodes], 
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        return normalize(combined, dim=0) # num_nodes x embed_dim


class Link_Prediction(nn.Module):

    def __init__(self, enc):
        super(Link_Prediction, self).__init__()
        self.enc = enc
        self.criterion = nn.BCELoss() #Binary Cross Entropy loss
        self.fc1 = nn.Linear(in_features=28, out_features=20) 
        self.fc2 = nn.Linear(in_features=20, out_features=enc.embed_dim)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        hidden =nn.LeakyReLU()(self.fc1(embeds))
        out =nn.LeakyReLU()(self.fc2(hidden))
        adj = out.mm(out.t()) #dot product decoder
        adj_prob = nn.Sigmoid()(adj) #map to probabilities
        return adj_prob
    
    def loss(self, nodes, labels):
        """
        @param labels: ground truth adjacency matrix
        """
        out = self.forward(nodes)
        return self.criterion(out, labels)