from dgl.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat, weights):
        h = self.conv1(g, in_feat, edge_weight=weights[g.edata[dgl.EID]])
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=weights[g.edata[dgl.EID]])
        return h