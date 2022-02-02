import torch
import torch.nn.functional as F
from collections import defaultdict

def recommend(pred, seeds, embed, g):
    """
    For each seed track s given in the playlist, get the predicted scores of all tracks c 
    where (s,c) is an edge in the graph
    
    @param pred: link predictor
    @param seeds: seed tracks in a playlist
    @embed: graphsage embedding of all songs
    @g: dgl graph 
    """
    
    score_dict = defaultdict(dict)
    for s in seeds:
        _, candidates = g.out_edges(s, form='uv')
        s_embed = embed[s].unsqueeze(dim=0)
        edge_embeds = [torch.cat([s_embed, embed[c.item()].unsqueeze(dim=0)],1) for c in candidates]
        edge_embeds = torch.cat(edge_embeds, 0)
        scores = pred.W2(F.relu(pred.W1(edge_embeds))).squeeze(1)
        score_dict[s] = list(zip(candidates.detach().numpy(), scores.detach().numpy())) 
    return score_dict