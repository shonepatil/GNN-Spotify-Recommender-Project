from train import train
import network as nx
from utils import load_data

def main(feat_dir, graph_dir, feat_dim, embed_dim):
    G = nx.read_gpickle(graph_dir)
    feat_data, adj_list, dgl_G = load_data(G, feat_dir)
    train(feat_dim, embed_dim, dgl_G, feat_data, adj_list)

if __name__ == "__main__":
    main()