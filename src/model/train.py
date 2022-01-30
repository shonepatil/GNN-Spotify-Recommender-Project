from sklearn.neighbors import NearestNeighbors
import numpy as np

def train_test(X, adj_list, int_to_label):

    print('Running KNN on node2vec embeddings')
    n = 1000

    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    print('Calculate recommender precision and recall')
    precision = 0
    recall = 0
    for i in range(len(indices)):
        node = indices[i]
        size = min(n, len(adj_list[i]))
        sim = set(node[1:size + 1])
        inter = sim.intersection(adj_list[i])
        precision += (len(inter) / len(sim))
        recall += (len(inter) / size)

    print('Average Precision: ', precision / len(indices))
    print('Average Recall: ', recall / len(indices))
        
