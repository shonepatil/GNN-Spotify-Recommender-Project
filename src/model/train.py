from sklearn.neighbors import NearestNeighbors
import numpy as np

def train_test(X, adj_list, int_to_label):

    print('Running KNN on node2vec embeddings')
    n = 500

    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    print('Calculate recommender accuracy')
    accuracies = 0
    for i in range(len(indices)):
        node = indices[i]
        sim = set(node[1:])
        inter = sim.intersection(adj_list[i])
        accuracies += (len(inter) / min(n, len(adj_list[i])))

    print('Average Accuracy: ', accuracies / len(indices))
        
