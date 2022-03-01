from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train_test(X, adj_list, int_to_label):

    print('Running KNN on node2vec embeddings')
    n = 50

    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    print('Calculate recommender precision and recall')
    levels = [50]
    precision = [0]
    recall = [0]
    preds = []
    labels = []
    for i in range(len(indices)):
        node = indices[i]
        dists = distances[i]
        size = len(adj_list[i])
        for j in range(len(levels)):
            sim = set(node[1:levels[j] + 1])
            inter = sim.intersection(adj_list[i])
            precision[j] += (len(inter) / len(sim))
            recall[j] += (len(inter) / size)

            min_dist = dists[1]
            preds += [min_dist/dists[d] for d in range(1, 51)]
            labels += [1 for _ in range(min(50, size))] + [0 for _ in range(50 - min(50, size))]


    for i in range(len(levels)):
        print('Precision and Recall at K = ' + str(levels[i]))
        print('Average Precision: ', precision[i] / len(indices))
        print('Average Recall: ', recall[i] / len(indices))
        print('AUC: ' + str(roc_auc_score(labels, preds, average='weighted')))

        fpr, tpr, _ = roc_curve(labels,  preds)

        #create ROC curve
        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('node2vec_knn_roc_curve.png')
    
