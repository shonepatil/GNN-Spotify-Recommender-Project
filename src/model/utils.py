import numpy as np
import torch

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def frac_mat_power(m, n):
    evals, evecs = torch.eig (m, eigenvectors = True)  # get eigendecomposition
    evals = evals[:, 0]                                # get real part of (real) eigenvalues
    # rebuild original matrix
    mchk = torch.matmul (evecs, torch.matmul (torch.diag (evals), torch.inverse (evecs)))
    mchk - m                                           # check decomposition
    evpow = evals**(n)                              # raise eigenvalues to fractional power
    # build exponentiated matrix from exponentiated eigenvalues
    mpow = torch.matmul (evecs, torch.matmul (torch.diag (evpow), torch.inverse (evecs)))
    return mpow

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    