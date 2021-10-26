import copy
import math
import random
import numpy as np

from sklearn.preprocessing import StandardScaler


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
    indices = list(range(n_samples))
    random.seed(1111)
    random.shuffle(indices)
    split_point = int(n_samples * train_val_split_ratio)
    train_indices, valid_indices = indices[:split_point], indices[split_point:]
    if sample_size_cap is not None:
        train_indices = indices[:min(split_point, sample_size_cap)]

    return  train_indices, valid_indices 

# def powerlaw(sample_indices, n_participants, alpha=1.65911332899, shuffle=False):
#     # the smaller the alpha, the more extreme the division
#     if shuffle:
#         random.seed(1234)
#         random.shuffle(sample_indices)

#     from scipy.stats import powerlaw
#     import math
#     party_size = int(len(sample_indices) / n_participants)
#     b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_participants)
#     shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
#     indices_list = []
#     accessed = 0
#     for participant_id in range(n_participants):
#         indices_list.append(sample_indices[accessed:accessed + shard_sizes[participant_id]])
#         accessed += shard_sizes[participant_id]
#     return indices_list

def scale_normal(datasets, datasets_test):
    """
        Scale both the training and test set to standard normal distribution. The training set is used to fit.
        Args:
            datasets (list): list of datasets of length M
            datasets_test (list): list of test datasets of length M
        
        Returns:
            two lists containing the standarized training and test dataset
    """
    
    scaler = StandardScaler()
    scaler.fit(torch.vstack(datasets))
    return [torch.from_numpy(scaler.transform(dataset)).float() for dataset in datasets], [torch.from_numpy(scaler.transform(dataset)).float() for dataset in datasets_test]

def get_synthetic_datasets(n_participants, d=1, sizes=[], s=50, ranges=None):
    """
        Args:
            n_participants (int): number of data subsets to generate
            d (int): dimension
            sizes (list of int): number of data samples for each participant, if supplied
            s (int): number of data samples for each participant (equal), if supplied
            ranges (list of list): the lower and upper bound of the input domain for each participant, if supplied

        Returns:
            list containing the generated synthetic datasets for all participants
    """

    if 0 == len(sizes): 
        sizes = torch.ones(n_participants, dtype=int) * s

    datasets = []
    for i, size in enumerate(sizes):
        if ranges != None:
            dataset = torch.rand((size, d)) * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        else:
            dataset = torch.rand((size, d)) * (1 - 0) + 0 
            # dataset = np.random.uniform(0, 1, (size, d))
            # dataset = np.random.normal(0, 1, (size,d))
        datasets.append(dataset.reshape(-1, d))
    return datasets

def generate_linear_labels(datasets, d=1, weights=None, bias=None):

    # generate random true weights and the bias
    if weights is None:
        weights = torch.normal(0, 1, size=(d,))
    if bias is None:
        bias = torch.normal(0, 1, size=(1,))


    labels = []
    w_b = torch.cat((weights, bias))
    for X in datasets:
        one_padded_X = torch.cat((X, torch.ones((len(X), 1))), axis=1)
        y = (one_padded_X @ w_b).reshape(-1, 1)
        labels.append(y)
    return labels, weights, bias


def friedman_function(X, noise_std=0):
    """
    Create noisy friedman values: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html

    """
    assert X.shape[1] >= 5, "The input features must have at least 5 dimensions."
    M = len(X)
    y = 10 * torch.sin(math.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] 
    return y.reshape(M, 1)

import torch
from botorch.test_functions.synthetic import Hartmann

def hartmann_function(X, noise_std=0.05):
    """
    Create noisy Hartmann values: https://www.sfu.ca/~ssurjano/hart4.html

    """

    (M, dim) = X.shape
    assert dim in (3, 4, 6), "Hartmann function of dimensions: (3,4,6) is implemented."

    neg_hartmann = Hartmann(dim=dim, negate=True)
    y = neg_hartmann(X)
    return y.reshape(M, 1)