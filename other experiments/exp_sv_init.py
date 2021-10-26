### Shapley Value of Empty Set Experiment
### Section 5.2, Figure 4

import random

import numpy as np
import torch
import copy

from itertools import permutations
from math import factorial

import sys
sys.path.insert(0, '..')
from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
from data_utils import load_used_car, load_uber_lyft, load_credit_card
from volume import replicate

# Reproducebility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------- DATA PREPARATION ----------

# ---------- CONFIGS ----------
function = 'hartmann'

n_participants = M = 3
D = 6
train_sizes = [200, 200, 200]
test_sizes = [200] * M

# -----------------------------

feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D)
feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D)

if function == 'friedman':
    friedman_labels, friedman_noisy_labels, friedman_test_labels = [], [], []
    assert D >= 5 
    if D >= 5:
        for X in feature_datasets:
            friedman_y = friedman_function(X)
            friedman_labels.append(friedman_y)

            friedman_y_noisy = friedman_y + torch.randn(friedman_y.shape) * 0.05
            friedman_noisy_labels.append(friedman_y_noisy)

        for X in feature_datasets_test:
            friedman_test_labels.append(friedman_function(X))
    labels, test_labels = friedman_noisy_labels, friedman_test_labels
elif function == 'hartmann':
    hartmann_labels, hartmann_noisy_labels, hartmann_test_labels = [], [], []
    assert D in (3, 4, 6)
    if D in (3, 4, 6):
        for X in feature_datasets:
            hartmann_y = hartmann_function(X)
            hartmann_labels.append(hartmann_y)

            hartmann_y_noisy = hartmann_y + torch.randn(hartmann_y.shape) * 0.0005
            hartmann_noisy_labels.append(hartmann_y_noisy)

        for X in feature_datasets_test:
            hartmann_test_labels.append(hartmann_function(X))
    labels, test_labels = hartmann_noisy_labels, hartmann_test_labels
else:
    raise NotImplementedError('Function not implemented.')
    
feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
labels, test_labels = scale_normal(labels, test_labels)

# ---------- DATA VALUATIONS ----------

test_X = torch.cat(feature_datasets_test).reshape(-1, D)
test_y = torch.cat(test_labels).reshape(-1, 1)

# Random initilaization, needed to compute marginal against empty set
random_init = torch.normal(0, 1, (D, 1))
init_test_loss = (torch.norm( test_X @ random_init - test_y )) ** 2 / test_X.shape[0]

# # Zero initialization
# zero_init_test_loss = (torch.norm(test_y )) ** 2 / test_X.shape[0]

init_range = np.linspace(0, init_test_loss, 100)


"""
test loss-based Shapley values

"""

res = []

for init in init_range:
    
    orderings = list(permutations(range(M)))

    s_values = np.zeros(M)
    monte_carlo_s_values = np.zeros(M)

    # Monte-carlo : shuffling the ordering and taking the first K orderings
    np.random.shuffle(orderings)
    K = 4 # number of permutations to sample

    # Truncated monte-carlo: in addition to MC, truncate the marginal contribution calculation when loss is within tolerance
    train_X = torch.cat(feature_datasets).reshape(-1, D)
    train_y = torch.cat(labels).reshape(-1, 1)

    test_X = torch.cat(feature_datasets_test).reshape(-1, D)
    test_y = torch.cat(test_labels).reshape(-1, 1)

    pinv = torch.pinverse(train_X)
    test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2 / test_X.shape[0]

    truncated_mc_s_values = np.zeros(M)

    # Bootstrap 10 percent samples for 10 times to calculate tol
    percentage = 0.1
    bootstrap_times = 10
    data_size = np.sum([len(test_label) for test_label in test_labels])
    bootstrap_indices = torch.rand(bootstrap_times, data_size).argsort(1)[:,:int(data_size * percentage)]
    bootstrap_test_X = test_X[bootstrap_indices]
    bootstrap_test_y = test_y[bootstrap_indices]
    bootstrap_errors = (bootstrap_test_X @ pinv @ train_y - bootstrap_test_y)
    bootstrap_losses = np.apply_along_axis(lambda error: np.linalg.norm(error) ** 2, 0, bootstrap_errors) * (1/percentage)
    tol = np.std(bootstrap_losses)/np.sqrt(bootstrap_times) # This is to be determined through performance vairation in bootstrap samples (standard error)

    # # Random initilaization, needed to compute marginal against empty set
    # random_init = torch.normal(0, 1, (D, 1))
    # init_test_loss = (torch.norm( test_X @ random_init - test_y )) ** 2
    init_test_loss = init

    for ordering_count, ordering in enumerate(orderings):

        prefix_pinvs = []
        prefix_test_losses = []
        for position, i in enumerate(ordering):

            curr_indices = set(ordering[:position+1])

            curr_train_X = torch.cat([dataset for j, dataset in enumerate(feature_datasets) if j in curr_indices ]).reshape(-1, D)
            curr_train_y = torch.cat([label for j, label in enumerate(labels)  if j in curr_indices ]).reshape(-1, 1)

            # curr_pinv_ = np.linalg.pinv(curr_train_X)
            curr_pinv = torch.pinverse(curr_train_X)
            curr_test_loss = (torch.norm(test_X @ curr_pinv @ curr_train_y - test_y ))**2 / test_X.shape[0]

            if position == 0: # first in the ordering
                marginal = init_test_loss - curr_test_loss         
            else:
                marginal = prefix_test_losses[-1] - curr_test_loss
            s_values[i] += marginal

            prefix_pinvs.append(curr_pinv)
            prefix_test_losses.append(curr_test_loss)

            if ordering_count < K:
                monte_carlo_s_values[i] += marginal
                
                if np.abs(test_loss - curr_test_loss) > tol or ordering_count == 0:
                    truncated_mc_s_values[i] += marginal

    s_values /= factorial(M)
    monte_carlo_s_values /= K
    truncated_mc_s_values /= K
    # print('------Test loss Shapley value Statistics ------')
    # print("Test loss-based Shapley values:", s_values)
    # print("Test loss-based MC-Shapley values:", monte_carlo_s_values)
    # print("Test loss-based TMC-Shapley values:", truncated_mc_s_values)
    # print('-------------------------------------')
    if np.sum(s_values < 0) > 0:
        s_values = s_values + np.min(s_values)
    s_values = s_values / np.sum(s_values)
    res.append(s_values)
    

np.savez('../outputs/sv_init_exp.npz', res=res, init_range=init_range)
