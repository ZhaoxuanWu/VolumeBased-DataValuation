### Selection of Discretization Coefficeint \omega Experiement
### Appendix B.3

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
from volume import replicate, compute_volumes, compute_pinvs, compute_X_tilde_and_counts, compute_robust_volumes

# ---------- DATA PREPARATION ----------

# ---------- CONFIGS ----------
function = 'hartmann'

n_participants = M = 3
D = 6
train_sizes = [200, 200, 200]
test_sizes = [200] * M

superset = False

size = True
if size:
    train_sizes = [20, 50, 200]

disjoint = False
ranges = [[0,1/3], [1/3, 2/3], [2/3, 1]] if disjoint else None

runs = 100
# -----------------------------

def shapley_volume(Xs, omega=0.1):
        M = len(Xs)
        orderings = list(permutations(range(M)))

        s_values = torch.zeros(M)
        monte_carlo_s_values = torch.zeros(M)

        s_value_robust = torch.zeros(M)
        monts_carlo_s_values_robust = torch.zeros(M)

        # Monte-carlo : shuffling the ordering and taking the first K orderings
        random.shuffle(orderings)
        K = 4 # number of permutations to sample
        for ordering_count, ordering in enumerate(orderings):

            prefix_vol = 0
            prefix_robust_vol = 0
            for position, i in enumerate(ordering):

                curr_indices = set(ordering[:position+1])

                curr_train_X = torch.cat([dataset for j, dataset in enumerate(Xs) if j in curr_indices]).reshape(-1, D)

                curr_vol = torch.sqrt(torch.linalg.det(curr_train_X.T @ curr_train_X) + 1e-8)


                marginal = curr_vol - prefix_vol
                prefix_vol = curr_vol
                s_values[i] += marginal

                X_tilde, cubes = compute_X_tilde_and_counts(curr_train_X, omega)

                robust_vol = compute_robust_volumes([X_tilde], [cubes])[0]


                marginal_robust = robust_vol - prefix_robust_vol
                s_value_robust[i] += marginal_robust
                prefix_robust_vol = robust_vol

                if ordering_count < K:
                    monte_carlo_s_values[i] += marginal

                    monts_carlo_s_values_robust[i] += marginal_robust

        s_values /= factorial(M)
        s_value_robust /= factorial(M)
        monte_carlo_s_values /= K
        monts_carlo_s_values_robust /= K

        # print('------Volume-based Shapley value Statistics ------')
        # # print("alpha : {}, omega : {}.".format(alpha, omega))
        # print("Volume-based Shapley values:", s_values)
        # print("Robust Volume Shapley values:", s_value_robust)
        # print("Volume-based MC-Shapley values:", monte_carlo_s_values)
        # print("Robust Volume MC-Shapley values:", monts_carlo_s_values_robust)
        # print('-------------------------------------')
        return s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust

omega_res_rvsvs = []
for run in range(runs):
    print('{}/{}'.format(run, runs))
    
    # Reproducebility
    seed = 1234 + run
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D, ranges=ranges)
    feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D, ranges=ranges)

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
    
    if superset:
        # Create dataset such that party i is superset of party i-1
        feature_datasets_ = copy.deepcopy(feature_datasets)
        labels_ = copy.deepcopy(labels)
        
        for i in range(1, len(feature_datasets)):
            feature_datasets_[i] = torch.cat((feature_datasets[i], feature_datasets_[i-1]), axis=0)
            labels_[i] = torch.cat((labels[i], labels_[i-1]), axis=0)

        feature_datasets, labels = feature_datasets_, labels_

    feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
    labels, test_labels = scale_normal(labels, test_labels)

    # ---------- DATA VALUATIONS ----------
    res = {}

    omega_res_rvsv = []
    omega_upper = 0.5
    for omega in np.linspace(0.001, omega_upper, 30):
        Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets))
        Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
        robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
        s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=omega)
        rvsv = np.array(s_value_robust)
        rvsv[rvsv < 0] = 0
        omega_res_rvsv.append(rvsv/np.sum(rvsv))
    omega_res_rvsv = np.array(omega_res_rvsv)
    omega_res_rvsvs.append(omega_res_rvsv)
    
suffix = 'disjoint' if disjoint else '' + 'size' if size else '' + 'superset' if superset else ''
suffix += 'iid' if suffix == '' else ''

np.savez('../outputs/omega_exp_{}_normal_{}.npz'.format(suffix, omega_upper), omega_res_rvsvs=omega_res_rvsvs)