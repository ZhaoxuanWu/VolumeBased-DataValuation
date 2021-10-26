### Simulation for Replication Experiments on Volume and RV
### Appendix B.2

import copy
import random
import numpy as np
import torch

import sys
sys.path.insert(0, '..')
from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal

# Reproducebility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

function = 'friedman'

n_participants = M = 3
D = 6
train_sizes = [200, 200, 200]
test_sizes = [200] * M

feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D)
feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D)

if function == 'linear':
    labels, true_weights, true_bias = generate_linear_labels(feature_datasets, d=D)
    test_labels, _, _ = generate_linear_labels(feature_datasets_test, d=D, weights=true_weights, bias=true_bias)
elif function == 'friedman':
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
    
feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
labels, test_labels = scale_normal(labels, test_labels)

"""
Direct Volume-based values

"""
from volume import compute_volumes, compute_pinvs, compute_X_tilde_and_counts, compute_robust_volumes

from volume import replicate, replicate_perturb

def get_robust_volumes(feature_datasets, omega):

    # N = sum([len(dataset) for dataset in feature_datasets])
    # alpha = 1.0 / (10 * N) # it means we set beta = 10

    feature_datasets_include_all = copy.deepcopy(feature_datasets) + [torch.vstack(feature_datasets)]
    X_tildes, dcube_collections = [], []
    for dataset in feature_datasets_include_all:
        X_tilde, dcubes = compute_X_tilde_and_counts(dataset, omega=omega)
        X_tildes.append(X_tilde)
        dcube_collections.append(dcubes)
    # cubes = dcube_collections[0]
    robust_volumes = compute_robust_volumes(X_tildes, dcube_collections)

    return robust_volumes


'''
1). robust to replication

'''

# 1.1 direct replication

# 1.2 0-mean noise perturbation

def replication_robustness_helper(feature_datasets, c=3, full=True, random=True, perturb=True, sigmas=[1e-1], omega=0.1):
    """
    
    """
    vol = []
    rob_vol = []

    D = feature_datasets[0].shape[1]
    volumes, vol_all = compute_volumes(feature_datasets, D)
    volumes_all = np.asarray(list(volumes) + [vol_all])
    print('-------  Volume Statistics ------')
    print("Original volumes: ", volumes, "volume all:", vol_all)
    print("Relative volumes: ", volumes / sum(volumes))

    """
    Discretized Robust Volume-based values
    """
    robust_volumes = get_robust_volumes(feature_datasets, omega)
    print("Robust volumes: ", robust_volumes)
    print("Relative Robust volumes: ", robust_volumes / sum(robust_volumes[:-1]))

    # Full Replication
    if full:
        feature_datasets_ = copy.deepcopy(feature_datasets)
        replicated = replicate(feature_datasets_[0], c=c)

        feature_datasets_[0] = replicated

        volumes, vol_all = compute_volumes(feature_datasets_, D)
        volumes_all = np.asarray(list(volumes) + [vol_all])
        print('-------  Volume Statistics with Replication ------')
        print("Volumes (with [0] replicated fully): ", volumes, "volume all:", vol_all)
        print("Relative volumes: ", volumes / sum(volumes))
        robust_volumes = get_robust_volumes(feature_datasets_, omega)
        print("Robust volumes: ", robust_volumes)
        print("Relative Robust volumes: ", robust_volumes / sum(robust_volumes[:-1]))
        vol.append(volumes[0])
        rob_vol.append(robust_volumes[0])

    if random:
    # Random Replication
        feature_datasets_ = copy.deepcopy(feature_datasets)
        replicated = replicate(feature_datasets_[0], c=c, mode='random')
        feature_datasets_[0] = replicated

        volumes, vol_all = compute_volumes(feature_datasets_, D)
        volumes_all = np.asarray(list(volumes) + [vol_all])
        print("Volumes (with [0] replicated randomly): ", volumes, "volume all:", vol_all)
        print("Relative volumes: ", volumes / sum(volumes))
        robust_volumes = get_robust_volumes(feature_datasets_, omega)
        print("Robust volumes: ", robust_volumes)
        print("Relative Robust volumes: ", robust_volumes / sum(robust_volumes[:-1]))

        print('-------------------------------------')
        vol.append(volumes[0])
        rob_vol.append(robust_volumes[0])


    if perturb:
        # Replication with noise
        for sigma in sigmas:
            feature_datasets_ = copy.deepcopy(feature_datasets)
            replicated = replicate_perturb(feature_datasets_[0], c=c, sigma=sigma)
            feature_datasets_[0] = replicated

            volumes, vol_all = compute_volumes(feature_datasets_, D)
            volumes_all = np.asarray(list(volumes) + [vol_all])
            print("Volumes (with [0] replicated with noise): ", volumes, "volume all:", vol_all)
            print("Relative volumes: ", volumes / sum(volumes))
            robust_volumes = get_robust_volumes(feature_datasets_, omega)
            print("Robust volumes: ", robust_volumes)
            print("Relative Robust volumes: ", robust_volumes / sum(robust_volumes[:-1]))
            print('-------------------------------------')
            
            vol.append(volumes[0])
            rob_vol.append(robust_volumes[0])
        
    return vol, rob_vol


replication_robustness_helper(feature_datasets, c=5, sigmas=[1e-2], omega=0.1)

runs = 10
vols, rob_vols = [], []
for c in range(2,10):
    t1, t2 = [], []
    for _ in range(runs):
        vol, rob_vol = replication_robustness_helper(feature_datasets, c=c, sigmas=[1e-2, 3e-2], omega=0.1)
        t1.append(vol)
        t2.append(rob_vol)
    # t1 = np.mean(np.array(t1), axis=0)
    # t2 = np.mean(np.array(t2), axis=0)
    vols.append(t1)
    rob_vols.append(t2)

vols = np.array(vols)
rob_vols = np.array(rob_vols)

np.savez('../outputs/replication_{}_{}D_{}M_{}runs.npz'.format(function, D, M, runs),
         vols=vols,
         rob_vols=rob_vols,
         M=M, D=D, train_sizes=train_sizes, test_sizes=test_sizes, function=function, seed=seed)
