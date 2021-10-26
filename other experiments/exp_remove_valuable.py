### Robust Volume and Learning Performance Experiment
### Appendix B.4

import random

import numpy as np
import torch
import copy

from itertools import permutations
from math import factorial

import sys
sys.path.insert(0, '..')
from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
from data_utils import load_uber_lyft, load_credit_card, load_hotel_reviews, load_used_car

from volume import compute_volumes, compute_X_tilde_and_counts, compute_robust_volumes

# ---------- DATA PREPARATION ----------

# ---------- CONFIGS ----------
function = 'used_car' # 'credit_card' or 'uber_lyft' or 'hotel_reviews' or 'used_car'
exp_type = 'remove' # 'remove' or 'add'

n_participants = M = 8
s = 50
D = 5
omega = 0.1

runs = 50
# -----------------------------

def compute_losses_high_or_low_remove(feature_datasets, labels, feature_datasets_test, test_labels, M, omega, high_or_low):
    
    original_train_X = torch.cat(feature_datasets).reshape(-1, D)
    original_train_y = torch.cat(labels).reshape(-1, 1)
    test_X = torch.cat(feature_datasets_test).reshape(-1, D)
    test_y = torch.cat(test_labels).reshape(-1, 1)
    
    feature_datasets_, labels_, feature_datasets_test_, test_labels_ = copy.deepcopy(feature_datasets), copy.deepcopy(labels), copy.deepcopy(feature_datasets_test), copy.deepcopy(test_labels)

    test_losses = []
    train_losses = []
        
    for i in range(M):
        # Calculate current loss
        train_X = torch.cat(feature_datasets_).reshape(-1, D)
        train_y = torch.cat(labels_).reshape(-1, 1)

        pinv = torch.pinverse(train_X)
        test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2 / test_X.shape[0]
        test_losses.append(test_loss.item())
        
        train_loss = (torch.norm( original_train_X @ pinv @ train_y - original_train_y )) ** 2 / original_train_X.shape[0]
        train_losses.append(train_loss.item())
        
        # Calculate robust volumes
        Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_))
        Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
        robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
        
        if high_or_low == 'high':
            idx = np.argmax(robust_volumes)
        elif high_or_low == 'low':
            idx = np.argmin(robust_volumes)
        elif high_or_low == 'random':
            idx = np.random.randint(low=0, high=robust_volumes.shape[0])
        else:
            raise NotImplementedError()
        
        _ = feature_datasets_.pop(idx)
        _ = labels_.pop(idx)
    
    return test_losses, train_losses

def compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega, high_or_low):
    original_train_X = torch.cat(feature_datasets).reshape(-1, D)
    original_train_y = torch.cat(labels).reshape(-1, 1)
    test_X = torch.cat(feature_datasets_test).reshape(-1, D)
    test_y = torch.cat(test_labels).reshape(-1, 1)
    
    feature_datasets_, labels_, feature_datasets_test_, test_labels_ = [], [], [], []
    feature_datasets_copy, labels_copy, feature_datasets_test_copy, test_labels_copy = copy.deepcopy(feature_datasets), copy.deepcopy(labels), copy.deepcopy(feature_datasets_test), copy.deepcopy(test_labels)
    
    test_losses = []
    train_losses = []
    
    for i in range(M):
        Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_copy))
        Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
        robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
        
        if high_or_low == 'high':
            idx = np.argmax(robust_volumes)
        elif high_or_low == 'low':
            idx = np.argmin(robust_volumes)
        elif high_or_low == 'random':
            idx = np.random.randint(low=0, high=robust_volumes.shape[0])
        else:
            raise NotImplementedError()
        
        feature_datasets_.append(feature_datasets_copy.pop(idx))
        labels_.append(labels_copy.pop(idx))
        
        # Calculate current loss
        train_X = torch.cat(feature_datasets_).reshape(-1, D)
        train_y = torch.cat(labels_).reshape(-1, 1)

        pinv = torch.pinverse(train_X)
        test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2 / test_X.shape[0]
        test_losses.append(test_loss.item())
        
        train_loss = (torch.norm( original_train_X @ pinv @ train_y - original_train_y )) ** 2 / original_train_X.shape[0]
        train_losses.append(train_loss.item())
    
    return test_losses, train_losses

high_test_arr = []
high_train_arr = []
low_test_arr = []
low_train_arr = []
rand_test_arr = []
rand_train_arr = []

for i in range(runs):
    print(i, "/", runs)
    # Reproducebility
    seed = 1234 + i
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if function == 'uber_lyft':
        assert D == 12
        feature_datasets, labels, feature_datasets_test, test_labels = load_uber_lyft(n_participants=M, s=s, reduced=False, path_prefix='../')
    elif function == 'credit_card':
        assert D == 8
        feature_datasets, labels, feature_datasets_test, test_labels = load_credit_card(n_participants=M, s=s, path_prefix='../')
    elif function == 'hotel_reviews':
        assert D == 8
        feature_datasets, labels, feature_datasets_test, test_labels = load_hotel_reviews(n_participants=M, s=s, path_prefix='../')
    elif function == 'used_car':
        assert D == 5
        feature_datasets, labels, feature_datasets_test, test_labels = load_used_car(n_participants=M, s=s, train_test_diff_distr=False, path_prefix='../')
    else:
        raise NotImplementedError('Function not implemented.')
    
    feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
    labels, test_labels = scale_normal(labels, test_labels)


    Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets))
    Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
    robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
    print("Robust volumes: {} with omega {}".format( robust_volumes, omega))

    if exp_type == 'remove':
        high_test, high_train = compute_losses_high_or_low_remove(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='high') 
        low_test, low_train = compute_losses_high_or_low_remove(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='low') 
        rand_test, rand_train = compute_losses_high_or_low_remove(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='random') 
    elif exp_type == 'add':
        high_test, high_train = compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='high') 
        low_test, low_train = compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='low') 
        rand_test, rand_train = compute_losses_high_or_low_add(feature_datasets, labels, feature_datasets_test, test_labels, M, omega=0.1, high_or_low='random') 
    
    high_test_arr.append(high_test)
    high_train_arr.append(high_train)
    low_test_arr.append(low_test)
    low_train_arr.append(low_train)
    rand_test_arr.append(rand_test)
    rand_train_arr.append(rand_train)

np.savez('../outputs/{}_valuable_{}_{}M.npz'.format(exp_type, function, M), 
         high_test=high_test_arr, high_train=high_train_arr, low_test=low_test_arr, low_train=low_train_arr, rand_test=rand_test_arr, rand_train=rand_train_arr,
         M=M, D=D, s=s, function=function)