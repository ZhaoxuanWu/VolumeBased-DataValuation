### Overlap of Input Domains Experiement
### Appendix B.5

import random

import numpy as np
import torch
import copy

from scipy.stats import sem
import random

from itertools import permutations
from math import factorial

import sys
sys.path.insert(0, '..')
from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
from data_utils import load_used_car, load_uber_lyft, load_credit_card
from volume import replicate
from volume import compute_volumes, compute_pinvs, compute_X_tilde_and_counts, compute_robust_volumes

from gpytorch_ig import compute_IG, fit_model

import time

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

omega = 0.1
runs = 10
overlaps = 10
have_ig = True
res = {}

methods = ['vol', 'robust_vol', 'vol_sv', 'vol_sv_robust', 'loo', 'loss_sv', 'ig_sv']
for method in methods:
    res[method] = np.zeros((runs, overlaps, M))

# -----------------------------

def get_synthetic_datasets(n_participants, d=1, sizes=[], s=50, overlap=0):

    if 0 == len(sizes): 
        sizes = torch.ones(n_participants, dtype=int) * s

    datasets = []
    for i, size in enumerate(sizes):
        if i == 2:
            scale = 0.5
            lower = 0.5
        else:
            scale = 0.5 + overlap
            lower = 0

        dataset = torch.rand((size, d)) * (scale - 0) + lower
        # dataset = np.random.uniform(0, 1, (size, d))
        # dataset = np.random.normal(0, 1, (size,d))
        datasets.append(dataset.reshape(-1, d))
    return datasets

def calculate_relative(arr):
    value = np.array(arr)
    if np.sum(value < 0) > 0:
        value[value < 0] = 0
    value = value/np.sum(value)
    
    return value

start_time = time.time()

for run in range(runs):
    
    seed = 1234 + run
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    for idx, overlap in enumerate(np.linspace(0, 0.5, overlaps)):
        feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D, overlap=overlap)
        feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D, overlap=overlap)

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
        
        volumes, vol_all = compute_volumes(feature_datasets, D)

        volumes_all = np.asarray(list(volumes) + [vol_all])
        print('-------Volume Statistics ------')

        print("Original volumes: ", volumes, "volume all:", vol_all)
        res['vol'][run, idx] = calculate_relative(volumes)


        Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets))

        Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
        robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)

        print("Robust volumes: {} with omega {}".format( robust_volumes, omega) )
        res['robust_vol'][run, idx] = calculate_relative(robust_volumes)

        import random
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

            print('------Volume-based Shapley value Statistics ------')
            # print("alpha : {}, omega : {}.".format(alpha, omega))
            print("Volume-based Shapley values:", s_values)
            print("Robust Volume Shapley values:", s_value_robust)
            print("Volume-based MC-Shapley values:", monte_carlo_s_values)
            print("Robust Volume MC-Shapley values:", monts_carlo_s_values_robust)
            print('-------------------------------------')
            return s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust



        feature_datasets_include_all = copy.deepcopy(feature_datasets) + [torch.vstack(feature_datasets) ]

        # s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=0.5, alpha=alpha)
        s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=0.1)

        res['vol_sv'][run, idx] = calculate_relative(s_values)
        res['vol_sv_robust'][run, idx] = calculate_relative(s_value_robust)
        # res['vol_mc_sv'], res['vol_mc_sv_robust'] = monte_carlo_s_values, monts_carlo_s_values_robust




        train_X = torch.cat(feature_datasets).reshape(-1, D)
        train_y = torch.cat(labels).reshape(-1, 1)

        test_X = torch.cat(feature_datasets_test).reshape(-1, D)
        test_y = torch.cat(test_labels).reshape(-1, 1)

        pinv = torch.pinverse(train_X)
        test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2 / test_X.shape[0]

        loo_values = []
        loo_losses = []
        for i in range(M):
            loo_dataset = []
            loo_label = []

            for j, (dataset, label) in enumerate(zip(feature_datasets, labels)):
                if i == j: continue

                loo_dataset.append(dataset)
                loo_label.append(label)
            
            loo_dataset = torch.cat(loo_dataset).reshape(-1, D)
            loo_label = torch.cat(loo_label).reshape(-1, 1)

            pinv = torch.pinverse(loo_dataset)

            loo_test_loss = (torch.norm(  test_X @ pinv @ loo_label - test_y ))**2 / test_X.shape[0]
            loo_losses.append(loo_test_loss.item())
            loo_values.append((loo_test_loss -  test_loss).item())

        print('------Leave One Out Statistics ------')
        print("Full test loss:", test_loss.item())
        print("Leave-one-out test losses:", loo_losses)
        print("Leave-one-out (loo_loss - full_loss) :", loo_values)
        print('-------------------------------------')

        res['loo'][run, idx] = calculate_relative(loo_values)



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

        # Random initilaization, needed to compute marginal against empty set
        random_init = torch.normal(0, 1, (D, 1))
        init_test_loss = (torch.norm( test_X @ random_init - test_y )) ** 2 / test_X.shape[0]

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
        print('------Test loss Shapley value Statistics ------')
        print("Test loss-based Shapley values:", s_values)
        print("Test loss-based MC-Shapley values:", monte_carlo_s_values)
        print("Test loss-based TMC-Shapley values:", truncated_mc_s_values)
        print('-------------------------------------')

        res['loss_sv'][run, idx] = calculate_relative(s_values)
    
        if have_ig:
            trials = 2

            s_values_IG_trials = []
            mc_s_values_IG_trials = []

            for t in range(trials):
                all_train_X = torch.cat(feature_datasets)
                all_train_y = torch.cat(labels).reshape(-1 ,1).squeeze()
                joint_model, joint_likelihood = fit_model(all_train_X, all_train_y)


                s_values_IG = torch.zeros(M)
                monte_carlo_s_values_IG = torch.zeros(M)

                orderings = list(permutations(range(M)))
                # Monte-carlo : shuffling the ordering and taking the first K orderings
                random.shuffle(orderings)
                K = 4 # number of permutations to sample

                for ordering_count, ordering in enumerate(orderings):

                    prefix_IGs = []
                    for position, i in enumerate(ordering):

                        curr_indices = set(ordering[:position+1])

                        curr_train_X = torch.cat([dataset for j, dataset in enumerate(feature_datasets)  if j in curr_indices ]).reshape(-1, D)
                        curr_IG = compute_IG(curr_train_X, joint_model, joint_likelihood)

                        if position == 0: # first in the ordering
                            marginal = curr_IG  - 0
                        else:
                            marginal = curr_IG - prefix_IGs[-1] 
                        s_values_IG[i] += marginal
                        prefix_IGs.append(curr_IG)

                        if ordering_count < K:
                            monte_carlo_s_values_IG[i] += marginal

                s_values_IG /= factorial(M)
                monte_carlo_s_values_IG /= K

                s_values_IG_trials.append(s_values_IG)
                mc_s_values_IG_trials.append(monte_carlo_s_values_IG)

            s_values_IG_trials = torch.stack(s_values_IG_trials)
            mc_s_values_IG_trials = torch.stack(mc_s_values_IG_trials)


            print('------Information Gain Shapley value Statistics ------')
            print("IG-based Shapley values: mean {}, sem {}".format(torch.mean(s_values_IG_trials, 0), sem(s_values_IG_trials, axis=0)))
            print("IG-based MC-Shapley values: mean {}, sem {}".format(torch.mean(mc_s_values_IG_trials, 0), sem(mc_s_values_IG_trials, axis=0)))
    
            res['ig_sv'][run, idx] = calculate_relative(torch.mean(s_values_IG_trials, 0))
            
        
        print('Finished {}/{} in {} seconds.'.format(run*overlaps + (idx+1), runs*overlaps, time.time() - start_time))
    
np.savez('../outputs/overlap_exp.npz', res=res)
