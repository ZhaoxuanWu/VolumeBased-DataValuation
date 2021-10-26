import random

import numpy as np
import torch
import copy

from itertools import permutations
from math import factorial

from main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
from data_utils import load_used_car, load_uber_lyft, load_credit_card, load_hotel_reviews
from volume import replicate

# Reproducebility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------- DATA PREPARATION ----------

# ---------- CONFIGS ----------
# function choices: 'linear', 'friedman', 'hartmann', 'used_car', 'uber_lyft', 'credit_card', 'hotel_reviews'
function = 'hartmann'

n_participants = M = 3
D = 6
train_sizes = [200, 200, 200]
test_sizes = [200] * M

size = False

disjoint = False

rep = False
rep_factors = [1, 2, 10] if rep else [1,] * M

superset = False

train_test_diff_distr = False
# -----------------------------

if size:
    train_sizes = [20, 50, 200]
    
ranges = [[0,1/3], [1/3, 2/3], [2/3, 1]] if disjoint else None

feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D, ranges=ranges)
feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D, ranges=ranges)

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
elif function == 'used_car':
    assert D == 5
    s = 50
    train_sizes = [50] * M
    train_sizes = []
    feature_datasets, labels, feature_datasets_test, test_labels = load_used_car(n_participants=M, s=300, train_test_diff_distr=train_test_diff_distr)
elif function == 'uber_lyft':
    assert D == 12
    feature_datasets, labels, feature_datasets_test, test_labels = load_uber_lyft(n_participants=M, s=300, reduced=True)
elif function == 'credit_card':
    assert D == 8
    feature_datasets, labels, feature_datasets_test, test_labels = load_credit_card(n_participants=M, s=50, train_test_diff_distr=train_test_diff_distr)
elif function == 'hotel_reviews':
    assert D == 8
    feature_datasets, labels, feature_datasets_test, test_labels = load_hotel_reviews(n_participants=M, s=30)
else:
    raise NotImplementedError('Function not implemented.')

if rep:
    feature_datasets_ = copy.deepcopy(feature_datasets)
    labels_ = copy.deepcopy(labels)
    
    for i in range(len(feature_datasets)):
        if rep_factors[i] == 1:
            continue
        to_replicate = torch.cat((feature_datasets[i], labels[i]), axis=1)
        replicated = replicate(to_replicate, c=rep_factors[i])
        feature_datasets_[i] = replicated[:,:-1]
        labels_[i] = replicated[:, -1:]
    
    feature_datasets, labels = feature_datasets_, labels_

if superset:
    # Create dataset such that party i is superset of party i-1
    feature_datasets_ = copy.deepcopy(feature_datasets)
    labels_ = copy.deepcopy(labels)
    
    for i in range(1, len(feature_datasets)):
        feature_datasets_[i] = torch.cat((feature_datasets[i], feature_datasets_[i-1]), axis=0)
        labels_[i] = torch.cat((labels[i], labels_[i-1]), axis=0)

    feature_datasets, labels = feature_datasets_, labels_

# Standardize features to standard normal
feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
labels, test_labels = scale_normal(labels, test_labels)

# ---------- DATA VALUATIONS ----------
res = {}

"""
Direct Volume-based values

"""
from volume import compute_volumes, compute_pinvs, compute_X_tilde_and_counts, compute_robust_volumes

# train_features = [dataset.data for dataset in train_datasets]
volumes, vol_all = compute_volumes(feature_datasets, D)

volumes_all = np.asarray(list(volumes) + [vol_all])
print('-------Volume Statistics ------')


print("Original volumes: ", volumes, "volume all:", vol_all)
res['vol'] = volumes

"""
Discretized Robust Volume-based Shapley values
"""
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
    print("Volume-based Shapley values:", s_values)
    print("Robust Volume Shapley values:", s_value_robust)
    print("Volume-based MC-Shapley values:", monte_carlo_s_values)
    print("Robust Volume MC-Shapley values:", monts_carlo_s_values_robust)
    print('-------------------------------------')
    return s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust


feature_datasets_include_all = copy.deepcopy(feature_datasets) + [torch.vstack(feature_datasets) ]

# s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=0.5, alpha=alpha)
s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=0.1)

# omega_res_rv = []
# omega_res_rvsv = []
# omega_res_vsv = []
# omega_upper = 0.5
# for omega in np.linspace(0.001,omega_upper,30):
#     Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets))
#     Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
#     robust_volumes = compute_robust_volumes(Xtildes, dcube_collections)
#     rv = np.array(robust_volumes)
#     omega_res_rv.append(rv/np.sum(rv))
#     s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust = shapley_volume(feature_datasets, omega=omega)
#     rvsv = np.array(s_value_robust)
#     rvsv[rvsv < 0] = 0
#     omega_res_rvsv.append(rvsv/np.sum(rvsv))
#     vsv = np.array(s_values)
#     omega_res_vsv.append(vsv/np.sum(vsv))
# omega_res_rv = np.array(omega_res_rv)
# omega_res_rvsv = np.array(omega_res_rvsv)
# omega_res_vsv = np.array(omega_res_vsv)
# np.savez('outputs/omega_exp_disjoint_normal_{}.npz'.format(omega_upper), omega_res_rv=omega_res_rv, omega_res_rvsv=omega_res_rvsv, omega_res_vsv=omega_res_vsv)


res['vol_sv'], res['vol_sv_robust'] = s_values, s_value_robust
res['vol_mc_sv'], res['vol_mc_sv_robust'] = monte_carlo_s_values, monts_carlo_s_values_robust

'''
Code for calculating RV

Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_include_all))

Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)
robust_volumes = compute_robust_volumes(Xtildes, dcube_collections, alpha)

print("Robust volumes: {} with omega {}".format( robust_volumes, omega) )

omega = 0.25
Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_include_all))
Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)

robust_volumes = compute_robust_volumes(Xtildes, dcube_collections, alpha)
print("Robust volumes: {} with omega {}".format( robust_volumes, omega) )


omega = 0.1
Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_include_all))
Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)

robust_volumes = compute_robust_volumes(Xtildes, dcube_collections, alpha)
print("Robust volumes: {} with omega {}".format( robust_volumes, omega) )

omega = 0.01
Xtildes, dcube_collections = zip(*(compute_X_tilde_and_counts(dataset, omega=omega) for dataset in feature_datasets_include_all))
Xtildes, dcube_collections = list(Xtildes), list(dcube_collections)

robust_volumes = compute_robust_volumes(Xtildes, dcube_collections, alpha)
print("Robust volumes: {} with omega {}".format( robust_volumes, omega) )


print('-------------------------------------')
'''

"""
Leave-one-out OLS

"""

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

res['loo'] = loo_values

"""
test loss-based Shapley values

"""

from itertools import permutations
from math import factorial

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

res['loss_sv'], res['loss_mc_sv'], res['loss_tmc_sv'] = s_values, monte_carlo_s_values, truncated_mc_s_values


"""
Information theoretic data valuation

"""
from scipy.stats import sem
import random
have_gp = True
if have_gp:
    from gpytorch_ig import compute_IG, fit_model

    trials = 5

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
                # curr_train_y = torch.cat([label for j, label in enumerate(labels)  if j in curr_indices ]).reshape(-1, 1)
                # curr_train_y = curr_train_y.squeeze()
                # curr_train_X, curr_train_y = torch.from_numpy(curr_train_X), torch.from_numpy(curr_train_y).squeeze()

                # NO NEED TO RETRAIN
                # model, likelihood = fit_model(curr_train_X, curr_train_y)
                # curr_IG = compute_IG(all_train_X, model, likelihood)

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
    print('-------------------------------------')
    
    res['ig_sv'], res['ig_mc_sv'] = torch.mean(s_values_IG_trials, 0), torch.mean(mc_s_values_IG_trials, 0)

have_spgp = False
if have_spgp:
    from gpytorch_ig import compute_IG, fit_model

    trials = 5

    s_values_IG_trials = []
    mc_s_values_IG_trials = []

    for t in range(trials):

        inducing_ratio = 0.25
        inducing_count = int(torch.sum(torch.tensor(train_sizes)) * inducing_ratio * M)

        end, begin = 1, 0
        # uniform distribution of inducing
        inducing_points = torch.rand((inducing_count, D)) * (end - begin) + begin 

        all_train_X = torch.cat(feature_datasets)
        all_train_y = torch.cat(labels).reshape(-1 ,1).squeeze()
        joint_model, joint_likelihood = fit_model(all_train_X, all_train_y, inducing_points=inducing_points)


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
                # curr_train_y = torch.cat([label for j, label in enumerate(labels)  if j in curr_indices ]).reshape(-1, 1)
                # curr_train_y = curr_train_y.squeeze()

                # model, likelihood = fit_model(curr_train_X, curr_train_y, inducing_points=inducing_points)
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

    print('------Information Gain SPGP Shapley value Statistics ------')
    print("SPGP IG-based Shapley values: mean {}, sem {}".format(torch.mean(s_values_IG_trials, 0), sem(s_values_IG_trials, axis=0)))
    print("SPGP IG-based MC-Shapley values: mean {}, sem {}".format(torch.mean(mc_s_values_IG_trials, 0), sem(mc_s_values_IG_trials, axis=0)))
    print('-------------------------------------')

    res['spgp_ig_sv'], res['spgp_ig_mc_sv'] = torch.mean(s_values_IG_trials, 0), torch.mean(mc_s_values_IG_trials, 0)


suffix = '_rep' if rep else '' + '_superset' if superset else '' + '_train_test_diff_distri' if train_test_diff_distr else '' + '_size' if size else '' + '_disjoint' if disjoint else ''

np.savez('outputs/res_{}_{}D_{}M{}.npz'.format(function, D, M, suffix), 
         res=res, M=M, D=D, train_sizes=train_sizes, test_sizes=test_sizes, function=function, seed=seed)