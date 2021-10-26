# Bayesian Linear Regression 
# Information Gain on the parameters

import os
from os.path import join as oj
import math
from math import factorial as fac
import random

from scipy import stats
import numpy as np
from numpy.linalg import slogdet
import pandas as pd

import itertools
import matplotlib.pyplot as plt

from sklearn import metrics, model_selection
import torch
from torch import rand, randn, cat, stack


from blr import BayesLinReg, BatchBayesLinReg, shapley_volume, compute_IG_SVs


def test_loss_LOO(Xs, ys, test_X, test_y):
    M = len(Xs)
    D = Xs[0].shape[1]

    Xs = torch.cat(Xs).reshape(-1, D)
    ys = torch.cat(ys).reshape(-1, 1)


    pinv = torch.pinverse(Xs)
    test_loss = (torch.norm( test_X @ pinv @ ys - test_y )) ** 2

    loo_values = []
    loo_losses = []
    for i in range(M):
        loo_dataset = []
        loo_label = []

        for j, (dataset, label) in enumerate(zip(Xs, ys)):
            if i == j: continue

            loo_dataset.append(dataset)
            loo_label.append(label)
        
        loo_dataset = torch.cat(loo_dataset).reshape(-1, D)
        loo_label = torch.cat(loo_label).reshape(-1, 1)

        pinv = torch.pinverse(loo_dataset)

        loo_test_loss = (torch.norm(  test_X @ pinv @ loo_label - test_y ))**2
        loo_losses.append(loo_test_loss.item())
        loo_values.append((loo_test_loss -  test_loss).item())

    # print('------Leave One Out Statistics ------')
    # print("Full test loss:", test_loss.item())
    # print("Leave-one-out test losses:", loo_losses)
    # print("Leave-one-out (loo_loss - full_loss) :", loo_values)
    # print('-------------------------------------')

    return np.asarray(loo_values)

from itertools import permutations
from math import factorial

def test_loss_SV(Xs, ys, test_X, test_y):

    M = len(Xs)
    D = Xs[0].shape[1]
    orderings = list(permutations(range(M)))

    s_values = np.zeros(M)
    monte_carlo_s_values = np.zeros(M)

    # Monte-carlo : shuffling the ordering and taking the first K orderings
    np.random.shuffle(orderings)
    K = 4 # number of permutations to sample

    # Truncated monte-carlo: in addition to MC, truncate the marginal contribution calculation when loss is within tolerance
    train_X = torch.cat(Xs).reshape(-1, D)
    train_y = torch.cat(ys).reshape(-1, 1)

    pinv = torch.pinverse(train_X)
    test_loss = (torch.norm( test_X @ pinv @ train_y - test_y )) ** 2

    truncated_mc_s_values = np.zeros(M)

    # Bootstrap 10 percent samples for 10 times to calculate tol
    percentage = 0.1
    bootstrap_times = 10
    # data_size = np.sum([len(test_label) for test_label in test_labels])
    data_size = len(test_y)
    bootstrap_indices = torch.rand(bootstrap_times, data_size).argsort(1)[:,:int(data_size * percentage)]
    bootstrap_test_X = test_X[bootstrap_indices]
    bootstrap_test_y = test_y[bootstrap_indices]
    bootstrap_errors = (bootstrap_test_X @ pinv @ train_y - bootstrap_test_y)
    bootstrap_losses = np.apply_along_axis(lambda error: np.linalg.norm(error) ** 2, 0, bootstrap_errors) * (1/percentage)
    tol = np.std(bootstrap_losses)/np.sqrt(bootstrap_times) # This is to be determined through performance vairation in bootstrap samples (standard error)

    # Random initilaization, needed to compute marginal against empty set
    random_init = torch.normal(0, 1, (D, 1)).double()

    init_test_loss = (torch.norm( test_X @ random_init - test_y )) ** 2

    for ordering_count, ordering in enumerate(orderings):

        prefix_pinvs = []
        prefix_test_losses = []
        for position, i in enumerate(ordering):

            curr_indices = set(ordering[:position+1])

            curr_train_X = torch.cat([dataset for j, dataset in enumerate(Xs) if j in curr_indices ]).reshape(-1, D)
            curr_train_y = torch.cat([label for j, label in enumerate(ys)  if j in curr_indices ]).reshape(-1, 1)

            # curr_pinv_ = np.linalg.pinv(curr_train_X)
            curr_pinv = torch.pinverse(curr_train_X)
            curr_test_loss = (torch.norm(test_X @ curr_pinv @ curr_train_y - test_y ))**2

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

    return s_values, monte_carlo_s_values, truncated_mc_s_values


import random
import torch
from math import factorial as fac
import itertools
import pandas as pd
import matplotlib.pyplot as plt


# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.preprocessing import StandardScaler, minmax_scale, MinMaxScaler


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for name in ['CaliH', 'KingH', 'USCensus', 'FaceA'][3:]:

        # -- set up data --
        if name == 'CaliH':
            X = pd.read_csv('data/California_housing/CaliH-NN_features.csv')
            X = X.drop(columns=['7']) # column 7 is all 0, so we drop it upon inspection
            y = pd.read_csv('data/California_housing/CaliH-labels.csv').values        

        elif name == 'KingH':
            X = pd.read_csv('data/House_sales/KingH-NN_features.csv')
            y = pd.read_csv('data/House_sales/KingH-labels.csv').values

        elif name == 'USCensus':
            # X = pd.read_csv('data/US_Census/USCensus-2015-NN_features.csv')
            # y = pd.read_csv('data/US_Census/USCensus-2015-labels.csv')
            # print(X.values.shape, np.linalg.matrix_rank(X.values)).values
            # print(X.describe())
            X = pd.read_csv('data/US_Census/USCensus-2017-NN_features.csv')
            y = pd.read_csv('data/US_Census/USCensus-2017-labels.csv').values

        elif name == 'FaceA':
            X = pd.read_csv('data/Face_Age/face_age-CNN_features.csv')
            y = pd.read_csv('data/Face_Age/face_age-labels.csv').values
            X = X.drop(columns=['9']) # column 9 is all 0, so we drop it upon inspection


        print(name, X.values.shape, np.linalg.matrix_rank(X.values))
        X = MinMaxScaler().fit_transform(X=X.values)

        exp_dir = oj(name, 'repli-against')

        indices1 = np.random.choice(range(len(X)), size=1000)
        X1 = torch.from_numpy(X[indices1])
        y1 = torch.from_numpy(y[indices1])

        indices2 = np.random.choice(range(len(X)), size=1000)
        X2 = torch.from_numpy(X[indices2])
        y2 = torch.from_numpy(y[indices2])

        indices3 = np.concatenate([indices1[:800], indices2[:200]])
        X3 = torch.from_numpy(X[indices3])
        y3 = torch.from_numpy(y[indices3])

        used_indices = set(indices1).union(set(indices2)).union(set(indices3))
        test_indices = set(range(len(X))) - used_indices

        test_indices  = np.random.choice(list(test_indices), size=3000)
        test_X, test_y = X[test_indices], y[test_indices]

        test_X, test_y = torch.from_numpy(test_X), torch.from_numpy(test_y)

        repli_times = list(range(1, 101, 5))

        replicated_ig_sv = []
        replicated_v_sv = []

        replicated_rv_sv_01 = []
        replicated_rv_sv_005 = []

        replicated_tl_sv = []
        replicated_mc_tl_sv = []
        replicated_tmc_tl_sv = []
        
        replicated_tl_loo = []

        for i in repli_times:
            Xs = [torch.cat([X1 for _ in range(i+1)]), X1, X2]
            ys = [torch.cat([y1 for _ in range(i+1)]), y1, y2]

            tl_loo = test_loss_LOO(Xs, ys, test_X, test_y)
            replicated_tl_loo.append(tl_loo/tl_loo.sum())

            tl_sv, mc_tl_sv, tmc_tl_sv = test_loss_SV(Xs, ys, test_X, test_y)

            replicated_tl_sv.append(tl_sv/tl_sv.sum())
            replicated_mc_tl_sv.append(mc_tl_sv/mc_tl_sv.sum())
            replicated_tmc_tl_sv.append(tmc_tl_sv/tmc_tl_sv.sum())

            ig_svs = compute_IG_SVs(Xs, ys, prior_alpha=0.05, prior_beta=1)

            v_svs, rv_svs, *_ = shapley_volume(Xs, omega=0.1) 
            v_svs, rv_svs = v_svs.numpy(), rv_svs.numpy()
            replicated_rv_sv_01.append(rv_svs/rv_svs.sum())

            v_svs, rv_svs, *_ = shapley_volume(Xs, omega=0.05)  
            v_svs, rv_svs = v_svs.numpy(), rv_svs.numpy()

            replicated_rv_sv_005.append(rv_svs/rv_svs.sum())
            replicated_v_sv.append(v_svs/v_svs.sum())

            replicated_ig_sv.append(ig_svs/ig_svs.sum())



        replicated_tl_sv = np.asarray(replicated_tl_sv)
        replicated_mc_tl_sv = np.asarray(replicated_mc_tl_sv)
        replicated_tmc_tl_sv = np.asarray(replicated_tmc_tl_sv)

        replicated_rv_sv_01 = np.asarray(replicated_rv_sv_01)
        replicated_rv_sv_005 = np.asarray(replicated_rv_sv_005)
        replicated_v_sv = np.asarray(replicated_v_sv)

        replicated_ig_sv = np.asarray(replicated_ig_sv)
        replicated_tl_loo = np.asarray(replicated_tl_loo)


        os.makedirs(exp_dir, exist_ok=True)

        np.savetxt(oj(exp_dir, 'ig-svs.txt'), replicated_ig_sv)
        np.savetxt(oj(exp_dir, 'rv-svs-01.txt'), replicated_rv_sv_01)
        np.savetxt(oj(exp_dir, 'rv-svs-005.txt'), replicated_rv_sv_005)

        np.savetxt(oj(exp_dir, 'v-svs.txt'), replicated_v_sv)

        np.savetxt(oj(exp_dir, 'tl-svs.txt'), replicated_tl_sv)
        np.savetxt(oj(exp_dir, 'mc_tl-svs.txt'), replicated_mc_tl_sv)
        np.savetxt(oj(exp_dir, 'tmc_tl-svs.txt'), replicated_tmc_tl_sv)

        np.savetxt(oj(exp_dir, 'tl-loo.txt'), replicated_tl_loo)
        # -- plot  -- 
        markers = ['v', '^', '<', '>', '+', 'x']

        plt.figure(figsize=(12, 8))


        plt.plot(repli_times, replicated_v_sv[:, 0], marker='+',label='VSV', linewidth=3)
        plt.plot(repli_times, replicated_rv_sv_01[:, 0], marker='<',label='RVSV $\\omega$=0.1', linewidth=3)
        plt.plot(repli_times, replicated_rv_sv_005[:, 0], marker='>',label='RVSV $\\omega$=0.05', linewidth=3)

        plt.plot(repli_times, replicated_ig_sv[:, 0], marker='o',label='IGSV $\\sigma$=0.05', linewidth=3)
        plt.plot(repli_times, replicated_tl_sv[:, 0], marker='v',label='SV', linewidth=3)
        plt.plot(repli_times, replicated_tl_loo[:, 0], marker='^',label='LOO', linewidth=3)
        # plt.plot(repli_times, replicated_mc_tl_sv[:, 0], marker='x',label='IGSV $\\sigma$='+str(a), linewidth=3)
        # plt.plot(repli_times, replicated_tmc_tl_sv[:, 0], marker='+',label='IGSV $\\sigma$='+str(a), linewidth=3)


        plt.legend(ncol=2, fontsize=22, loc='upper left')
        plt.ylabel("Value of $D_1$ ", fontsize=30)
        plt.xlabel("Replication factor $c$", fontsize=30)

        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.title('Valuation vs. Replication', fontsize=32)
        plt.tight_layout()
        plt.savefig(oj(exp_dir, 'replication-all_{}.png'.format(name)) )
        # plt.show()
        plt.clf()
        plt.close()


