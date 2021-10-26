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
from itertools import permutations

import matplotlib.pyplot as plt

from sklearn import metrics, model_selection
import torch
from torch import rand, randn, cat, stack


from blr import BayesLinReg, BatchBayesLinReg, shapley_volume, compute_IG_SVs
from blr_disjoint import get_sorted_via_row_norms

from robust_exp import test_loss_LOO, test_loss_SV


# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.preprocessing import StandardScaler, minmax_scale, MinMaxScaler


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for name in ['CaliH', 'KingH', 'USCensus', 'FaceA'][2:]:

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

        out_exp_dir = oj(name, 'repli-against', 'subsup')

        indices1 = np.random.choice(range(len(X)), size=1000)
        X1 = torch.from_numpy(X[indices1])
        y1 = torch.from_numpy(y[indices1])

        indices3 = np.random.choice(range(len(X)), size=1000) 
        X3 = torch.from_numpy(X[indices3])
        y3 = torch.from_numpy(y[indices3])


        for subsup_ratio in range(1, 11):

            exp_dir = oj(out_exp_dir, str(subsup_ratio))

            # player 2 has some overlap to player 1, ratio from 0.1 to 1. 
            indices2_sup = np.random.choice(indices1, size=int(len(indices1) * (subsup_ratio/10.0) ))
            
            # player 2 tops up another 1000
            indices2_topup = np.random.choice(range(len(X)), size=1000) 
            indices2 = np.concatenate([indices2_sup, indices2_topup])

            X2 = torch.from_numpy(X[indices2])
            y2 = torch.from_numpy(y[indices2])

            used_indices = set(indices1).union(set(indices2)).union(set(indices3))
            test_indices = set(range(len(X))) - used_indices

            test_indices  = np.random.choice(list(test_indices), size=3000)
            test_X, test_y = X[test_indices], y[test_indices]

            test_X, test_y = torch.from_numpy(test_X), torch.from_numpy(test_y)

            # -- running the experiments  -- 

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

            plt.plot(repli_times, replicated_v_sv[:, 0], marker='+',label='VSV', linewidth=3.5)
            plt.plot(repli_times, replicated_rv_sv_01[:, 0], marker='<',label='RVSV $\\omega$=0.1', linewidth=3.5)
            plt.plot(repli_times, replicated_rv_sv_005[:, 0], marker='>',label='RVSV $\\omega$=0.05', linewidth=3.5)

            plt.plot(repli_times, replicated_ig_sv[:, 0], marker='o',label='IGSV', linewidth=3.5)
            plt.plot(repli_times, replicated_tl_sv[:, 0], marker='v',label='VLSV', linewidth=3.5)
            plt.plot(repli_times, replicated_tl_loo[:, 0], marker='^',label='LOO', linewidth=3.5)
            # plt.plot(repli_times, replicated_mc_tl_sv[:, 0], marker='x',label='IGSV $\\sigma$='+str(a), linewidth=3.5)
            # plt.plot(repli_times, replicated_tmc_tl_sv[:, 0], marker='+',label='IGSV $\\sigma$='+str(a), linewidth=3.5)
            plt.legend(ncol=2, fontsize=22, loc='upper left')

            plt.ylabel("Value of $\mathbf{X}_1$ ", fontsize=30)
            plt.xlabel("Replication factor $c$", fontsize=30)

            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.title('Valuation vs. Replication', fontsize=32)
            plt.tight_layout()
            plt.savefig(oj(exp_dir, 'replication-all_{}.png'.format(name)) )
            # plt.show()
            plt.clf()
            plt.close()



