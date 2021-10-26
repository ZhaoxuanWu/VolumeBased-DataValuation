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

def get_sorted_via_row_norms(a):
    sortidxs = np.einsum('ij,ij->i', a, a).argsort()
    return a[sortidxs]


# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.preprocessing import StandardScaler, minmax_scale, MinMaxScaler

if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for name in ['CaliH', 'KingH', 'USCensus', 'FaceA']:

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

        X = get_sorted_via_row_norms(X)

        out_exp_dir = oj(name, 'disj')

        # disj_r=0, completely disjoint data, disj_r = 1 10% overlap of domain, disj_r = 10 complete overlap domain
        for disj_r in range(0, 11):

            exp_dir = oj(out_exp_dir, str(disj_r))

            disj_r /= 10.0

            indices1 = np.random.choice(range(  int( len(X) // 3 * (1 + 2 * disj_r))  ), size=1000)
            X1 = torch.from_numpy(X[indices1])
            y1 = torch.from_numpy(y[indices1])

            indices2 = np.random.choice(range(  int(len(X) // 3 *(1 - disj_r)) ,  int( (len(X) * 2 // 3) +  ( len(X) //3 *  disj_r)  ) ), size=1000)             
            X2 = torch.from_numpy(X[indices2])
            y2 = torch.from_numpy(y[indices2])

            indices3 = np.random.choice(range(  int(len(X) * 2// 3  -  ( len(X) * 2 //3 * disj_r)),  len(X)), size=1000) 
            X3 = torch.from_numpy(X[indices3])
            y3 = torch.from_numpy(y[indices3])

            # -- running the experiments  -- 

            repli_times = list(range(1, 101, 5))

            replicated_ig_sv = []
            replicated_v_sv = []
            replicated_rv_sv = []

            sigmas = [0.0001, 0.001, 0.01, 0.1, 1]
            omegas = [0.01, 0.05, 0.1, 0.2, 0.25]

            for sigma in sigmas:
                temp_r_ig_svs = []
                for i in repli_times:
                    Xs = [torch.cat([X1 for _ in range(i+1)]), X2, X3]
                    ys = [torch.cat([y1 for _ in range(i+1)]), y2, y3]

                    ig_svs = compute_IG_SVs(Xs, ys, prior_alpha=sigma, prior_beta=1)
                    temp_r_ig_svs.append(ig_svs[0])

                replicated_ig_sv.append(temp_r_ig_svs)

            temp_r_svs = []
            for time, o in enumerate(omegas):
                print("omega :",o)
                temp_r_svs = []
                for i in repli_times:
                    Xs = [torch.cat([X1 for _ in range(i+1)]), X2, X3]
                    ys = [torch.cat([y1 for _ in range(i+1)]), y2, y3]
                    v_svs, rv_svs, *_ = shapley_volume(Xs, omega=0.1)  
                    temp_r_svs.append(rv_svs[0])

                    if time == 0: replicated_v_sv.append(v_svs[0])

                #print("rvsv:", temp_r_svs)
                replicated_rv_sv.append(temp_r_svs)

            os.makedirs(exp_dir, exist_ok=True)

            np.savetxt(oj(exp_dir, 'ig-svs.txt'), np.asarray(replicated_ig_sv))
            np.savetxt(oj(exp_dir, 'rv-svs.txt'), np.asarray(replicated_rv_sv))
            np.savetxt(oj(exp_dir, 'v-svs.txt'), np.asarray(replicated_v_sv))

            # -- plot  -- 
            markers = ['v', '^', '<', '>', '+', 'x']

            plt.figure(figsize=(8, 6))
            plt.plot(repli_times, np.log(repli_times), label='$\ln$', linewidth=2)

            replicated_ig_sv = np.asarray(replicated_ig_sv).reshape(len(sigmas), -1)
            for i, a in enumerate(sigmas):
                plt.plot(repli_times, replicated_ig_sv[i], marker='o',label='$\\sigma$='+str(a), linewidth=2)

            # plt.plot(repli_times, replicated_v_sv, marker='^', label='Vol SV')

            replicated_rv_sv = np.asarray(replicated_rv_sv).reshape(len(omegas), -1)
            for i, o in enumerate(omegas):
                plt.plot(repli_times, replicated_rv_sv[i], marker=markers[i], linestyle='dashed', label='$\\omega$='+str(o), linewidth=2)


            plt.legend(ncol=2, fontsize=14, loc='lower left')
            plt.ylabel("Shapley values", fontsize=18)
            plt.xlabel("Replication factor $c$", fontsize=18)

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.title('SV. vs. Replication', fontsize=22)
            # plt.ylim(0.2)
            plt.tight_layout()

            # plt.show()
            plt.savefig(oj(exp_dir, 'repli_IG_vs_RV_{}.png'.format(name)) )
            plt.clf()

            plt.close()
