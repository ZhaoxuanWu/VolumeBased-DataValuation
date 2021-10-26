# Bayesian Linear Regression 
# Information Gain on the parameters

from os.path import join as oj
import math
import random

import numpy as np
from scipy import stats

from sklearn import metrics, model_selection
from torch import rand, randn, cat, stack

from numpy.linalg import slogdet

class BayesLinReg:

    def __init__(self, n_features, alpha, beta, mean=[], cov=None):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta

        if len(mean) == 0:
            self.mean = np.zeros(n_features)
        elif len(mean) == 1:
            self.mean = np.ones(n_features) * mean
        else:
            assert len(mean) == n_features
            self.mean = np.asarray(mean)

        if cov is None:
            self.cov_inv = np.identity(n_features) / alpha

        else:
            cov = np.asarray(cov)
            try:
                cov = cov.reshape(n_features, n_features)
                self.cov_inv = np.linalg.inv(cov)
            except Exception as e:
                print(e)
                print("Given cov prior is of the wrong shape.")
                self.cov_inv = np.identity(n_features) / alpha


    def learn(self, x, y):

        # Update the inverse covariance matrix (Bishop eq. 3.51)
        cov_inv = self.cov_inv + self.beta * np.outer(x, x)

        # Update the mean vector (Bishop eq. 3.50)
        cov = np.linalg.inv(cov_inv)
        mean = cov @ (self.cov_inv @ self.mean + self.beta * y * x)

        self.cov_inv = cov_inv
        self.mean = mean

        return self

    def predict(self, x):

        # Obtain the predictive mean (Bishop eq. 3.58)
        y_pred_mean = x @ self.mean

        # Obtain the predictive variance (Bishop eq. 3.59)
        w_cov = np.linalg.inv(self.cov_inv)
        y_pred_var = 1 / self.beta + x @ w_cov @ x.T

        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)

    @property
    def weights_dist(self):
        cov = np.linalg.inv(self.cov_inv)
        return stats.multivariate_normal(mean=self.mean, cov=cov)

    @property
    def cov(self):
        return np.linalg.inv(self.cov_inv)


class BatchBayesLinReg(BayesLinReg):

    def learn(self, x, y):

        # If x and y are singletons, then we coerce them to a batch of length 1
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        # Update the inverse covariance matrix (Bishop eq. 3.51)
        cov_inv = self.cov_inv + self.beta * x.T @ x

        # Update the mean vector (Bishop eq. 3.50)
        cov = np.linalg.inv(cov_inv)

        mean = cov @ ( (self.cov_inv @ self.mean).reshape(-1, 1) + self.beta * x.T @ y)

        self.cov_inv = cov_inv
        self.mean = mean
        return self

    def predict(self, x):

        x = np.atleast_2d(x)

        # Obtain the predictive mean (Bishop eq. 3.58)
        y_pred_mean = x @ self.mean

        # Obtain the predictive variance (Bishop eq. 3.59)
        w_cov = np.linalg.inv(self.cov_inv)
        y_pred_var = 1 / self.beta + (x @ w_cov * x).sum(axis=1)

        # Drop a dimension from the mean and variance in case x and y were singletons
        # There might be a more elegant way to proceed but this works!
        y_pred_mean = np.squeeze(y_pred_mean)
        y_pred_var = np.squeeze(y_pred_var)

        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)


import sys
sys.path.insert(0, '..')
from main_utils import hartmann_function, friedman_function
from volume import compute_volumes, compute_pinvs, compute_X_tilde_and_counts, compute_robust_volumes, compute_log_robust_volumes


def compute_IG_SVs(Xs, ys, prior_alpha=0.05, prior_beta=1):
    permutations = list(itertools.permutations(list(range(len(Xs)))))
    svs = np.zeros(len(Xs))
    if len(Xs) >= 5:
        random.shuffle(permutations)
        permutations = permutations[:30]

    n_features = Xs[0].shape[1]

    all_X = torch.cat(Xs)
    all_y = torch.cat(ys)
    model = BatchBayesLinReg(n_features=all_X.shape[1], alpha=prior_alpha, beta=prior_beta)
    model.learn(all_X, all_y)

    prior_cov = torch.from_numpy(model.cov)

    sigma_2 = prior_beta + (all_X @ model.cov * all_X).sum(axis=1)    
    sigma_2 = np.squeeze(sigma_2).mean()

    for pi in permutations:
        curr = 0
        for index, party in enumerate(pi):
            X_ = torch.cat([ Xs[party_index] for party_index in pi[:index+1]   ]).float()            
            new = 0.5 *(torch.logdet( torch.eye(n_features)  +  (prior_cov / sigma_2) * (X_.T @ X_)   ) )  

            svs[party] += new - curr
            curr = new
    
    if len(Xs) >= 5:
        svs /= len(permutations)
    else:
        svs /= fac(len(Xs))

    return svs


def test_code():
    D = 5
    M = 100
    X = rand((M, D)) * (1 - 0) + 0 
    Y_clean = friedman_function(X)
    y = Y_clean + randn(Y_clean.shape) * 0.05
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=.3,
        shuffle=True,
        random_state=42
    )

    X1 = rand((M, D)) * (0.6 - 0) + 0 
    Y_clean = friedman_function(X1)
    y1 = Y_clean + randn(Y_clean.shape) * 0.05

    X2 = rand((M, D)) * (0.6 - 0.2) + 0.2 
    Y_clean = friedman_function(X2)
    y2 = Y_clean + randn(Y_clean.shape) * 0.05


    X3 = rand((M, D)) * (1.0 - 0.6) + 0.6 
    Y_clean = friedman_function(X3)
    y3 = Y_clean + randn(Y_clean.shape) * 0.05

    Xs = [X1, X2, X3]
    ys = [y1, y2, y3]

    for beta in np.linspace(0.01, 1, 20):
        svs = compute_IG_SVs(Xs, ys, prior_beta=beta)
        svs /= np.sum(svs)
        print(svs)


def shapley_volume(Xs, omega=0.1):
    M = len(Xs)
    D = Xs[0].shape[1]
    # print("Number of parties:", M, "number of dimensions:" ,D)
    # print([X.shape for X in Xs])

    permutations = list(itertools.permutations(range(M)))

    s_values = torch.zeros(M)
    monte_carlo_s_values = torch.zeros(M)

    s_value_robust = torch.zeros(M)
    monts_carlo_s_values_robust = torch.zeros(M)

    # Monte-carlo : shuffling the ordering and taking the first K permutations
    random.shuffle(permutations)
    K = 0 # number of permutations to sample

    for pi_count, pi in enumerate(permutations):
        prefix_vol = 0
        prefix_robust_vol = 0
        for index, party in enumerate(pi):
            # print(Xs[:index+1])
            X_ = torch.cat([ Xs[party_index] for party_index in pi[:index+1] ])
            # y_ = torch.cat([ ys[party_index] for party_index in pi[:index+1] ])

            curr_vol = torch.logdet(X_.T @ X_ )
            # curr_vol = torch.sqrt(torch.linalg.det(X_.T @ X_ ))

            marginal = curr_vol - prefix_vol
            s_values[party] += marginal
            prefix_vol = curr_vol

            X_tilde, cubes = compute_X_tilde_and_counts(X_, omega)
            # print(np.linalg.matrix_rank(X_tilde))
            robust_vol = compute_log_robust_volumes([X_tilde], [cubes])[0]

            marginal_robust = robust_vol - prefix_robust_vol
            s_value_robust[party] += marginal_robust
            prefix_robust_vol = robust_vol

            if pi_count < K:
                monte_carlo_s_values[party] += marginal
                monts_carlo_s_values_robust[party] += marginal_robust


    s_values /= fac(M)
    s_value_robust /= fac(M)
    monte_carlo_s_values /= K
    monts_carlo_s_values_robust /= K
    '''
    print('------Volume-based Shapley value Statistics ------')
    # print("alpha : {}, omega : {}.".format(alpha, omega))
    print("Volume-based Shapley values:", s_values / s_values.sum())
    print("Robust Volume Shapley values:", s_value_robust / s_value_robust)
    print("Volume-based MC-Shapley values:", monte_carlo_s_values/ monte_carlo_s_values.sum())
    print("Robust Volume MC-Shapley values:", monts_carlo_s_values_robust /monts_carlo_s_values_robust.sum())
    print('-------------------------------------')
    '''
    return s_values, s_value_robust, monte_carlo_s_values, monts_carlo_s_values_robust


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


    for name in ['CaliH', 'KingH', 'USCensus', 'FaceA']:

        # -- set up data --

        if name == 'CaliH':
            X = pd.read_csv('data/California_housing/CaliH-NN_features.csv')
            X = X.drop(columns=['7']) # column 7 is all 0, so we drop it upon inspection
            y = pd.read_csv('data/California_housing/CaliH-labels.csv').values
        
            data_dir = 'data/California_housing/'

        elif name == 'KingH':
            X = pd.read_csv('data/House_sales/KingH-NN_features.csv')
            print(X.values.shape, np.linalg.matrix_rank(X.values))
            y = pd.read_csv('data/House_sales/KingH-labels.csv').values
            data_dir = 'data/House_sales/'

        elif name == 'USCensus':
            # X = pd.read_csv('data/US_Census/USCensus-2015-NN_features.csv')
            # y = pd.read_csv('data/US_Census/USCensus-2015-labels.csv')
            # print(X.values.shape, np.linalg.matrix_rank(X.values)).values
            # print(X.describe())

            X = pd.read_csv('data/US_Census/USCensus-2017-NN_features.csv')
            y = pd.read_csv('data/US_Census/USCensus-2017-labels.csv').values
            print(X.values.shape, np.linalg.matrix_rank(X.values))
            data_dir = 'data/US_Census/'

        elif name == 'FaceA':

            X = pd.read_csv('data/Face_Age/face_age-CNN_features.csv')
            y = pd.read_csv('data/Face_Age/face_age-labels.csv').values
            print(X.values.shape, np.linalg.matrix_rank(X.values))
            X = X.drop(columns=['9']) # column 9 is all 0, so we drop it upon inspection
            data_dir = 'data/Face_Age/'


        X = MinMaxScaler().fit_transform(X=X.values)

        indices1 = np.random.choice(range(len(X)), size=1000)
        X1 = torch.from_numpy(X[indices1])
        y1 = torch.from_numpy(y[indices1])

        indices2 = np.random.choice(range(len(X)), size=1000)
        X2 = torch.from_numpy(X[indices2])
        y2 = torch.from_numpy(y[indices2])

        indices3 = np.concatenate([indices1[:800], indices2[:200]])

        X3 = torch.from_numpy(X[indices3])
        y3 = torch.from_numpy(y[indices3])

        indices4 = np.concatenate([indices1[:200], indices2[:200],indices3[:200], np.random.choice(range(len(X)), size=400)])
        X4 = torch.from_numpy(X[indices4])
        y4 = torch.from_numpy(y[indices4])

        indices5 = np.concatenate([indices1[-200:], indices2[-200:],indices3[-200:], np.random.choice(range(len(X)), size=400)])
        X5 = torch.from_numpy(X[indices5])
        y5 = torch.from_numpy(y[indices5])


        # Xs = [X1, X2, X3]
        # ys = [y1, y2, y3]

        # Xs = [torch.cat([X1 for _ in range(10)]), X1, X2, X3]
        # ys = [torch.cat([y1 for _ in range(10)]), y1, y2, y3]


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
                Xs = [torch.cat([X1 for _ in range(i+1)]), X1, X2]
                ys = [torch.cat([y1 for _ in range(i+1)]), y1, y2]

                ig_svs = compute_IG_SVs(Xs, ys, prior_alpha=sigma, prior_beta=1)
                temp_r_ig_svs.append(ig_svs[0])

            replicated_ig_sv.append(temp_r_ig_svs)

        temp_r_svs = []
        for time, o in enumerate(omegas):
            temp_r_svs = []
            for i in repli_times:
                Xs = [torch.cat([X1 for _ in range(i+1)]), X1, X2]
                ys = [torch.cat([y1 for _ in range(i+1)]), y1, y2]
                v_svs, rv_svs, *_ = shapley_volume(Xs, omega=0.1)  
                temp_r_svs.append(rv_svs[0])

                if time == 0:
                    replicated_v_sv.append(v_svs[0])

            replicated_rv_sv.append(temp_r_svs)

        np.savetxt(oj(data_dir, 'ig-svs.txt'), np.asarray(replicated_ig_sv))
        np.savetxt(oj(data_dir, 'rv-svs.txt'), np.asarray(replicated_rv_sv))
        np.savetxt(oj(data_dir, 'v-svs.txt'), np.asarray(replicated_v_sv))



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
        plt.savefig('latex/diagrams/replication_IG_V_RV_{}.png'.format(name))
        plt.clf()
        plt.close()

