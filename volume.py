from math import ceil, floor
from collections import defaultdict, Counter

import torch
import numpy as np
from torch import stack, cat, zeros_like, pinverse

def compute_volumes(datasets, d=1):
    d = datasets[0].shape[1]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det( dataset.T @ dataset ) + 1e-8)

    volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8).round(3)
    return volumes, volume_all

def compute_log_volumes(datasets, d=1):
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    log_volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        log_volumes[i] = np.linalg.slogdet(dataset.T @ dataset)[1]

    log_vol_all = np.linalg.slogdet(X.T @ X)[1]
    return log_volumes, log_vol_all

def compute_pinvs(datasets, d=1):
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)
    X = cat(datasets).reshape(-1, d)

    zero_padded_datasets = []
    pinvs = []

    count = 0 
    for i, dataset in enumerate(datasets):
        zero_padded_dataset = zeros_like(X)
        # fill the total set X with the rows of individual dataset
        for j, row in enumerate(dataset):
            zero_padded_dataset[j+count] = row
        count += len(dataset)

        zero_padded_datasets.append(zero_padded_dataset)        

        pinvs.append(pinverse(zero_padded_dataset))

    pinv = pinverse(X)

    return pinvs, pinv


def compute_X_tilde_and_counts(X, omega):
    """
    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    D = X.shape[1]

    # assert 0 < omega <= 1, "omega must be within range [0,1]."

    m = ceil(1.0 / omega) # number of intervals for each dimension

    cubes = Counter() # a dictionary to store the freqs
    # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
    # value: counts

    Omega = defaultdict(list)
    # Omega = {}
    
    min_ds = torch.min(X, axis=0).values

    # a dictionary to store cubes of not full size
    for x in X:
        cube = []
        for d, xd in enumerate(x - min_ds):
            d_index = floor(xd / omega)
            cube.append(d_index)

        cube_key = tuple(cube)
        cubes[cube_key] += 1

        Omega[cube_key].append(x)

        '''
        if cube_key in Omega:
            
            # Implementing mean() to compute the average of all rows which fall in the cube
            
            Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
            # Omega[cube_key].append(x)
        else:
             Omega[cube_key] = x
        '''
    X_tilde = stack([stack(list(value)).mean(axis=0) for key, value in Omega.items()])

    # X_tilde = stack(list(Omega.values()))

    return X_tilde, cubes




def compute_robust_volumes(X_tildes, dcube_collections):
        
    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    volumes, volume_all = compute_volumes(X_tildes, d=X_tildes[0].shape[1])
    robust_volumes = np.zeros_like(volumes)
    for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

            rho_omega_prod *= rho_omega

        robust_volumes[i] = (volume * rho_omega_prod).round(3)
    return robust_volumes



def compute_log_robust_volumes(X_tildes, dcube_collections):
        
    N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    log_volumess, log_volume_all = compute_log_volumes(X_tildes, d=X_tildes[0].shape[1])
    log_robust_volumes = np.zeros_like(log_volumess)
    for i, (log_volume, hypercubes) in enumerate(zip(log_volumess, dcube_collections)):
        rho_omega_prod = 1.0
        for cube_index, freq_count in hypercubes.items():
            
            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

            rho_omega_prod += np.log(rho_omega)

        log_robust_volumes[i] = (log_volume + rho_omega_prod).round(3)
    return log_robust_volumes




def compute_bias(S1, S2, d=1):
    X = np.append(S1, S2)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    # print("data matrices shapes:", S1.shape, S2.shape, X.shape)
    XI_S1 = np.zeros( (X.shape[0], X.shape[0]) )
    XI_S2 = np.zeros( (X.shape[0], X.shape[0]) )

    IS1 = np.append(np.ones(s), np.zeros(s))
    IS2 = np.append(np.zeros(s), np.ones(s))
    for i in range(X.shape[0]):
        XI_S1[i,i] = IS1[i]
        XI_S2[i,i] = IS2[i]

    XI_S1 = XI_S1 @ X
    XI_S2 = XI_S2 @ X

    S1_pinv, S2_pinv = np.linalg.pinv(XI_S1), np.linalg.pinv(XI_S2)
    X_pinv = np.linalg.pinv(X)
    return np.linalg.norm(S1_pinv - X_pinv), np.linalg.norm(S2_pinv - X_pinv)

def compute_loss(S1, S2, f, d, param=False):
    y1 = np.asarray([f(s1) for s1 in S1 ])
    y2 = np.asarray([f(s2) for s2 in S2 ])
    # y1 = f(S1)
    # y2 = f(S2)
    
    X = np.append(S1, S2)
    y = np.append(y1, y2).reshape(-1, 1)

    S1 = S1.reshape(-1, d)
    S2 = S2.reshape(-1, d)
    X = X.reshape(-1, d)

    XI_S1_pinv, XI_S2_pinv = np.linalg.pinv(S1), np.linalg.pinv(S2)

    w_S1 = XI_S1_pinv @ y1
    w_S2 = XI_S2_pinv @ y2

    X_pinv = np.linalg.pinv(X)
    w_X =  X_pinv @ y

    loss1 = np.linalg.norm( X @ w_S1 - y )
    loss2 = np.linalg.norm( X @ w_S2 - y )
    loss = np.linalg.norm( X @ w_X - y )
    if not param:
        return loss1/len(y), loss2/len(y), loss/len(y)
    else:
        return loss1/len(y), loss2/len(y), loss/len(y), w_S1, w_S2


def check_freq_count(dcube_collections):
    for collection in dcube_collections:
        for cube_index, freq_count in collection.items():
            if freq_count != 1:
                print('freq count != 1', freq_count)
                break
    return


"""
Robustness helper functions
"""

def replicate(X, c, mode='full'):
    """
    Arguments: X: np.ndarray matrix of n x d shape representing the feature
               c: replication factor, c > 1
               mode: ['full', 'random']
               'full': replicate the entire dataset for c times
               'random': pick any row randomly to replicate (c*n) times
            
    Returns:
        A repliacted dataset
            X_r np.ndarray matrix of (n*c) x d shape
    """
    assert c > 1, "Replication factor must be larger than 1."
    c = int(c)
    if mode == 'full':
        X_replicated  = np.repeat(X, repeats=(c-1), axis=0)
        X_r = np.vstack((X, X_replicated))

    elif mode == 'random':
        L = len(X)
        probs = [1.0/L] * L
        repeats = np.random.multinomial(L* (c-1), probs, size=1)[0]
        X_replicated = np.repeat(X, repeats=repeats, axis=0)
        X_r = np.vstack((X, X_replicated))

    return torch.from_numpy(X_r)

def replicate_perturb(X, c=3, sigma=0.1):

    """
    Arguments: X: np.ndarray matrix of n x d shape representing the feature
               c: replication factor, c > 1
               sigma: the variance of zero-mean noise in each dimension
            
    Returns:
        A repliacted dataset
            X_r np.ndarray matrix of (n*c) x d shape
    """

    assert c > 1, "Replication factor must be larger than 1."
    c = int(c)

    assert 0 < sigma < 1, "For a standardized/normalized feature space, have sigma in [0, 1]."
    
    L = len(X)
    probs = [1.0/L] * L
    repeats = np.random.multinomial(L *(c-1), probs, size=1)[0]
    X_perturbed = np.repeat(X, repeats=repeats, axis=0)
    X_perturbed += np.random.normal(loc=0, scale=sigma, size=X_perturbed.shape)
    X_r = np.vstack((X, X_perturbed))

    return torch.from_numpy(X_r)


if __name__ == '__main__':
    d = 10
    s = d * 10

    for t in range(5):
        X = torch.from_numpy(np.random.normal(0, 1, (s,d)))
        v, _ = compute_volumes([X], d=d)
        print(v)
        break

    for t in range(5):
        X = torch.from_numpy(np.random.normal(0, 0.1, (s,d)))
        v, _ = compute_volumes([X], d=d)
        print(v)
        break
        rvs = []
        for o in [0.01, 0.1, 0.2, 0.49, 0.5, 0.51]:
            X_tilde, cubes = compute_X_tilde_and_counts(X, omega=o)
            rv = compute_robust_volumes([X_tilde], [cubes])
            if v[0] > rv:
                print('break', v[0], rv[0], o)

    exit()
    for t in range(100):
        X = torch.from_numpy(np.random.uniform(0, 1, (s,d)))
        v, _ = compute_volumes([X],  d=d)
        rvs = []
        for o in [0.01, 0.1, 0.2, 0.4, 0.49, 0.5, 0.51]:
            X_tilde, cubes = compute_X_tilde_and_counts(X, omega=o)
            rv = compute_robust_volumes([X_tilde], [cubes])
            if v[0] > rv:
                print('break', v[0], rv[0], o)



