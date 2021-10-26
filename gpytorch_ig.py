import math
from matplotlib import pyplot as plt

import torch
import gpytorch

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.is_sparse = False
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points=None, inducing_count=0.1):

        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.is_sparse = True

        if inducing_points is None:
            inducing_points = train_x[:int(len(train_x) * inducing_count),: ]
        self.inducing_points = inducing_points
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# helper function
# get_logdet = lambda mat : torch.log(torch.linalg.det(mat))

def compute_K(X, model, is_sparse=False):

    M, D = X.shape
    if is_sparse:
        assert len(model.inducing_points) > 0, "Sparse GP requires non-zero number of inducing points."
        inducing_points = model.inducing_points

        K_U = model.covar_module(inducing_points).evaluate()
        K_U_inv = torch.linalg.inv(K_U)
        
        K_XU = model.covar_module(X, inducing_points).evaluate()
        K = K_XU @ K_U_inv @ K_XU.T
    else:
        K = model.covar_module(X).evaluate()

    return K


def fit_model(X, Y, likelihood=None, max_iters=100, verbose=False, lr=0.05, inducing_points=[]):

    # initialize likelihood and model
    likelihood = likelihood or gpytorch.likelihoods.GaussianLikelihood()
    
    if len(inducing_points) > 0:
        model = GPRegressionModel(X, Y, likelihood, inducing_points)
    else:
        model = ExactGPModel(X, Y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], 
        lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(max_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, max_iters, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

    return model, likelihood


def compute_IG(X, model, likelihood):
    model.eval()
    likelihood.eval()

    K = compute_K(X, model, is_sparse=model.is_sparse)
    noise_variance = model.likelihood.noise.item() # it is sigma**2
    likelihood_mat = torch.eye(len(K)) + K / noise_variance    
    return 0.5 * torch.logdet(likelihood_mat).item()


from main_utils import hartmann_function, friedman_function

import os
from copy import deepcopy as dcopy

import numpy as np
from scipy.stats import sem

if __name__ == '__main__':

    T = 10
    D = 6
    M = 120
    inducing_ratio = 0
    inducing_count = int(M * inducing_ratio)
    print("Inducing count is :", inducing_count)
    begin, end, = 0, 1.0

    ig_1s, ig_alls = [], []
    for t in range(T):
        inducing_points = torch.rand((inducing_count, D)) * (end - begin) + begin 

        fixed_train = torch.rand((M * 2, D)) * (end - begin) + begin 


        X = torch.rand((M, D)) * (1 - 0) + 0 

        # Y_clean, Y = friedman_function(X)
        Y_clean = friedman_function(X) #hartmann_function
        Y = (Y_clean + torch.randn(Y_clean.shape) * 0.1).squeeze()

        model, likelihood = fit_model(X, Y, inducing_points=inducing_points)

        IG = compute_IG(fixed_train, model, likelihood)
        ig_1s.append(IG)

        # set up a second set and combine both later to create Xall
        
        X2 = torch.rand((M * 2, D)) * (end - begin) + begin 

        Y2_clean = hartmann_function(X2)
        Y2 = (Y2_clean + torch.randn(Y2_clean.shape) * 0.1).squeeze()

        model_all, likelihood_all = fit_model(X2, Y2, inducing_points=inducing_points)

        IG = compute_IG(fixed_train, model_all, likelihood)
        ig_alls.append(IG)


    print("IG 1 mean: {}, sem : {}".format(np.mean(ig_1s), sem(ig_1s)))
    print("IG all mean: {}, sem : {}".format(np.mean(ig_alls), sem(ig_alls)))


    # Get into evaluation (predictive posterior) mode
    '''
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.from_numpy( np.random.uniform(-0., 1., (M, D)))
        observed_pred = likelihood(model(test_x))
    '''