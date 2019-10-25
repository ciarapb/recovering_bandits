#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
K=10, spiky GP setup
squared exponential kernel with l=0.5
"""

# add necessary modules
import numpy as np
from scipy.stats import norm
import GPy

# set the seed
np.random.seed(5389)

# define number of arms
K = 10

# define maximum waiting time (when expected reward should be back to max)
maxwait = 30.0

# define variance of noise in observations
noisevar = 0.01


# define arm class
class Arm:
    def __init__(self, samples, index, zinit, noisevar):
        self.index = index
        self.samples = samples
        self.noisevar = noisevar
        self.zinit = zinit
        self.z = zinit
        self.numplays = 0.0
        self.predrew = 0.0
        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.ucb = 0.0
        self.res = 0.0
        self.model = None
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.5)

    # associate reward 'function' to each arm
    def rewfunct(self, z):
        zint = int(z)
        rew = self.samples[zint]
        return rew

    # define function to sample from arm
    def sample(self, z):
        erew = self.rewfunct(z)
        noise = norm(0, np.sqrt(self.noisevar)).rvs()
        return erew + noise

    # define function to update posterior
    def UpdatePosterior(self, znew, ynew):
        self.zhist = np.vstack((self.zhist, znew))
        self.yhist = np.vstack((self.yhist, ynew))
        if self.model is None:
            self.model = GPy.models.GPRegression(X=self.zhist, Y=self.yhist,
                                                 kernel=self.kernel, noise_var=noisevar)
        else:
            self.model.set_XY(self.zhist, self.yhist)

    # define function to reset counts etc every replication
    def reset(self):
        self.z = self.zinit
        self.numplays = 0.0
        self.predrew = 0.0
        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.ucb = 0.0
        self.res = 0.0
        self.model = None



# set the seed
np.random.seed(5389)

# sample the reward functions from a gp prior
kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.5)
z = np.linspace(0., 30., 31)
z = z[:, None]
mu = np.zeros(31)
C = kern.K(z, z)
f = np.random.multivariate_normal(mu, C, K)

# initialize arms
Arms = []
for i in range(K):
    mysamples = f[i]
    Arms = Arms + [Arm(mysamples, i, 0., noisevar)]

