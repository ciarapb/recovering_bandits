#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
reflected gamma function setup
GP kernel used: squared exponential with l=2.5
"""

# add necessary modules
import numpy as np
from scipy.stats import norm
import GPy

# set the seed
np.random.seed(6592)

# define number of arms
K = 10

# define maximum waiting time (when expected reward should be back to max)
maxwait = 30.0

# define variance of noise in observations
noisevar = 0.01


# define reflected gamma function to get expected rewards
def rewfunct(theta, z):
    # use 31 to make sure derrivatives dont go to infinity
    num = ((31.-z)**theta[0])*np.exp(-theta[1]*(31.-z))
    denom = ((theta[0]/theta[1])**theta[0])*np.exp(-theta[0])
    fun = theta[2]*num/denom
    return fun


# define derivative of reward function
def diffrew(theta, z):
    num = ((31.-z)**theta[0])*np.exp(-theta[1]*(31.-z))
    denom = ((theta[0]/theta[1])**theta[0])*np.exp(-theta[0])
    dtheta0 = theta[2]*num/denom*(np.log(31.-z)-np.log(theta[0]/theta[1]))
    dtheta1 = theta[2]*num/denom*(theta[0]/theta[1] - (31.-z))
    dtheta2 = num/denom
    return np.array([dtheta0, dtheta1, dtheta2])


# define arm class
class Arm:
    def __init__(self, theta, index, zinit, noisevar):
        self.index = index
        self.theta = theta
        self.noisevar = noisevar
        self.zinit = zinit
        self.z = zinit
        self.numplays = 0.0
        self.predrew = 0.0
        self.thetahat = np.ones(len(self.theta))*0.5
        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.ucb = 0.0
        self.res = 0.0
        self.invobsinf = 0.0
        self.model = None
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=2.5)

    # add function to associate reward function to each arm
    def rewfunct(self, z):
        theta = self.theta
        num = ((31.-z)**theta[0])*np.exp(-theta[1]*(31.-z))
        denom = ((theta[0]/theta[1])**theta[0])*np.exp(-theta[0])
        fun = theta[2]*num/denom
        return fun
   
    # add function to sample from an arm
    def sample(self, z):
        erew = self.rewfunct(z)
        noise = norm(0, np.sqrt(self.noisevar)).rvs()
        return erew + noise

    # add function to update the posteror of the arm
    def UpdatePosterior(self, znew, ynew):
        self.zhist = np.vstack((self.zhist, znew))
        self.yhist = np.vstack((self.yhist, ynew))
        if self.model is None:
            self.model = GPy.models.GPRegression(X=self.zhist, Y=self.yhist,
                                                 kernel=self.kernel, noise_var=noisevar)
        else:
            self.model.set_XY(self.zhist, self.yhist)

    # define function to reset counts etc every replication in experiments
    def reset(self):
        self.z = self.zinit
        self.numplays = 0.0
        self.predrew = 0.0
        self.thetahat = np.ones(len(self.theta))*0.5
        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.ucb = 0.0
        self.res = 0.0
        self.invobsinf = 0.0
        self.model = None



# set up random value of theta
theta1 = np.random.uniform(0.1, 0.8, K)
theta0 = np.random.uniform(2*theta1, 20.*theta1, K)
theta2 = np.random.uniform(0.1, 1, K)


# initialize arms
Arms = []
for i in range(K):
    mytheta = [theta0[i], theta1[i], theta2[i]]
    Arms = Arms + [Arm(mytheta, i, 0., noisevar)]

