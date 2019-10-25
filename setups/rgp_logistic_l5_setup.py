#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
logistic function setup
GP kernel used: squared exponential with l=5
"""

# add necessary modules
import numpy as np
from scipy.stats import norm
import GPy

# set the seed
np.random.seed(2374)

# define number of arms
K = 10

# define maximum waiting time (when expected reward should be back to max)
maxwait = 30.0

# define variance of noise in observations
noisevar = 0.01


# define logit function to get expected rewards
def rewfunct(theta, z):
    exppart = np.exp(-theta[1]*(z - theta[2]))
    logit = theta[0]/(1 + exppart)
    return logit


# define derivative of reward function
def diffrew(theta, z):
    exppart = np.exp(-theta[1]*(z - theta[2]))
    dtheta0 = 1/(exppart+1)
    num1 = theta[0]*(z - theta[2])*exppart
    dtheta1 = num1/((1+exppart)**2)
    num2 = theta[0]*theta[1]*exppart
    dtheta2 = num2/((1+exppart)**2)
    return np.array([dtheta0, dtheta1, dtheta2])


# define arm class
class Arm:
    def __init__(self, theta, index, zinit, noisevar):
        self.index = index
        self.theta = theta
        self.noisevar = noisevar
        self.zinit = zinit
        self.z = maxwait
        self.numplays = 0.0
        self.predrew = 0.0
        self.thetahat = np.ones(len(self.theta))*0.5
        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.ucb = 0.0
        self.res = 0.0
        self.invobsinf = 0.0
        self.model = None
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=5.)

    # associate reward function to each arm
    def rewfunct(self, z):
        theta = self.theta
        exppart = np.exp(-theta[1]*(z - theta[2]))
        logit = theta[0]/(1 + exppart)
        return logit
   
    # define function to sample reward of arm
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
        self.thetahat = np.ones(len(self.theta))*0.5
        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.ucb = 0.0
        self.res = 0.0
        self.invobsinf = 0.0
        self.model = None



# set up random value of theta
theta0 = np.random.uniform(0.1, 1.0, K)
theta1 = np.random.uniform(0.1, 1.0, K)
theta2 = np.random.uniform(0.1, maxwait, K)

# define the diameter of theta
diamtheta = np.sqrt(2+ maxwait**2)

# define lipschitz constant of likelihood ratio
Lf = 0.5 #wrt theta
Lp = 2 #wrt Y

# initialize arms
Arms = []
for i in range(K):
    mytheta = [theta0[i], theta1[i], theta2[i]]
    Arms = Arms + [Arm(mytheta, i, 0., noisevar)]

