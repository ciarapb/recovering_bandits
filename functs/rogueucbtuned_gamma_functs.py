#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for running the rogueucb algorithm
Need to define separate functions for each model
These are for GAMMA model
"""

# add necessary modules
import gen_functs as g
import numpy as np
from scipy.optimize import minimize
from setups.rgp_gamma_l1_setup import rewfunct, noisevar, diffrew, maxwait
from operator import attrgetter


# define function to calulate residuals
def Residuals(theta, z, y):
    myfunct = rewfunct(theta, z)
    return (myfunct - y)**2


# define likelihood function
def NegLogLik(theta, zhist, yhist, sigma2):
    const = -np.log(np.sqrt(2*np.pi*sigma2))/(2*sigma2)
    res = Residuals(theta, zhist, yhist)
    negloglik = const+sum(res)
    return negloglik


# calculate mle and inverse of observed information (hessian)
def UpdateTheta(myarm):
    x0 = myarm.thetahat
    res = minimize(NegLogLik, x0,
                   args=(np.array(myarm.zhist), np.array(myarm.yhist),
                         noisevar), 
                   bounds=([0.1, maxwait], [0.1, 0.8], [0.1, 1.]))
    myarm.thetahat = res.x
    myarm.invobsinf = res.hess_inv


# define function to find kl divergence for different thetas
def KL(theta1, theta2, zhist, sigma2):
    diff = (rewfunct(theta1, zhist) - rewfunct(theta2, zhist))**2
    kl = sum(diff)/(2*sigma2)
    return kl


# define function for derrivative vector of kl wrt theta2
def DiffKL(theta1, theta2, zhist, sigma2):
    drew = (rewfunct(theta1, zhist) - rewfunct(theta2, zhist))
    drew2 = diffrew(theta2, zhist)
    diffkl = np.sum(drew2*drew, axis=1)/(2*sigma2)
    return diffkl


# define function to compute their S function
def S(theta, myarm):
    theta2 = myarm.thetahat
    DKL = DiffKL(theta, theta2, myarm.zhist, noisevar)
    A = myarm.invobsinf
    matpart = (DKL.T.dot(A)*DKL.T).sum(axis=1)
    S = ((1/myarm.numplays)**2)*matpart
    return S


# define constraint function for optimization problem
# note constraint is of the form ...>=0
def consfunct(theta, myarm, t, eta, sigma2):
    firstterm = (1/myarm.numplays)*KL(theta, myarm.thetahat, myarm.zhist, sigma2)
    secondterm = np.sqrt(min(eta/4.0, S(theta, myarm))*np.log(t)/myarm.numplays)
    return secondterm - firstterm

    
# need to define a negative reward function to minimize
def negrewfunct(theta, z):
    return -1*rewfunct(theta,z)


# define function to calculate UCB
def CalcUCB(myarm, t, eta, sigma2):
    cons = {'type':'ineq', 'fun':consfunct, 'args':(myarm, t, eta, sigma2)}
    x0 = myarm.thetahat
    res = minimize(negrewfunct, x0, args=myarm.z, 
                   bounds=([0.1, maxwait], [0.1, 0.8], [0.1, 1.]),
                   constraints=cons)
    ucb = -1*res.fun
    return ucb


# RGPUCB algorithm
# define function to run RUCB once
def RogueUCBTuned(T, Arms, K, Z, eta, sigma2):
    # initialize parameters
    reg = np.zeros(T)
    rew = np.zeros(T)

    # run inital rounds
    for t in range(0, K):
        myarm = Arms[t]
        ynew = g.PlayArm(myarm, myarm.z)
        g.UpdateHist(myarm, myarm.z, ynew)
        UpdateTheta(myarm)
        g.UpdateErews(Arms)
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        g.UpdateZs(Arms, myarm, Z)
        rew[t] = ynew

    # run full algorithm
    for t in range(K, T):
        # first caculate up to date ucbs and rewards
        for arm in Arms:
            arm.ucb = CalcUCB(arm, t, eta, sigma2)
        g.UpdateErews(Arms)

        # select arm and play it
        myarm = max(Arms, key=attrgetter('ucb'))
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        ynew = g.PlayArm(myarm, myarm.z)
        rew[t] = ynew
        UpdateTheta(myarm)

        # update all z's ready for next round
        g.UpdateZs(Arms, myarm, Z)
        
        # print t,
        
    return reg, rew
