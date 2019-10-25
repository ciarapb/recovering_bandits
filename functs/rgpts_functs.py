#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Thompson sampling for gaussian process recovering bandits
in single step case
"""

# add necessary modules
import gen_functs as g
import numpy as np
from operator import attrgetter
from scipy.stats import norm


# define function to sample reward of an arm at z
def GetTS(myarm, z, t, K, Z):
    m = myarm.model
    pred = m.predict(np.array([[z]]))
    mean = pred[0]
    sd = np.sqrt(pred[1])
    mysample = norm.rvs(mean, sd)
    myarm.ts = mysample


# RGP-TS algorithm
# define function to run RGP-TS once
def RGPTS(T, Arms, K, Z):
    # initialize parameters
    reg = np.zeros(T)
    rew = np.zeros(T)

    # run inital rounds
    for t in range(0, K):
        myarm = Arms[t]
        ynew = g.PlayArm(myarm, myarm.z)
        rew[t] = ynew
        myarm.UpdatePosterior(myarm.z, ynew)
        g.UpdateErews(Arms)
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        g.UpdateZs(Arms, myarm, Z)
        rew[t] = ynew

    # run full algorithm
    for t in range(K, T):
        # first caculate up to date ucbs and rewards
        for arm in Arms:
            GetTS(arm, arm.z, t, K, Z)
        g.UpdateErews(Arms)

        # select arm and play it
        myarm = max(Arms, key=attrgetter('ts'))
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        ynew = g.PlayArm(myarm, myarm.z)
        rew[t] = ynew
        myarm.UpdatePosterior(myarm.z, ynew)

        # update all z's ready for next round
        g.UpdateZs(Arms, myarm, Z)
        
        #print t

    return reg, rew
