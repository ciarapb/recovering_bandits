#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for running RGP-UCB when d=1
"""

# add necessary modules
import gen_functs as g
import numpy as np
from operator import attrgetter


# define function to calculate ucb of an arm at z
def CalcUCB(myarm, z, t, K, Z):
    m = myarm.model
    pred = m.predict(np.array([[z]]))
    cb = np.sqrt(2*np.log(t**2*Z*K))
    ucb = pred[0] + cb*np.sqrt(pred[1])
    myarm.ucb = ucb
    #return ucb


# RGPUCB algorithm
# define function to run RUCB once
def RGPUCB(T, Arms, K, Z):
    # initialize parameters
    reg = np.zeros(T)
    rew = np.zeros(T)

    # run inital rounds
    for t in range(0, K):
        myarm = Arms[t]
        ynew = g.PlayArm(myarm, myarm.z)
        myarm.UpdatePosterior(myarm.z, ynew)
        g.UpdateErews(Arms)
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        g.UpdateZs(Arms, myarm, Z)
        rew[t] = ynew

    # run full algorithm
    for t in range(K, T):
        # first caculate up to date ucbs and rewards
        for arm in Arms:
            CalcUCB(arm, arm.z, t, K, Z)
        g.UpdateErews(Arms)

        # select arm and play it
        myarm = max(Arms, key=attrgetter('ucb'))
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        ynew = g.PlayArm(myarm, myarm.z)
        rew[t] = ynew
        myarm.UpdatePosterior(myarm.z, ynew)

        # update all z's ready for next round
        g.UpdateZs(Arms, myarm, Z)
        
        #print t

    return reg, rew
