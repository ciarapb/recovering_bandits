#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for the basic UCB-Z algorithm
that runs ucb for each arm-z pair
"""

# add necessary modules
import gen_functs as g
import numpy as np
from operator import attrgetter
import random


# define function to calculate ucb of an arm at z
def CalcUCBz(myarm, z, T, K, Z, sigma2):
    numplays = float(myarm.numplaysz[z])
    mu = float(myarm.totrewz[z])/numplays
    cb = np.sqrt(2*sigma2*(2+6*np.log(T*Z*K))/numplays)
    ucb = mu + cb
    myarm.ucbz[z] = ucb


# RGPUCB algorithm
# define function to run RUCB once
def UCB(T, Arms, K, Z, sigma2):
    # initialize parameters
    reg = np.zeros(T)
    rew = np.zeros(T)

    # define ucb, numplays, totrew as dictionaries per arm
    for arm in Arms:
        setattr(arm, 'ucbz', {})
        setattr(arm, 'numplaysz', {})
        setattr(arm, 'totrewz', {})
        for z in range(int(Z)+1):
            arm.ucbz[z], arm.numplaysz[z], arm.totrewz[z] = 0.,0.,0.

    # run inital rounds playing each arm at each z
    t=0
    for j in range(0, K):
        if t>= T:
            break
        myarm = Arms[j]
        ynew = g.PlayArm(myarm, myarm.z)
        myarm.numplaysz[int(myarm.z)]+=1
        myarm.totrewz[int(myarm.z)]+=1
        g.UpdateErews(Arms)
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        g.UpdateZs(Arms, myarm, Z)
        rew[t] = ynew
        t+=1
        if t>= T:
            break
        otherarms = [arm for arm in Arms if arm != myarm]
        for z in range(0,int(Z)):
            if myarm.numplaysz[int(z)]==0:
                while int(myarm.z) != z:
                    # select other arm to play at random while waiting for our arm
                    mytemparm = random.choice(otherarms)
                    ynew = g.PlayArm(mytemparm, mytemparm.z)
                    mytemparm.numplaysz[int(mytemparm.z)]+=1
                    mytemparm.totrewz[int(mytemparm.z)]+=1
                    g.UpdateErews(Arms)
                    reg[t] = max([arm.erew for arm in Arms]) - mytemparm.erew
                    g.UpdateZs(Arms, mytemparm, Z)
                    rew[t] = ynew
                    t+=1
                    if t >= T:
                        break
                if t>= T:
                    break
                ynew = g.PlayArm(myarm, myarm.z)
                myarm.numplaysz[int(myarm.z)]+=1
                myarm.totrewz[int(myarm.z)]+=1
                g.UpdateErews(Arms)
                reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
                g.UpdateZs(Arms, myarm, Z)
                rew[t] = ynew
                t+=1
                if t>= T:
                   break

    # print t

    # run full algorithm
    while t < T:
        # first caculate up to date ucbs and rewards
        for arm in Arms:
            CalcUCBz(arm, arm.z, t, K, Z)
            arm.ucb = arm.ucbz[int(arm.z)]
        g.UpdateErews(Arms)

        # select arm and play it
        myarm = max(Arms, key=attrgetter('ucb'))
        reg[t] = max([arm.erew for arm in Arms]) - myarm.erew
        ynew = g.PlayArm(myarm, myarm.z)
        myarm.numplaysz[int(myarm.z)]+=1
        myarm.totrewz[int(myarm.z)]+=1
        rew[t] = ynew

        # update all z's ready for next round
        g.UpdateZs(Arms, myarm, Z)

        t+=1
        
        #print t

    return reg, rew
