#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
General functions for optimistic planning
This is similar to those in gen_funts with the difference 
that here we define a class of sequence objects
(may not be necessary to have two versions of this)
"""

# add necessary modules
import numpy as np
from operator import attrgetter, itemgetter
import itertools
from scipy.stats import norm


# define function to play arm
def PlayArm(myarm, z):
    # sample reward
    rew = myarm.sample(z)

    # update number of plays
    myarm.numplays += 1

    return rew


# define function to update all z's
def UpdateZs(allarms, playedarm, Z):
    for arm in allarms:
        arm.z = min(arm.z+1, Z)
    playedarm.z = 0.


# define function to caclulate all expected rewards (for regret)
def UpdateErews(allarms):
    for arm in allarms:
        arm.erew = arm.rewfunct(arm.z)


# define function to update the history (only used by ROGUEucb)
def UpdateHist(myarm, z, rew):
    # update zhist and yhist
    myarm.zhist = np.vstack((myarm.zhist, z))
    myarm.yhist = np.vstack((myarm.yhist, rew))


# functions for d step lookahead
# define function to get entire sequence of z's for an arm in a given sequence
def ZSeq(myarm, myseq, Z):
    d = len(myseq)
    z = np.zeros(d+1)
    z[0] = myarm.z
    # also need to go up to d+1 to make sure next z after the d step policy is accurate
    for i in range(1,d+1):
        if myseq[i-1]==myarm:
            z[i] = 0
        else:
            z[i] = min(z[i-1] +1, Z)
    return z


# define function to get sequence of z where the arms are played in each sequence
def ZPlayedSeq(myseq, Z):
    zseqs = {}
    for arm in myseq:
        zseqs[arm] = ZSeq(arm, myseq, Z)
    allzs = [zseqs[myseq[i]][i] for i in range(len(myseq))]
    return allzs

   
# define function to get expected reward for a d step sequence
def ErewSeq(myseq, Z):
    d = len(myseq)
    zseq = ZPlayedSeq(myseq, Z)
    erew = sum([myseq[i].rewfunct(zseq[i]) for i in range(d)])
    return erew


# define function to get d step optimal sequence
def GetBestSeq(Arms, d, Z):
    erews = {}
    for p in itertools.permutations(Arms, d):
        erews[p] = ErewSeq(p, Z)
    myseq = max(erews.iteritems(), key=itemgetter(1))[0]
    return myseq
    

# define function to get d step optimal sequence when repetitions are allowed
def GetBestSeqRep(Arms, d, Z):
    erews = {}
    for p in itertools.product(Arms, repeat = d):
        erews[p] = ErewSeq(p, Z)
    myseq = max(erews.iteritems(), key=itemgetter(1))[0]
    return myseq


# define a class of sequence objects to be used in optimistic planning
class Sequence:
    def __init__(self, indicies, value, zcurr):
        self.indicies = indicies
        self.value = value
        self.zcurr = zcurr
        self.depth = len(self.indicies)
        self.b = 0.

