#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for running the dRGP-UCB algorihtm
"""

# add necessary modules
import gen_functs as g
import numpy as np
from operator import attrgetter
from operator import itemgetter
import itertools


# define function to calculate ucb of an arm at z
def CalcUCB(myarm, z, t, K, d, Z):
    m = myarm.model
    pred = m.predict(np.array([[z]]))
    cb = np.sqrt(2*np.log(((t+d-1)**2)*((K*Z)**d)))
    ucb = pred[0] + cb*np.sqrt(pred[1])
    return ucb


# define function to get the ucb of a sequence of arms
def UCBSeq(myseq, t, K, Z):
    d = len(myseq)
    zseq = g.ZPlayedSeq(myseq, Z)
    pred = np.zeros((d,2))
    for i in range(d):
        m = myseq[i].model
        pred[i] = m.predict(np.array([[zseq[i]]]))
    mean, var = np.sum(pred, axis=0)
    cb = np.sqrt(2*np.log(((t+d-1)**2)*(K**d)*Z))
    ucb = mean + cb*np.sqrt(var)
    return ucb


# define function to get the ucb of a sequence of arms with repetition 
def UCBSeqRep(myseq, t, K, Z):
    d = len(myseq)
    zseq = g.ZPlayedSeq(myseq, Z)
    pred = np.zeros((d,2))
    for i in range(d):
        m = myseq[i].model
        pred[i] = m.predict(np.array([[zseq[i]]]))
    mean, var = np.sum(pred, axis=0)
    
    # need to add on covariance for any repeats from the same arm
    duplicates = set([arm for arm in myseq if myseq.count(arm) > 1])
    if len(duplicates) > 0:
        for myarm in duplicates:
            ismyarm = [arm==myarm for arm in myseq]
            myzs = np.array(zseq)[ismyarm]
            for zpair in itertools.combinations(myzs, 2):
                cov = myarm.model.posterior_covariance_between_points(np.array([[zpair[0]]]),np.array([[zpair[1]]]))
                var += 2*cov
    
    cb = np.sqrt(2*np.log(((t+d-1)**2)*(K**d)*Z))
    ucb = mean + cb*np.sqrt(var)
    return ucb


# define function to get sequence with highest d step ucb
def GetUCBSeq(Arms, d, t, K, Z):
    ucbs = {}
    for p in itertools.permutations(Arms, d):
        ucbs[p] = UCBSeq(p, t, K, Z)
    myseq = max(ucbs.iteritems(), key=itemgetter(1))[0]
    return myseq
    

# define function to get sequence with highest d step ucb when repetitions are allowed
def GetUCBSeqRep(Arms, d, t, K, Z):
    ucbs = {}
    for p in itertools.product(Arms, repeat=d):
        ucbs[p] = UCBSeqRep(p, t, K, Z)
    myseq = max(ucbs.iteritems(), key=itemgetter(1))[0]
    return myseq


# dRGP-UCB algorithm
# define function to run dRGP-UCB once
def RGPUCBds(T, Arms, d, K, Z, repetitions=False):
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
        rew[t] = ynew
        g.UpdateZs(Arms, myarm, Z)
    
    if repetitions:
        # run full algorithm with repetitions allowed in sequences
        for t in range(K, T):
            seqpoint = (t-K) % d #find out where in the sequence we are
            if seqpoint == 0:
                g.UpdateErews(Arms)
                myseq = GetUCBSeqRep(Arms, d, t, K, Z)
                bestseq = g.GetBestSeqRep(Arms, d, Z)
                reg2 = g.ErewSeq(bestseq, Z)- g.ErewSeq(myseq, Z)
            myarm = myseq[seqpoint]
            ynew = g.PlayArm(myarm, myarm.z)
            rew[t] = ynew
            reg[t] = reg2/d #average out regret from d step lookahead over the d steps
            # update posterior - we have already chosen sequence so it doesnt matter if we do this now
            myarm.UpdatePosterior(myarm.z, ynew)
            # update all z's ready for next round
            g.UpdateZs(Arms, myarm, Z)
            # print t,
    else:
        # run full algorithm without repetitions in sequences
        for t in range(K, T):
            g.UpdateErews(Arms)
            seqpoint = (t-K) % d #find out where in the sequence we are
            if seqpoint == 0:
                g.UpdateErews(Arms)
                myseq = GetUCBSeq(Arms, d, t, K, Z)
                bestseq = g.GetBestSeq(Arms, d, Z)
                reg2 = g.ErewSeq(bestseq, Z)- g.ErewSeq(myseq, Z)
            myarm = myseq[seqpoint]
            ynew = g.PlayArm(myarm, myarm.z)
            rew[t] = ynew
            reg[t] = reg2/d #average out regret from d step lookahead over the d steps
            # update posterior - we have already chosen sequence so it doesnt matter if we do this now
            myarm.UpdatePosterior(myarm.z, ynew)
            # update all z's ready for next round
            g.UpdateZs(Arms, myarm, Z)
            # print t,

    return reg, rew
