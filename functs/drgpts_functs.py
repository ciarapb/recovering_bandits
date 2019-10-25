#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions to perform Thompson sampling for d step look ahead 
"""

# add necessary modules
import gen_functs as g
import numpy as np
from operator import itemgetter
import itertools
from scipy.stats import norm


# define function to sample from the functions at the appropriate values
def SampleFuncts(Arms, d, zcurr, Z):
    for arm in Arms:
        covariates = np.array([min(zcurr[arm.index] + i,Z) for i in range(d)])
        covariates = np.append(covariates,np.array(range(d)))
        gpycovariates = covariates.reshape(-1, 1)
        samples = arm.model.posterior_samples_f(gpycovariates, size=1)
        samples = samples.reshape(1,len(covariates))[0]
        arm.sampledfunc = dict(zip(covariates, samples))


# define function to obtain the 'sampled' value
def TSSeq(myseq, t, K, Z):
    d = len(myseq)
    zseq = g.ZPlayedSeq(myseq, Z)
    mysample = np.zeros(d)
    for i in range(d):
        mysample[i] = myseq[i].sampledfunc[zseq[i]]
    mytotsample = sum(mysample)
    return mytotsample


# define function to find best sampled sequence
def GetTSSeq(Arms, d, t, K, Z):
    tss = {}
    for p in itertools.permutations(Arms, d):
        tss[p] = TSSeq(p, t, K, Z)
    myseq = max(tss.iteritems(), key=itemgetter(1))[0]
    return myseq

    
# define function to find best sampled sequence when repetitions are allowed
def GetTSSeqRep(Arms, d, t, K, Z):
    tss = {}
    for p in itertools.product(Arms, repeat=d):
        tss[p] = TSSeq(p, t, K, Z)
    myseq = max(tss.iteritems(), key=itemgetter(1))[0]
    return myseq


# dRGP-TS algorithm
# define function to run dRGP-TS once
def RGPTSds(T, Arms, d, K, Z, repetitions=False):
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
                zcurr = [arm.z for arm in Arms]
                SampleFuncts(Arms, d, zcurr, Z)
                g.UpdateErews(Arms)
                myseq = GetTSSeqRep(Arms, d, t, K, Z)
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
            seqpoint = (t-K) % d #find out where in the sequence we are
            if seqpoint == 0:
                zcurr = [arm.z for arm in Arms]
                SampleFuncts(Arms, d, zcurr, Z)
                g.UpdateErews(Arms)
                myseq = GetTSSeq(Arms, d, t, K, Z)
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
