#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for using the Optimistic Planning procedure 
within dRGP-TS
"""

# add necessary modules
import functs.gen_functs_opplan as g
import numpy as np
from operator import itemgetter
from operator import attrgetter


# define function to get bound on how much extra reward can be gained by playing
# an arm at some point in the next l steps
def Bound(arm, zcurr, l, Z):
    if l==0:
        mybound = 0.
    else:
        covariates = [min(zcurr[arm.index] + i,Z) for i in range(l)]
        covariates = covariates + range(l)
        mybound = max([arm.sampledfunc[c] for c in covariates])
    return mybound


# define function to sample from the functions at the appropriate values
def SampleFuncts(Arms, d, zcurr, Z):
    for arm in Arms:
        covariates = np.array([min(zcurr[arm.index] + i,Z) for i in range(d)])
        covariates = np.append(covariates,np.array(range(d)))
        gpycovariates = covariates.reshape(-1, 1)
        samples = arm.model.posterior_samples_f(gpycovariates, size=1)
        samples = samples.reshape(1,len(covariates))[0]
        arm.sampledfunc = dict(zip(covariates, samples))



# define function to get sequence from optimistic planning
def OptimisticPlanning(numpols, zcurr, d, Arms, Z):
    # initialize lists of sequences
    myseqs = []
    myoldseqs = []
    foundoptpol = 0

    # get initial sequences of just one arm
    for arm in Arms:
        myind = [arm.index]
        myval = arm.sampledfunc[zcurr[arm.index]]
        myzcurr = [min(zcurr[i] + 1, Z) for i in range(len(zcurr))]
        myzcurr[arm.index] = 0.
        myseq = g.Sequence(myind, myval, myzcurr)
        boundval = max([Bound(arm,myseq.zcurr,int(d-myseq.depth), Z) for arm in Arms])
        myseq.b = myseq.value + (d-myseq.depth)*boundval
        myseqs = myseqs + [myseq]

    n = len(Arms)

    while n<=numpols and foundoptpol==0:
        # select sequence to expand
        myparent = max(myseqs, key=attrgetter('b'))
        # if parent is of depth d stop and output parent
        if myparent.depth >= d:
            foundoptpol=1
            myoptpol = myparent
            myd = myparent.depth
            break
        # add child policies to list
        for arm in Arms:
            myinds = myparent.indicies + [arm.index]
            myvalue = myparent.value + arm.sampledfunc[myparent.zcurr[arm.index]]
            myzcurr = [min(myparent.zcurr[i]+1,Z) for i in range(len(zcurr))]
            myzcurr[arm.index] = 0.
            myseq = g.Sequence(myinds, myvalue, myzcurr)
            boundval = max([Bound(arm, myseq.zcurr, int(d-myseq.depth), Z) for arm in Arms])
            myseq.b = myseq.value + (d-myseq.depth)*boundval
            myseqs = myseqs + [myseq]
            n+=1
            if n>numpols:
                break
        myseqs.remove(myparent)
        myoldseqs = myoldseqs + [myparent]

    # give value to output if we didnt stop early
    if foundoptpol==0:
        maxd = max(myoldseqs, key=attrgetter('depth')).depth
        myoptpol = max((seq for seq in myoldseqs if seq.depth==maxd), key=attrgetter('b'))
        myd = maxd

        # if we didn't stop early need to select arms for rest of sequence - do this greedily
        for i in range(myoptpol.depth, d):
            nextrews = {}
            for arm in Arms:
                nextrews[arm.index] = arm.sampledfunc[zcurr[arm.index]]
            mynextarm = max(nextrews.iteritems(), key=itemgetter(1))[0]
            myoptpol.indicies = myoptpol.indicies + [mynextarm]
            zcurr = [min(zcurr[i]+1,Z) for i in range(len(zcurr))]
            zcurr[mynextarm] = 0.

    # translate sequence of indicies into list of arms
    myseq = [Arms[i] for i in myoptpol.indicies]
    print n, foundoptpol
    return myseq, n, myd


# Optimistic Planning in dRGP-TS algorithm
# define function to run dRGP-TS with OP once
def RGPTSop(T, Arms, d, K, Z, numpols):
    # initialize parameters
    rew = np.zeros(T)
    numlookaheads = []
    myds = []

    # run inital rounds
    for t in range(0, K):
        myarm = Arms[t]
        ynew = g.PlayArm(myarm, myarm.z)
        myarm.UpdatePosterior(myarm.z, ynew)
        g.UpdateErews(Arms)
        rew[t] = ynew
        g.UpdateZs(Arms, myarm, Z)
    
    # run full algorithm with repetitions allowed in sequences
    for t in range(K, T):
        seqpoint = (t-K) % d  #find out where in the sequence we are
        if seqpoint == 0:
            zcurr = [arm.z for arm in Arms]
            SampleFuncts(Arms, d, zcurr, Z)
            g.UpdateErews(Arms)
            myseq, myn, myd = OptimisticPlanning(numpols, zcurr, d, Arms, Z)
            numlookaheads.append(myn)
            myds.append(myd)
        myarm = myseq[seqpoint]
        ynew = g.PlayArm(myarm, myarm.z)
        rew[t] = ynew
        # update posterior - we have already chosen sequence so it doesnt matter if we do this now
        myarm.UpdatePosterior(myarm.z, ynew)
        # update all z's ready for next round
        g.UpdateZs(Arms, myarm, Z)
        # print t,
    
    #avgnumpols = N/(numlookaheads*1.0)

    return rew, numlookaheads, myds  
