#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
run the dRGP-TS algorithm with optimistic planning
in various settings
"""

# import modules
import setups.rgp_gp_l4_setup as rgp5
import setups.rgp_gp_K30_l4_setup as rgp30 
import functs.drgpts_opplan_functs as f
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import random
import itertools
#reload(r)
#reload(rf)


# how many times to run the algorithm
rep = 50

# define necessary constants
T = 1000

""" select job """
# jobs 1,..., 12 are K=10, d=4 
# jobs 13,..,26 are K=10, d=8
# jobs 27,...,45 are K=20, d=4
job = 1

""" define arms for the different settings """
if  1 <= job <= 12:
	# 1st lot of jobs are K=10, d=4
	# set the seed
	np.random.seed(5389)

	# define parameters
	K = rgp5.K
	Z = rgp5.maxwait

	# take the arms set from rgp
	Arms = rgp5.Arms

	# define d
	d=4

	# define how many policies to look at
	allnumpols = [100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
	
	jobnum = job-1
	numpols = allnumpols[jobnum]
elif 13 <= job <= 26:
	# 2nd lot of jobs are gp K=10, d=8
	# set the seed
	np.random.seed(5389)

	# define parameters
	K = rgp5.K
	Z = rgp5.maxwait

	# take the arms set from rgp
	Arms = rgp5.Arms

	# define d
	d=8

	# define how many policies to look at
	allnumpols = [100,500,1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]

	jobnum = job-13
	numpols = allnumpols[jobnum]
elif 27 <= job <= 45:
	# 3rd lot of jobs are gp K=30, d=4
	# set the seed
	np.random.seed(1239)

	# define parameters
	K = rgp30.K
	Z = rgp30.maxwait

	# take the arms set from rgp
	Arms = rgp30.Arms

	# define d
	d=4

	# define how many policies to look at
	allnumpols = [100,500,1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,
					110000,120000,130000,140000,150000]

	jobnum = job-27
	numpols = allnumpols[jobnum]
	


""" run rgpucb multiple times """
#CUMREG = []
CUMREW = []
NUMPOLS = []
MYDS = []
myseeds = random.sample(range(1, 10000), rep) #get a sequence of seeds to run repititions on
for r in range(rep):
	np.random.seed(myseeds[r])
	rew, numpolseval, myds = f.RGPTSop(T, Arms, d, K, Z, numpols)
	cumrew = np.cumsum(rew)
	CUMREW.append(cumrew)
	NUMPOLS = NUMPOLS + numpolseval
	MYDS = MYDS + myds
	[arm.reset() for arm in Arms]
	print r

# calculate statistics of output
meancumrewRGPUCB = np.mean(CUMREW, axis=0)
q25rewRGPUCB, q975rewRGPUCB = np.percentile(CUMREW, [2.5,97.5], axis=0)
q5rewRGPUCB, q95rewRGPUCB = np.percentile(CUMREW, [5,95], axis=0)
quantCumRew = {'2.5':q25rewRGPUCB,
			'5':q5rewRGPUCB,
			'95':q95rewRGPUCB,
			'97.5':q975rewRGPUCB}


# save output
np.savez('saves/rgpts_opplan_l4_d%d_K%d_%d' %(d,K,jobnum), meancumrewRGPUCB=meancumrewRGPUCB, rep=rep,
		 quantCumRew=quantCumRew, NUMPOLS=NUMPOLS, MYDS=MYDS)


