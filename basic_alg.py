#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
run the ucbz algorithm in various settings
"""

# import modules
import setups.rgp_logistic_l1_setup as rl
import setups.rgp_gamma_l1_setup as rga 
import functs.ucbz_functs as f
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import random


# how many times to run the algorithm
rep = 500

# define necessary constants
T = 1000

""" select job """
# job=1 is logistic
# job=2 is gamma
job = 1


""" define arms for the different settings """
if job==1:
	# 1st job is logistic
	# set the seed
	np.random.seed(2374)

	# define parameters
	K = rl.K
	Z = rl.maxwait
	sigma2 = rl.noisevar

	# take the arms set from rl
	Arms = rl.Arms
elif job==2:
	# 2nd job is gamma
	# set the seed
	np.random.seed(6592)

	# define parameters
	K = rga.K
	Z = rga.maxwait
	sigma2=rga.noisevar

	# take the arms set from rga
	Arms = rga.Arms


# run the algorithm
""" run basic algorithm multiple times """
CUMREG = []
CUMREW = []
myseeds = random.sample(range(1, 10000), rep) #get a sequence of seeds to run repititions on
for r in range(rep):
	# myArms = copy.deepcopy(Arms)
	np.random.seed(myseeds[r])
	reg, rew = f.UCB(T, Arms, K, Z, sigma2)
	cumreg = np.cumsum(reg)
	cumrew = np.cumsum(rew)
	CUMREG.append(cumreg)
	CUMREW.append(cumrew)
	[arm.reset() for arm in Arms]
	print r

# calculate statistics of the output
meancumregBasicAlg = np.mean(CUMREG, axis=0)
meancumrewBasicAlg = np.mean(CUMREW, axis=0)
q25regRGPUCB, q975regRGPUCB = np.percentile(CUMREG, [2.5,97.5], axis=0)
q25rewRGPUCB, q975rewRGPUCB = np.percentile(CUMREW, [2.5,97.5], axis=0)
q5regRGPUCB, q95regRGPUCB = np.percentile(CUMREG, [5,95], axis=0)
q5rewRGPUCB, q95rewRGPUCB = np.percentile(CUMREW, [5,95], axis=0)
quantCumReg = {'2.5':q25regRGPUCB,
			'5':q5regRGPUCB,
			'95':q95regRGPUCB,
			'97.5':q975regRGPUCB}
quantCumRew = {'2.5':q25rewRGPUCB,
			'5':q5rewRGPUCB,
			'95':q95rewRGPUCB,
			'97.5':q975rewRGPUCB}

# save output
filenames = ['logistic', 'gamma']
myname = filenames[job-1]
np.savez('saves/basic_alg_%s' %myname, meancumregBasicAlg=meancumregBasicAlg, meancumrewBasicAlg=meancumrewBasicAlg, rep=rep,
		 quantCumReg=quantCumReg, quantCumRew=quantCumRew)