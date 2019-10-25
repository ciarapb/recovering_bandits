#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
run the 1RGP-UCB algorithm in various parametric settings
run for all lengthscales
"""

# import modules
import setups.rgp_logistic_l25_setup as rl25
import setups.rgp_logistic_l5_setup as rl5
import setups.rgp_logistic_l75_setup as rl75
import setups.rgp_gamma_l25_setup as rga25
import setups.rgp_gamma_l5_setup as rga5
import setups.rgp_gamma_l75_setup as rga75
import functs.rgpucb_functs as f
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
# jobs 1,..., 3 are logistic - lengthscales 1: l=2.5, 2: l=5, 3: l=7.5
# jobs 4,..., 6 are gamma - lengthscales 4: l=2.5, 4: l=5, 6: l=7.5
job = 1


""" define arms for the different settings """
if job==1:
	# 1st job is logistic l=2.5

	# set the seed
	np.random.seed(2374)

	# define parameters
	K = rl25.K
	Z = rl25.maxwait

	# take the arms set from rl
	Arms = rl25.Arms
elif job==2:
	# 2nd job is logistic l=5

	# set the seed
	np.random.seed(7522)

	# define parameters
	K = rl5.K
	Z = rl5.maxwait

	# take the arms set from rl
	Arms = rl5.Arms
elif job==3:
	# 3rd job is logistic l=7.5

	# set the seed
	np.random.seed(3417)

	# define parameters
	K = rl75.K
	Z = rl75.maxwait

	# take the arms set from rl
	Arms = rl75.Arms
elif job==4:
	# 4th job is gamma l=2.5

	# set the seed
	np.random.seed(8461)

	# define parameters
	K = rga25.K
	Z = rga25.maxwait

	# take the arms set from rl
	Arms = rga25.Arms
elif job==5:
	# 5th job is gamma l=5

	# set the seed
	np.random.seed(7834)

	# define parameters
	K = rga5.K
	Z = rga5.maxwait

	# take the arms set from rl
	Arms = rga5.Arms
elif job==8:
	# 6th job is gamma l=7.5

	# set the seed
	np.random.seed(6712)

	# define parameters
	K = rga75.K
	Z = rga75.maxwait

	# take the arms set from rl
	Arms = rga75.Arms



""" run 1RGP-UCB multiple times """
CUMREG = []
CUMREW = []
myseeds = random.sample(range(1, 10000), rep) #get a sequence of seeds to run repititions on
for r in range(rep):
	np.random.seed(myseeds[r])
	reg, rew = f.RGPUCB(T, Arms, K, Z)
	cumreg = np.cumsum(reg)
	cumrew = np.cumsum(rew)
	CUMREG.append(cumreg)
	CUMREW.append(cumrew)
	[arm.reset() for arm in Arms]
	print r

# calculate statistics of output
meancumregRGPUCB = np.mean(CUMREG, axis=0)
meancumrewRGPUCB = np.mean(CUMREW, axis=0)
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
filenames = ['logistic_l1', 'logistic_l5', 'logistic_l10', 'gamma_l1', 'gamma_l5', 'gamma_l10', 'logistic_l75', 'gamma_l75', 'logistic_l25', 'gamma_l25']
myname = filenames[job-1]
np.savez('saves/rgpucb_t_%s' %myname, meancumregRGPUCB=meancumregRGPUCB, meancumrewRGPUCB=meancumrewRGPUCB, rep=rep,
		 quantCumReg=quantCumReg, quantCumRew=quantCumRew)

