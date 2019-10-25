#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
run the rogueUCB-tuned algorithm in various parametric settings
"""

# import modules
# only need to input one for each logistic and gamma as parametric bits are the same
import setups.rgp_logistic_l5_setup as rl
import setups.rgp_gamma_l5_setup as rga 
import functs.rogueucbtuned_logistic_functs as f
import functs.rogueucbtuned_gamma_functs as fg
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
# jobs 1 is logistic
# jobs 2 is gamma
job = 1


""" define arms for the different settings and run algorithm"""
# here need to run a different algorithm in each setting
if job==1:
	# 1st job is logistic
	# set the seed
	np.random.seed(2374)

	# define parameters
	K = rl.K
	Z = rl.maxwait

	# take the arms set from rl
	Arms = rl.Arms

	# eta is maximum value of kl divergence
	eta = 1/(2*rl.noisevar)

	# run RogueUCB-Tuned multiple times
	CUMREG = []
	CUMREW = []
	myseeds = random.sample(range(1, 10000), rep) #get a sequence of seeds to run repititions on
	for r in range(rep):
		np.random.seed(myseeds[r])
		reg, rew = f.RogueUCBTuned(T, Arms, K, Z, eta, rl.noisevar)
		cumreg = np.cumsum(reg)
		cumrew = np.cumsum(rew)
		CUMREG.append(cumreg)
		CUMREW.append(cumrew)
		[arm.reset() for arm in Arms]
		print r

	# calculate statistics of output
	meancumregRogueUCBTuned = np.mean(CUMREG, axis=0)
	meancumrewRogueUCBTuned = np.mean(CUMREW, axis=0)
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
elif job==2:
	# 2nd job is gamm
	# set the seed
	np.random.seed(6592)

	# define parameters
	K = rga.K
	Z = rga.maxwait

	# take the arms set from rga
	Arms = rga.Arms

	# eta is maximum value of kl divergence
	eta = 1/(2*rga.noisevar)

	# run the RogueUCB-Tuned algorithm multiple times
	CUMREG = []
	CUMREW = []
	myseeds = random.sample(range(1, 10000), rep) #get a sequence of seeds to run repititions on
	for r in range(rep):
		np.random.seed(myseeds[r])
		reg, rew = fg.RogueUCBTuned(T, Arms, K, Z, eta, rga.noisevar)
		cumreg = np.cumsum(reg)
		cumrew = np.cumsum(rew)
		CUMREG.append(cumreg)
		CUMREW.append(cumrew)
		[arm.reset() for arm in Arms]
		print r

	# calculate statistics of output
	meancumregRogueUCBTuned = np.mean(CUMREG, axis=0)
	meancumrewRogueUCBTuned = np.mean(CUMREW, axis=0)
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
np.savez('saves/rogueucbtuned_%s' %myname, meancumregRogueUCBTuned=meancumregRogueUCBTuned, meancumrewRogueUCBTuned=meancumrewRogueUCBTuned, rep=rep,
		 quantCumReg=quantCumReg, quantCumRew=quantCumRew)