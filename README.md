# recovering_bandits
Code corresponding to the paper: C.Pike-Burke &amp; S.Grunewalder, Recovering Bandits, NeurIPS (2019).

This repository contains all the code to run the experiments included in the above paper.

It is in Python 2.7 and requires the GPy module (https://sheffieldml.github.io/GPy/).

The code is written to be run in a directory with the following structure:
```
main
 ├── setups       # files that contain the construction of arms and definition of priors
 ├── functs       # files that contain functions to run the algorithms
 ├── saves        # folder for saving output into
 ├── make_plots   # iPython notebook files for making the plots in the paper
 └── figures      # folder to save figures into
```
In the main directory, there are the wrapper functions to play each algorithm many times
