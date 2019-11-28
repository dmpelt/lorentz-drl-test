#-----------------------------------------------------------------------
#Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/msdnet/
#License: MIT
#
#This file is part of MSDNet, a Python implementation of the
#Mixed-Scale Dense Convolutional Neural Network.
#-----------------------------------------------------------------------

"""
Example 01: Train a network for regression
==========================================

This script trains a MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata.py first to generate required training data.
"""

# Import code
import msdnet
from pathlib import Path
import sys

datdir = sys.argv[1]

# Define dilations in [1,10] as in paper.
dilations = msdnet.dilations.IncrementDilations(5)

# Create main network object for regression, with 100 layers,
# [1,10] dilations, 1 input channel, 1 output channel, using
# the GPU (set gpu=False to use CPU)
n = msdnet.network.NumberMSDNet(100, dilations, 2, 1, gpu=True)

# Initialize network parameters
n.initialize()

# Define training data
# First, create lists of input files (noisy) and target files (noiseless)
flsin = sorted(Path(datdir).glob('inp*.tiff'))
flstg = sorted(Path(datdir).glob('tar*.tiff'))
# Create list of datapoints (i.e. input/target pairs)
dats = []
for i in range(len(flsin)-100):
    # Create datapoint with file names
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    # Augment data by rotating and flipping
    dats.append(d)
datsv = []
for i in range(len(flsin)-100,len(flsin)):
    # Create datapoint with file names
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    # Augment data by rotating and flipping
    datsv.append(d)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# dats = msdnet.utils.load_simple_data('train/noisy/*.tiff', 'train/noiseless/*.tiff', augment=True)

# Normalize input and output of network to zero mean and unit variance using
# training data images
# n.normalizeinout(datsv)

# n.gam_in[1] = 1
# print(n.gam_in, n.gam_out, n.off_in, n.off_out)

# Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats,100)

# Validate with Mean-Squared Error
val = msdnet.validate.MSEValidation(datsv)

# Use ADAM training algorithms
t = msdnet.train.AdamAlgorithm(n)

# Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_regr_'+datdir+'.txt')
# Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLogger('log_regr_'+datdir, onlyifbetter=True)

# Train network until program is stopped manually
# Network parameters are saved in regr_params.h5
# Validation is run after every len(datsv) (=25)
# training steps.
msdnet.train.train(n, t, val, bprov, 'regr_params_'+datdir+'.h5',loggers=[consolelog,filelog,imagelog], val_every=1, progress=True)
