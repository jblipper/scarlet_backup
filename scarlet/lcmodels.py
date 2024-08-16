# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt
#plt.locator_params(axis = 'x', nbins = 4)

#import matplotlib.dates as mdates
#from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec
from matplotlib import colors, ticker, cm #need for plotCarmaCloud()

#import scipy.io as spio7
from scipy.io.idl import readsav

from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel
import astropy
from scipy.interpolate import CubicSpline

#import astropy.io.fits as pf
#from astropy.time import Time

import pdb
#import pickle
#from pprint import pprint

#from utilities import remOutliers, calcChi2, find_nearest, calclnlike

import os
import sys

import pandas as pd
from copy import deepcopy
    
from bisect import bisect_left,bisect_right
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel
#import pyspectrum #, loadExoFit
from scipy.optimize import minimize
import time

import auxbenneke.utilities as ut
from auxbenneke.constants import pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom

from scarlet import radutils as rad
import scipy.io as sio
from astropy.io import fits

from time import time, sleep
import h5py
from bisect import bisect

import pkg_resources
from astropy import table 
from scipy import interpolate

from sys import stdout

from copy import deepcopy

import warnings

#import starry 


#%%


def computeLCs(atm,modelSetting):
    return lcs



def createWaveDepMap():
    
    thermalSpectra = np.ones([3,1000])
    
    thermalSpectra[0,:] = 1.0
    thermalSpectra[1,:] = 2.0
    thermalSpectra[2,:] = 3.0
    
    
    
    return map




#%% 

    
if __name__ == "__main__":

    #Make some starry light curves
    #Plotting
    print('Make some starry light curves')
