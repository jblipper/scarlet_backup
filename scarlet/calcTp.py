# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:17:57 2016

@author: bbenneke
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
#plt.locator_params(axis = 'x', nbins = 4)

#import matplotlib.dates as mdates
#from matplotlib.ticker import FuncFormatter
#from matplotlib import gridspec

#import scipy.io as spio
#from scipy.io.idl import readsav

from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel

#import astropy.io.fits as pf
#from astropy.time import Time

#import pdb
#import pickle
#from pprint import pprint

from auxbenneke.utilities import remOutliers, calcChi2, find_nearest, calclnlike
import auxbennek.utilities as ut


from auxbenneke.constants import day, Rearth, Mearth, Mjup, Rjup, sigmaSB
#import emcee
#import triangle

#import os
#import sys, select

#import pandas as pd
#from copy import deepcopy

#from bisect import bisect_left,bisect_right
#from astropy.table import Table

from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel
#import pyspectrum, pyplanet #, loadExoFit
#from scipy.optimize import minimize

jdref=2450000
big=1e10

#import time
from scarlet import radutils as rad


#%%

fig,ax=ut.newFig(xlabel='T',ylabel='tau',reverse=[False,True],log=[False,True])

tau = np.logspace(-5,8,num=100)
T=rad.TpGrey(tau,1000)
ax.plot(T,tau)
ax.set_xlim([0,3000])

fig,ax=ut.newFig(xlabel='Temperature [K]',ylabel='Pressure [bars]',reverse=[False,True],log=[False,True])
p = np.logspace(-4.5,8.5,num=100)
for val in [0.1,0.5,1.0]:
    T=rad.TpTwoVis(p,beta=val);  ax.plot(T,p/1e5, label='beta = '+str(val))
T=rad.TpTwoVis(p);  ax.plot(T,p/1e5,'k--')
ax.legend(loc='best')

fig,ax=ut.newFig(xlabel='Temperature [K]',ylabel='Pressure [bars]',reverse=[False,True],log=[False,True])
p = np.logspace(-4.5,8.5,num=100)
for val in [3e-4,3e-3,3e-2,3e-1]:
    T=rad.TpTwoVis(p,kappaIR=val);  ax.plot(T,p/1e5, label='kappaIR = '+str(val))
T=rad.TpTwoVis(p);  ax.plot(T,p/1e5,'k--')
ax.legend(loc='best')

fig,ax=ut.newFig(xlabel='Temperature [K]',ylabel='Pressure [bars]',reverse=[False,True],log=[False,True])
p = np.logspace(-4.5,8.5,num=100)
for val in [0.01,0.158,1,10]:
    T=rad.TpTwoVis(p,gamma1=val);  ax.plot(T,p/1e5, label='gamma1 = '+str(val))
T=rad.TpTwoVis(p);  ax.plot(T,p/1e5,'k--')
ax.legend(loc='best')

fig,ax=ut.newFig(xlabel='Temperature [K]',ylabel='Pressure [bars]',reverse=[False,True],log=[False,True])
p = np.logspace(-4.5,8.5,num=100)
for val in [0.2,0.5,0.8]:
    T=rad.TpTwoVis(p,gamma1=1,alpha=val);  ax.plot(T,p/1e5, label='alpha = '+str(val))
T=rad.TpTwoVis(p,gamma1=1);  ax.plot(T,p/1e5,'k--')
ax.legend(loc='best')


#
#fig,ax=ut.newFig(xlabel='Temperature [K]',ylabel='Pressure [bars]',reverse=[False,True],log=[False,True])
#p = np.logspace(-4.5,8.5,num=100)
#for val in [0.4,0.5,10]:
#    T=rad.TpTwoVis(p,alpha=val);  ax.plot(T,p/1e5, label='alpha = '+str(val))
#T=rad.TpTwoVis(p);  ax.plot(T,p/1e5,'k--')
#ax.legend(loc='best')