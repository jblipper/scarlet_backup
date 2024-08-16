# -*- coding: utf-8 -*-
"""
@author: bbenneke
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt

import sys  #, glob
import numpy as np
#import pandas as pd

from astropy import io #table
from astropy.convolution import (convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel)

import auxbenneke.utilities as ut
from auxbenneke.quickcmds import clf
#from auxbenneke import (pyplanet, thermo, pyspectrum)
from auxbenneke import pyspectrum
from auxbenneke.constants import Rearth

from copy import deepcopy

import scarlet
from time import time, sleep
import os
import h5py

#%%


class opacityPlotter(object):
    '''
    Opacity plotting from SCARLET Lookup Tables
    
    Usage:
    opac=scarlet.opacityPlotter()
    fig,ax,maxValueThisCurve,wavePlotted,sigmaPlotted=opac.compareMoleculeOpacities(iMols=[5,9,21])

    '''
    
    def __init__(self,LUTFile='/Users/bbenneke/Research/GitHub/scarlet_LookUpQuickRead/LookUpQuickRead_R_grid_20210120_0.5_6.0_16.mat'):
        
        print('Loading...')
        t0 = time()
        
        f = h5py.File(LUTFile,'r')
        self.sigma   = np.array(f['sigma_mol'])[:]
        self.wave=np.array(f['Wave_microns']);   self.nWave=len(self.wave)
        ind=3; 
        self.resPowerOpac = self.wave[ind] / (self.wave[ind+1]-self.wave[ind])
        self.TGrid   = np.array(f['LookUpTGrid']); self.nT=len(self.TGrid)
        self.PGrid   = np.array(f['LookUpPGrid']); self.nP=len(self.PGrid)
        self.MolNames= f['LookUpMolNames'][:];     self.nMol=len(self.MolNames)
        f.close()
        
        t1 = time()
        print('Loading time: {0:8.2f} seconds ({1:8.2f} MB/s)\n\n'.format(t1-t0,os.path.getsize(LUTFile)/1024.**2 / (t1-t0)))
        
        self.sigma = np.moveaxis(self.sigma,[0,1,2,3],[0,1,3,2])
        
        
        print(self.TGrid)
        print(self.PGrid)
        
        for iMol, MolName in enumerate(self.MolNames):
            print(iMol, MolName)
    
    
    def plotResPower(self,ax,wave,y,resPower=100,label=None,xlim=None):
        #Plotting
        if resPower is None:
            x=wave
            ax.plot(x,y,label=label)
            ax.set_xlim([x[0],x[-1]])
        else:
            kernel=Gaussian1DKernel(self.resPowerOpac / resPower / 2.35)    # *2 because FWHM = 2 standard deviation
            l = int(kernel.shape[0]/2)
            x = wave[l:-l]
            y = convolve(y, kernel)[l:-l]
            ax.plot(x,y,label=label)
            ax.set_xlim([x[0],x[-1]])
            
        if xlim is not None:
            maxValue=np.max(y[np.where(np.logical_and(wave>xlim[0],wave<xlim[1]))])
    
        return maxValue, x, y
    

    def compareMoleculeOpacities(self,resPower=1000,iT=1,iP=20,iMols=[5,9,21],xlim=[1.0,5.0]):

        figsize=[16,9]
        
        #iMols=range(len(self.MolNames))
        
        fig,ax=plt.subplots(figsize=figsize)
        ax.set_xlabel(r'Wavelength $[\mu m]$')
        ax.set_ylabel(r'Opacity')
        ax.set_yscale('log')
        
        maxOpac=0
        for iMol in iMols:
            maxValueThisCurve, wavePlotted, sigmaPlotted = self.plotResPower(ax,self.wave,self.sigma[iT,iP,iMol,:],resPower=resPower,label=self.MolNames[iMol],xlim=xlim)
            maxOpac = np.max([maxOpac,maxValueThisCurve])
        
        ax.set_xlim(xlim)
        ax.set_ylim(np.array([maxOpac/1e8,maxOpac])*10)    
        ax.set_title('T = {} K,  P = {} mbar'.format(self.TGrid[iT],self.PGrid[iP]/100))
        ax.legend(fontsize=8)
        ax.xaxis.grid()
        ax.yaxis.grid()
        
        return fig,ax,maxValueThisCurve,wavePlotted,sigmaPlotted
    
    
    def plotOverData(self,fig,axData,resPower=1000,iT=1,iP=20,iMols=[21],xlim=[1.0,5.0]):
        
        ax = axData.twinx()  

        ax.set_ylabel(r'Opacity')
        ax.set_yscale('log')

        maxOpac=0
        for iMol in iMols:
            maxValueThisCurve, wavePlotted, sigmaPlotted = self.plotResPower(ax,self.wave,self.sigma[iT,iP,iMol,:],resPower=resPower,label=self.MolNames[iMol],xlim=xlim)
            maxOpac = np.max([maxOpac,maxValueThisCurve])
        
        ax.set_ylim(np.array([maxOpac/1e8,maxOpac])*10)    
        ax.set_title('T = {} K,  P = {} mbar'.format(self.TGrid[iT],self.PGrid[iP]/100))
        ax.legend(fontsize=8)
        
    
    
        


#%%

if __name__ == "__main__":

    opac=opacityPlotter()
    fig,ax,maxValueThisCurve,wavePlotted,sigmaPlotted=opac.compareMoleculeOpacities(iMols=[21])
    
    sys.exit()
    
    
    
    #%% Load observations
    
    specFiles='''
    /Users/bbenneke/Research/GitHub/observations/TRAPPIST_1_g/TRAPPIST_1_g_20230204_V1.spec
    '''
    specFiles=specFiles.replace(' ','').split('\n')[1:-1]
    
    specs=[]
    for specFile in specFiles:
        specs.append(pyspectrum.Spectrum(inputdata=specFile,inputtype='ecsv'))
    
    
    
    #%% Check for correlation
    
    
    fig,ax=plt.subplots()
    ax.plot(wavePlotted, sigmaPlotted)
    ax.set_xlabel(r'Wavelength $[\mu m]$')
    ax.set_ylabel(r'Opacity')
    ax.set_yscale('log')
    ax.set_ylim(np.array([maxValueThisCurve/1e8,maxValueThisCurve])*10)    
    
    
    ax2 = ax.twinx()  
    ax2.set_ylabel('Transit Depth [ppm]')  
    for spec in specs:
        spec.plot(ax=ax2,color='black')
    ax2.set_ylim(6500,8500)
    
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    
    
    sys.exit()
    
    
    
    #%% Compare Temperatures
    
    figsize=[16,9]
    resPower=100
    iMol=0; iP=21
    iTs=range(len(TGrid))
    xlim=[1.05,1.95]
    
    fig,ax=plt.subplots(figsize=figsize)
    ax.set_xlabel(r'Wavelength $[\mu m]$')
    ax.set_ylabel(r'Opacity')
    ax.set_yscale('log')
    
    maxOpac=0
    for iT in iTs:
        maxValueThisCurve = plotResPower(ax,wave,sigma[iT,iP,iMol,:],resPower=resPower,label='{} K'.format(TGrid[iT]),xlim=xlim)
        maxOpac = np.max([maxOpac,maxValueThisCurve])
    
    ax.set_xlim(xlim)
    ax.set_ylim(np.array([maxOpac/1e8,maxOpac])*10)    
    ax.set_title('{} at P = {} mbar'.format(MolNames[iMol],PGrid[iP]/100))
    ax.legend(fontsize=6)
    ax.xaxis.grid()
    ax.yaxis.grid()
    
    
    #%%
    
    # ax2 = ax.twinx()  
    # ax2.set_ylabel('Transit Depth [ppm]')  
    # for spec in specs:
    #     spec.plot(ax=ax2)
    
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    









