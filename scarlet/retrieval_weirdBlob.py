# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:17:57 2016

@author: bbenneke
"""


from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import numexpr as ne
ne.set_num_threads(1) #ne.utils.set_vml_num_threads(1)

import matplotlib.pyplot as plt
import matplotlib as mpl
#plt.locator_params(axis = 'x', nbins = 4)

#import matplotlib.dates as mdates
#from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec

#import scipy.io as spio
from scipy.io.idl import readsav
from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel

#import astropy.io.fits as pf
#from astropy.time import Time

import pdb
#import pickle
#from pprint import pprint

from auxbenneke.utilities import remOutliers, calcChi2, find_nearest, calclnlike
import auxbenneke.utilities as ut

from auxbenneke.constants import day, uAtom, Mearth
from auxbenneke.constants import *
import emcee
import nestle
import auxbenneke.triangle

import os
#import sys, select

import pandas as pd

from copy import deepcopy

import pkg_resources
from guppy import hpy

from bisect import bisect_left,bisect_right

import scarlet

from scarlet import retrievalPlotTools
  
#import h5py
import time
import math

from astropy import table
import subprocess
import corner
import h5py

#from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool

jdref=2450000
big=1e10



#%% to calculate size correctly
import sys
from numbers import Number
from collections import Set, Mapping, deque

try: # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: # Python 3
        zero_depth_bases = (str, bytes, Number, range, bytearray)
        iteritems = 'items'

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

#%% Retrieval class

class retrieval(object):
    '''
    Useful outputs
        #fit.bestfit     : Gaussian errors, MaxLike, MaxProb as a pandas dataframe
        #fit.bestfitRun  : transit light curve parameters etc.
        #fit.panda       : all MCMC samples as a pandas dataframe
                          --> unchanged complete chain of walkers including burn-in and outliers (not used much here)
        #fit.df          : like fit.panda, but burn-in, stuck walkers, and outliers removed
        #fit.samples     : samples from random set of walkers including spectra.
    '''
    
    def __init__(self,name,pla,modelSetting=None,filename=None,basedirec='../scarlet_results/',direc=None,defaultValueForPara=dict(),
                 waveRange=None,resolution=None,MoleculeListFile=None,TGridFile=None,
                 nBlobLayers=6,blobResPower=300,presOfLayForPandas=10*100,nLay=40,
                 saveSteps=np.array([1,20,100]),nSaveBlobs=1000,plotting=True,DoNotBlobArrays=False,
                 MolNames    = np.array(['He','H2','CH4','C2H2','O2','OH','H2O','CO','CO2','NH3','HCN','H2S','PH3','Na','K','N2']),
                 AbsMolNames = np.array(['CH4','C2H2','O2','OH','H2O','CO','CO2','NH3','HCN','H2S','PH3','Na','K']),pressureLevels=None):

        if name!='empty':
            self.name=name
            self.pla=pla
    
            self.specs=[]
            self.nspecs=0
    
            self.para = []          #one for each fitting parameter
            self.sysModel = []      #one for each data set
            self.paraBounds = []    #one for each data set
            self.defaultValueForPara = defaultValueForPara
    
            self.nsteps=0       #number of iterations in emcee
            self.istep=0
            self.saveSteps=saveSteps  #sets how frequently the pkl file is saved
            self.previStep=0  #to avoid multiple writes of intermediate output
            self.panda=None
            self.newPanda=None
            self.nSaveBlobs = nSaveBlobs
            self.plotting=plotting
            self.MolNames = MolNames
            self.AbsMolNames = AbsMolNames
            
            self.printIterations=True  #Default
            
            self.DoNotBlobArrays = DoNotBlobArrays
    
    
            datetxt=ut.datestr()+'_'
            if filename is None:
                self.filename=datetxt+self.name
            else:
                self.filename=filename
                
            self.scarletpath = os.path.dirname(pkg_resources.resource_filename('scarlet', ''))
            if basedirec.endswith('scarlet_results/'):
                basedirec = self.scarletpath+'/../scarlet_results/'
            
            if direc is None:
                self.direc=basedirec+self.filename
            else:
                self.direc=basedirec+direc
            if not os.path.exists(basedirec):
                try:
                    os.makedirs(basedirec)
                except:
                    print('Directory already exists: '+basedirec)
            if not os.path.exists(self.direc):
                try:
                    os.makedirs(self.direc)
                except:
                    print('Directory already exists: '+self.direc)
            self.filebase = self.direc + '/' + self.filename + '_'
    
            #Settings for atmosphere forward model
            self.MoleculeListFile=MoleculeListFile
            self.waveRange=waveRange
            self.resolution=resolution
            self.nLay=nLay
            self.pressureLevels=pressureLevels
    
            if modelSetting is not None:
                self.modelSetting = modelSetting
            else:
                self.modelSetting=dict()
                self.modelSetting['ComposType']='ChemEqui'  # 'WellMixed';  'ChemEqui',  'SetByFile', 'PhotoChem'
                self.modelSetting['TempType']='TeqUniform'  # 'TeqUniform';  'SemiGray',  'NonGray', 'SetByFile', parameters'
                self.modelSetting['CloudTypes']=[1,1,0,0]  #   [1,1,1,1] for 'pcloud', 'chaze', 'MieScattering', 'carma'

            #Load TpGrid LUT
            if modelSetting['TempType']=='TintHeatDist':
                self.TpLUT=ut.loadpickle(modelSetting['TGridFile'])
                
            #Init Atmosphere Model
            self.initAtmosphereModel()    

            #Blob
            self.nBlobLayers=nBlobLayers
            self.blobLayers = np.linspace(0,self.atm.nLay-1,self.nBlobLayers).astype(int)
            
            self.blobResPower=blobResPower
            self.blobSmoothing = self.atm.resPower/self.blobResPower
            self.blobKernel=Gaussian1DKernel(self.blobSmoothing)
            self.edge=4
            self.wavesm=self.atm.wave[::int(self.blobSmoothing)][self.edge:-self.edge]
    
            self.indLayToPandas = np.searchsorted(self.atm.p[self.blobLayers],presOfLayForPandas,side='right')
            self.pLayToPandas = self.atm.p[self.blobLayers[self.indLayToPandas]]

            self.MaxLnProb = -np.inf
            self.MaxLnLike = -np.inf
            self.bestLikeBlob = {} #blob with best lnlike
            self.bestProbBlob = {} #blob with best lnprob
            
            self.lastSaveToPanda = -1 #last iteration at which the information in the blobs was added to self.panda

    def initAtmosphereModel(self):
        if type(self.nLay) is int:
            if self.pressureLevels is None:
                pressureLevels=np.logspace(-5,9,self.nLay)
            else:
                pressureLevels=self.pressureLevels
        else:
            pressureLevels=np.r_[np.logspace(-5,1,self.nLay[0]),np.logspace(1,5,self.nLay[1])[1:-1],np.logspace(5,9,self.nLay[2])]
        self.atm = scarlet.atmosphere(filename='',basedirec='',direc=self.direc+'/atm',
                                      waveRange=self.waveRange,resolution=self.resolution,MoleculeListFile=self.MoleculeListFile,
                                      nLay=self.nLay,mieCondFiles=self.modelSetting['mieCondFiles'],pressureLevels=pressureLevels,
                                      MolNames=self.MolNames, AbsMolNames=self.AbsMolNames)
    
    def autoLabel(self,includeSpecInfo=False):
        txts=[]

        if includeSpecInfo:
            txtspec=[]
            for spec in self.specs:
                if spec.meta['spectype']=='dppm':
                    txtspec.append('Transit')
                if spec.meta['spectype']=='thermal':
                    txtspec.append('Thermal')
            txts.append('+'.join(txtspec))
                
        if self.modelSetting['ComposType']=='WellMixed':
            txts.append('+'.join(self.MolNamesToFit))
        if self.modelSetting['ComposType']=='ChemEqui':
            txts.append('Chem. Consistent')
        if self.modelSetting['CloudTypes'][0]:
            txts.append('gray clouds')
        if self.modelSetting['CloudTypes'][1]:
            txts.append('hazes')
        if self.modelSetting['CloudTypes'][2]:
            txts.append('Mie clouds')
        if np.any(self.modelSetting['CloudTypes'])==False:
            txts.append('no clouds')
        return ', '.join(txts)

    def runDate(self):
        return self.filename[:14]
        
    def addSpec(self,spec):
        #add spec to list of spectra specs
        self.specs.append(spec)
        self.nspecs=len(self.specs) #number of spectra data sets 
        #self.npoints=np.sum([len(x.bjd) for x in self.lcs])  #total number of data points

        #add fitting parameters for this systematics model to list of fitting parameter
        if spec.meta['sysModel']['addParas'] is None:
            n1 = len(self.para)
            n2 = len(self.para)
            self.paraBounds.append([n1,n2])
        else:
            n1 = len(self.para)
            newparas=spec.meta['sysModel']['addParas'](spec)  #Calls addParas method specified by user in lc
            for newpara in newparas:
                newpara.symbol=newpara.symbol+'_'+str(self.nspecs)
            self.para=self.para + newparas

            n2 = len(self.para)
            self.paraBounds.append([n1,n2])



    def dispFittingPara(self):
        print('   {:18s} | {:15s} |  {:12s}   |  {:12s}  |  {:12s}  |  {:12s}'.format('label','symbol','guess','low','high','srange'))

        for i,para in enumerate(self.para):
            print('{:2d} {:18s} | {:15s} |  {:12g}   |  {:12g}  |  {:12g}  | {:12g}'.format(i,para.label,para.symbol,para.guess,para.low,para.high,para.srange))

    def listOfPara(self):
        return np.array([x.symbol for x in self.para])


    def runInitGuess(self):
        
        #Check whether RpRs is a free parameter
        if self.fitRadiusInFwdModel:
            print('fitRadiusInFwdModel = True (RpRs is optimized in forward model)')
        else:                                              
            print('fitRadiusInFwdModel = False (RpRs is not optimized in forward model)')
            if np.any(np.array([x.symbol for x in self.para])=='RpRs'):
                print('RpRs is fitted as a free parameter in the retrieval')
            else:
                print('RpRs is fixed to default values')

        #Prepare intruments indexes for evaluation in self.instrResp 
        self.atm.params=self.defaultValueForPara

        print('Initital self.modelSettings:\n')
        ut.pprint(self.modelSetting)
        
        print('\n Initital atm.params:\n')
        ut.pprint(self.atm.params)
        
        param0 = np.array([x.guess for x in self.para])
        astroparams,systparams,scatter,insidePrior=self.splitParamVector(param0)
                
        
        #-- Apply offset to certain instruments if fitted
        if len(self.modelSetting['specOffsets']):
            for i,spec in enumerate(self.specs0):
                instrnames  = set(spec['instrname'])
                for instr in list(instrnames):
                    if instr in self.modelSetting['specOffsets']:
                        ind = np.where(spec['instrname']==instr)
                        self.specs[i]['yval'][ind] = spec['yval'][ind]+astroparams['offset_'+instr]
        
        self.atm.prepInstrResp(self.specs)
        
        
        if self.DoNotBlobArrays:
            prob,lnprior,lnprior_gamma,lnprior_TP,lnlike,chi2,redchi2,outgoingFlux,MuAve,grav,scaleHeight=self.lnprob(param0,plotting=True)
        else:
            prob,lnprior,lnprior_gamma,lnprior_TP,lnlike,chi2,redchi2,outgoingFlux,MuAve,grav,scaleHeight,T,qmol_lay,secEclppm,dppm,thermal=self.lnprob(param0,plotting=True)


        #Check initial guess        
        if np.isfinite(prob) == False:
            self.dispFittingPara()
            raise ValueError('!!! ERROR: Initial Guess Outside Prior Parameter Space!')

        #Plot guess        
        if self.plotting:
            #make plots
            self.makeAtmPlotsForLastRun(addToFileBase='InitialGuessFit')
                             
        #Set default guess for RpRs for radius fit in fwd model 
        if self.fitRadiusInFwdModel:
            self.defaultValueForPara['RpRs']=self.modelRpRs
        
        if self.plotting:
            figures,ax = self.plotLastLikelihoodEvaluation(extraTitle='Initial Guess:')
            for i,fig in enumerate(figures):
                fig.savefig(self.filebase + 'InitialGuessFit_Spec' + str(i) + '.pdf')

    def emceefit(self,nsteps=100,nsubsteps=None,nwalkers=None,printIterations=True,pos0=None):
        # Set up the sampler.
        param0 = np.array([x.guess for x in self.para])
        delta = np.array([x.srange for x in self.para])
        self.parasymbols=np.array([x.symbol for x in self.para])
        self.paralabels =np.array([x.label for x in self.para])

        self.ndim = len(param0)
        if nwalkers is None:
            self.nwalkers = self.ndim*2
        else:
            self.nwalkers = nwalkers
        self.nsteps=nsteps
        self.nsubsteps = nsubsteps
        
        #specify name and dtype of each blob you want to keep track of
        dt = [('lnprior',float),('lnprior_gamma',float),('lnprior_TP',float),('lnlike',float),
                 ('chi2',float),('redchi2',float),('outgoingFlux',float),('MuAve',float),('grav',float),
                 ('scaleHeight',float)]
        
        if not self.DoNotBlobArrays:
            for key in ['T','qmol_lay','secEclppm','dppm','thermal']:
                dt.append((key,object))
        
        self.blobKeys = []
        for i in range(len(dt)):
            self.blobKeys.append(dt[i][0])
        
        #Run sampler
        if pos0==None:
            pos0=ut.createWalkers(param0,delta,self.nwalkers)
            
        self.printIterations=printIterations
        
        self.istep = 0
        
        if self.nsubsteps == None:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, blobs_dtype = dt, args=())
    
            print("Running MCMC...")
            for i,result in enumerate(self.sampler.sample(pos0, iterations=self.nsteps, store=True)):    
                self.istep = i+1
    
                if np.any(self.istep==self.saveSteps) or ((self.istep%self.saveSteps[-1]) == 0):
                    if self.plotting:
                        self.chainplot(self.istep)
                        
                    self.transferBlobsFull()                
    
                    if self.plotting:
                        #Plot best fit
                            #calcSettings=deepcopy(self.modelSetting['calcSettings'])
                            #self.modelSetting['calcSettings'][3]=1
                        self.showParaEstimates()
                        self.runModelwithBestFit()
                        self.makeAtmPlotsForLastRun(addToFileBase='TempBestFit')
                        figures,ax = self.plotLastLikelihoodEvaluation(extraTitle='Best Fit:')
                        for i,fig in enumerate(figures):
                            fig.savefig(self.filebase + 'BestFit_Spec' + str(i) + '.pdf')
                            #self.modelSetting['calcSettings']=deepcopy(calcSettings)
    
                    lastsave=self.save_hdf5_v04()
                    print("filename='"+lastsave.name+"'")
    
                    if self.plotting:
                        #Plot Chain Progress and Samples
                        figFiles=[]
                        retrievalPlotTools.plotChainAndCorner(self,figFiles=figFiles)
                        if 'Tspline' in self.modelSetting.keys():
                            retrievalPlotTools.plotSpecSamples(self,randomInds=None,figFiles=figFiles,whichPlots=[1,1],plotTtau=(self.modelSetting['TempType'][:-2]=='TtauFreeProf'),Tspline = self.modelSetting['Tspline'])
                        else:
                            retrievalPlotTools.plotSpecSamples(self,randomInds=None,figFiles=figFiles,whichPlots=[1,1],plotTtau=(self.modelSetting['TempType'][:-2]=='TtauFreeProf'))
                        retrievalPlotTools.likelihoodPanel1D(self,figFiles=figFiles)
                        retrievalPlotTools.bayesianProb_vs_ProfLike(self,figFiles=figFiles)
                        
                        hp = hpy()
                        
                        with open(self.filebase + 'v04_sizeOfHeap.txt', 'a') as f:
    
                            print('Before removing figures, at step ',self.istep,'\n : ',hp.heap(), file=f)
            
                            list_all_figures=[manager.canvas.figure for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
                            for fg in list_all_figures: 
                                fg.clf(); plt.close(fg)
                            del list_all_figures
                            
                            print('\nMemory at step',self.istep,' :\n ',hp.heap(), '\n\n', file=f)
                        
                        #self.memoryUsage(file_end='sizeOfVarInEmceefit')
                        #pdb.set_trace()
    
            print("MCMC Done.")
    
            self.nsteps=self.sampler.iterations
            self.lastWalkers = self.sampler.chain[:,-1,:]
            self.transferBlobsFull()
        
        else:
            print("Running MCMC...")
            for i in range(int(self.nsteps/self.nsubsteps)):
                self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, blobs_dtype = dt, args=())
                
                for j,result in enumerate(self.sampler.sample(pos0, iterations=self.nsubsteps, store = True)):
                    self.istep += 1
                 
                if self.plotting:
                    self.chainplot(self.istep)
                      
                self.transferBlobs()                
            
                if self.plotting:
                    #Plot best fit
                        #calcSettings=deepcopy(self.modelSetting['calcSettings'])
                        #self.modelSetting['calcSettings'][3]=1
                    self.showParaEstimates()
                    self.runModelwithBestFit()
                    self.makeAtmPlotsForLastRun(addToFileBase='TempBestFit')
                    figures,ax = self.plotLastLikelihoodEvaluation(extraTitle='Best Fit:')
                    for i,fig in enumerate(figures):
                        fig.savefig(self.filebase + 'BestFit_Spec' + str(i) + '.pdf')
                        #self.modelSetting['calcSettings']=deepcopy(calcSettings)
                            
                lastsave=self.save_hdf5_v04()
                print("filename='"+lastsave.name+"'")
                
                del lastsave #so that it does not use memory
        
                if self.plotting:
                    #Plot Chain Progress and Samples
                    figFiles=[]
                    retrievalPlotTools.plotChainAndCorner(self,figFiles=figFiles)
                    if 'Tspline' in self.modelSetting.keys():
                        retrievalPlotTools.plotSpecSamples(self,randomInds=None,figFiles=figFiles,whichPlots=[1,1],plotTtau=(self.modelSetting['TempType'][:-2]=='TtauFreeProf'),Tspline = self.modelSetting['Tspline'])
                    else:
                        retrievalPlotTools.plotSpecSamples(self,randomInds=None,figFiles=figFiles,whichPlots=[1,1],plotTtau=(self.modelSetting['TempType'][:-2]=='TtauFreeProf'))
                    retrievalPlotTools.likelihoodPanel1D(self,figFiles=figFiles)
                    retrievalPlotTools.bayesianProb_vs_ProfLike(self,figFiles=figFiles)
                    
                    hp = hpy()
                    
                    with open(self.filebase + 'v04_sizeOfHeap.txt', 'a') as f:

                        print('Before removing figures, at step ',self.istep,'\n : ',hp.heap(), file=f)
        
                        list_all_figures=[manager.canvas.figure for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
                        for fg in list_all_figures: 
                            fg.clf(); plt.close(fg)
                        del list_all_figures
                        
                        print('\nMemory at step',self.istep,' :\n ',hp.heap(), '\n\n', file=f)
                    
                   # self.memoryUsage(file_end='sizeOfVarInEmceefit')
                    #pdb.set_trace()
                
                pos0 = self.sampler.get_last_sample().coords
                
            print("MCMC Done.")
    
            self.nsteps=self.istep
            self.lastWalkers = self.sampler.chain[:,-1,:]
            self.transferBlobs()


    def nestlefit(self,npoints=200):
        
        self.nestleNPoints=npoints
        self.nestleName='zz_nestle{0:05d}'.format(npoints)
        self.filebase=self.filebase+self.nestleName+'_'
        
        print('Setting number of numexpr threads back to 1')
        ne.utils.set_vml_num_threads(1)  
     
        def lnlike(theta):
            print(theta)
            lnlikelihood,blob = self.lnlike(theta)
            return lnlikelihood
        
        def prior_transform(theta):
            transformed = np.zeros(len(self.para))
            for i,para in enumerate(self.para): 
                transformed[i] = para.low + (para.high-para.low) * theta[i]
            return transformed
    
        print('\n\n')
        
        self.nestleresults = nestle.sample(lnlike, prior_transform, len(self.para), method='multi',
                                           npoints=npoints,callback=nestle.print_progress)
        
        print('\n\nCalculation summary:')
        print(self.nestleresults.summary())
        
        #ut.savepickle(self.nestleresults, self.filebase+'results.pkl')
        
        self.nestlesave()
        
    
    def nestlesave(self):

        #Re-show fitting parameters
        self.dispFittingPara()

        #Mean and covariance from samples
        p, cov = nestle.mean_and_cov(self.nestleresults.samples, self.nestleresults.weights)

        #Best fit paramters:
        self.MaxLnLikeIndex = np.argmax(self.nestleresults['logl'])
        self.MaxLnLike      = self.nestleresults['logl'][self.MaxLnLikeIndex]
        self.paraMaxLnLike  = self.nestleresults['samples'][self.MaxLnLikeIndex,:]
        
        self.nnestlesamples = self.nestleresults['samples'].shape[0]

        #Print results on screen:
        print('\nResults:') 
        for i,para in enumerate(self.para): 
            print("{0:25s} : {1:20s} = {2:5.8f} +/- {3:5.8f}".format(para.label,para.symbol, p[i], np.sqrt(cov[i, i]))) 
        
        #--Saving output to human-readable file----------------------------------------
        with open(self.filebase + 'results.txt', 'w') as f:
            print('Calculation summary:',file=f)
            print(self.nestleresults.summary(), file=f)
            print('\n\n')
            print('   %18s | %15s |  %12s   |  %12s  |  %12s  |  %12s' %('label','symbol','guess','low','high','srange'),file=f)
            for i,para in enumerate(self.para):
                print('%2d %18s | %15s |  %12g   |  %12g  |  %12g  | %12g' %(i,para.label,para.symbol,para.guess,para.low,para.high,para.srange),file=f)

            print('\nPosterior parameter estiamtes:', file=f) 
            for i,para in enumerate(self.para): 
                print("{0:25s} : {1:20s} = {2:5.8f} +/- {3:5.8f}".format(para.label,para.symbol, p[i], np.sqrt(cov[i, i])), file=f) 
        
            print('\nBest fit:', file=f)
            print('MaxLnLikeIndex: ', self.MaxLnLikeIndex, file=f) 
            print('MaxLnLike:      ', self.MaxLnLike, file=f) 
            print('paraMaxLnLike:  ', self.paraMaxLnLike, file=f) 


        #--Saving results to machine-readable h5 file----------------------------------------
        hf = h5py.File(self.filebase + 'results.h5', 'w')
        hf.create_dataset('logz',   data=self.nestleresults.logz)
        hf.create_dataset('logzerr',data=self.nestleresults.logzerr)
        hf.create_dataset('mean',   data=p)
        hf.create_dataset('cov',    data=cov)
        hf.close()
 
        #--Saving results and chain to machine-readable h5 file-------------------------------
        hf = h5py.File(self.filebase + 'results_with_samples.h5', 'w')
        for key in self.nestleresults.keys():
            hf.create_dataset(key,    data=self.nestleresults[key])
        hf.create_dataset('mean', data=p)
        hf.create_dataset('cov',  data=cov)
        hf.close()
        
        #--Save atmosphere object---------------------------------------------------------
        self.atm.save(filename=self.filebase+'RetrievalObj_atm.pkl')

        #--Run and save best fit model----------------------------------------------------
        self.MaxLnLike2, self.MaxLnLikeBlob = self.lnprob(self.paraMaxLnLike,bestfitrun=True)
        figs,axs=self.plotLastLikelihoodEvaluation(plotOrigScatter=False,plotRawData=True,plotLCsInOneFigure=False,showDataIndex=False,figsize=[9,10])        
        for i,fig in enumerate(figs):
            fig.savefig(self.filebase + 'MaxLikeFit_spec'+str(i)+'.pdf')
        
        #--Create random samples---------------------------------------------------------
        self.samples=self.createSamplesTable(retrievalMethod='nestle',nsamples=np.min([int(self.nnestlesamples/10),300]),save=True)

        #--Create bestfitModel in the same form as random samples-------------------------
        self.bestfitModel=self.createSamplesTable(retrievalMethod='nestle',bestfitOnly=True)
        
        #--Saves core object to pickle without atm, blobs, etc.---------------------------
        obj=dict()        
        start=time.time()
        for key in self.__dict__.keys():
            if key!='sampler' and key!='atm' and key!='blobs' and key!='samp' and key!='panda' and key!='samples' and key!='nestleresults':
                obj[key]=self.__dict__[key]
        ut.savepickle(obj, self.filebase+'RetrievalObj.pkl')
        print('core object pickle save time = ',time.time()-start,'sec')


        #--Plot corner---------------------------------------------------------
        fig = corner.corner(self.nestleresults.samples, weights=self.nestleresults.weights, labels=[para.symbol for para in self.para],bins=30)
                            #range=[[p[i]-5*np.sqrt(cov[i,i]),p[i]+5*np.sqrt(cov[i,i])] for i in range(4)],
        fig.savefig(self.filebase + 'corner.pdf')
        

        #--Plots spectra fit----------------------------------------------------
        figSpec,axSpec,figTp,axTp=retrievalPlotTools.plotSpecTpSamplesInOne(self,randomInds=None,
                                                  jointPlot=True,style=[0,0,0],bestFit=[1,1,0],presAxis=True,
                                                  xlim=[None,None,None],xscale=['log','log'],
                                                  ylim=[None,None,None],figsize=[9.5, 6.6],save=False)  
        figSpec[0].savefig(self.filebase + 'SpectraFit_range.pdf')
    
        figSpec,axSpec,figTp,axTp=retrievalPlotTools.plotSpecTpSamplesInOne(self,randomInds=None,
                                                  jointPlot=True,style=[1,1,1],bestFit=[1,1,0],presAxis=True,
                                                  xlim=[None,None,None],xscale=['log','log'],
                                                  ylim=[None,None,None],figsize=[9.5, 6.6],save=False)  
        figSpec[0].savefig(self.filebase + 'SpectraFit_sample.pdf')   

        fig=retrievalPlotTools.PlotTPAndContribution(self.atm,self.samples['T'],self.specs,spectype='secEclppm')
        fig.savefig(self.filebase + 'TP_contribution_range.pdf') 



    def createSamplesTable(self,retrievalMethod='nestle',nsamples=300,save=False,bestfitOnly=False,
                           keysFromAtm=['T','qmol_lay','dppm','thermal','secEclppm']):
    
        if retrievalMethod=='nestle':
            samplesOfParam =self.nestleresults['samples']
            weights        =self.nestleresults['weights']
        elif retrievalMethod=='emcee':
            samplesOfParam =self.panda[self.listOfPara()]
            nsamp=samplesOfParam.shape[0]
            weights=1.0/nsamp * np.ones(nsamp)
        
        #Choose which indices to rerun
        if bestfitOnly is False:
            #Determine indices of random sample    
            cumsum = np.cumsum(weights)    
            samples=table.Table([np.arange(nsamples)],names=['isamp'])
            samples['uniformDraw']=np.random.uniform(size=nsamples)
            samples['index']=[np.argmax(cumsum>x) for x in samples['uniformDraw']]
        else:
            #Only use index of bestfit
            nsamples=1
            samples=table.Table(np.array([0]),names=['isamp'])
            samples['index']=self.MaxLnLikeIndex

        #Take the parameters for the indices in samples['index']
        samples['param']=samplesOfParam[samples['index']]
        for ipara,para in enumerate(self.para):
            samples[para.symbol] = samples['param'][:,ipara]
            
        #Create one example blob and allocate the table accordingly
        lnp, blob = self.lnprob(samples[0]['param'])
        blobKeys=blob.keys()
        for key in blobKeys:
            if (type(blob[key]) is int) or (type(blob[key]) is float) :
                samples[key]=np.zeros(nsamples)  
            elif len(blob[key].shape)==0:
                samples[key]=np.zeros(nsamples)  
            else:            
                samples[key]=np.zeros(np.r_[nsamples,blob[key].shape])  
        
        for key in keysFromAtm:
            samples['atm_'+key]=np.zeros(np.r_[nsamples,self.atm.__dict__[key].shape])
        
        #Run all lines of the table through the atmosphere model and fill in the values
        #fig,ax = self.specs[0].plot(zorder=1000,color='black')
        for isamp,samp in enumerate(samples):
            
            print('Create random sample {0:d} of {1:d}'.format(isamp,nsamples))
            
            lnp, blob = self.lnprob(samp['param'])
            #self.atm.plotSpectrum(ax=ax,resPower=200,color='C0')
                    
            for key in blobKeys:
                samples[key][isamp]=blob[key]
    
            for key in keysFromAtm:
                samples['atm_'+key][isamp]=self.atm.__dict__[key]
            
        if save:
            samples.write(self.filebase+'RandomSamples.h5','hdf5',overwrite=True)
            
        if bestfitOnly:
            return samples[0]
        else:
            return samples




    #%% MODELING TO GET PROBABILITY

        
    def lnprob(self,param,plotting=False,bestfitrun=False):
        lp, lnprior_TP, lnprior_gamma = self.lnprior(param)
        if np.isfinite(lp):
            #Inside parameter space
            lnlikelihood,blob = self.lnlike(param,plotting,bestfitrun)
            lnp = lp + lnlikelihood
        else:
            #Outside parameter space
            lnp=-np.inf
            blob = dict()
            blob['lnlike']=float('nan')
            blob['chi2']=float('nan')
            blob['redchi2']=float('nan')

            blob['outgoingFlux']=float('nan')
            blob['MuAve']       =float('nan')
            blob['grav']        =float('nan')
            blob['scaleHeight' ]=float('nan')
            blob['T']=np.full(self.atm.nLay,np.nan)
            blob['qmol_lay']=np.full([len(self.blobLayers),self.atm.nMol],np.nan)
            nwaveBlob = len(convolve(self.modelDppm,self.blobKernel)[::int(self.blobSmoothing)][self.edge:-self.edge])
            blob['dppm']     =np.full(nwaveBlob,np.nan)
            blob['secEclppm']=np.full(nwaveBlob,np.nan)
            blob['thermal']  =np.full(nwaveBlob,np.nan)

        blob['lnprior']=lp
        blob['lnprior_TP']=lnprior_TP
        blob['lnprior_gamma']=lnprior_gamma
        if self.DoNotBlobArrays:
            return lnp, blob['lnprior'],blob['lnprior_gamma'],blob['lnprior_TP'],blob['lnlike'],blob['chi2'],blob['redchi2'],blob['outgoingFlux'],blob['MuAve'],blob['grav'],blob['scaleHeight']
        else:
            return lnp, blob['lnprior'],blob['lnprior_gamma'],blob['lnprior_TP'],blob['lnlike'],blob['chi2'],blob['redchi2'],blob['outgoingFlux'],blob['MuAve'],blob['grav'],blob['scaleHeight'],blob['T'],blob['qmol_lay'],blob['secEclppm'],blob['dppm'],blob['thermal']



    def lnprior(self,param):

        #Uniform priors
        lp=0
        for para in self.para:
            lp=lp+np.log(1/(para.high-para.low))    

        #Smoothness prior for temperature profile         
        if self.modelSetting['TempType'][-10:-2]=='FreeProf':
            nTFreeLayers    = int(self.modelSetting['TempType'][-2:])
            TProfile        = np.zeros(nTFreeLayers)
            for i in range(len(param)):
                if self.para[i].label=='gamma':    gamma = param[i]
                if self.para[i].label=='loggamma': gamma = 10**param[i]
                elif self.para[i].label[0]=='T':
                    TProfile[int(self.para[i].label[-2:])] = param[i]
            if 'logGamma' in self.modelSetting.keys():
                gamma=10**self.modelSetting['logGamma']       
            
            #prior on gamma   
            lnprior_gamma = lnpriorGamma_alt2(gamma, self.modelSetting['gammaPriorSlope'])
                        #            lnprior_gamma = lnpriorGamma(gamma, self.modelSetting['gammaPriorMode'])
                        #            lnprior_gamma = lnpriorGamma_alt(gamma, self.modelSetting['gammaPriorSlope'])
            
            #prior on T
            TProfile    = np.array(TProfile)
            lnprior_TP  = lnpriorT(gamma, nTFreeLayers, TProfile)
        else:
            lnprior_TP = 0
            lnprior_gamma = 0

        #Sum of all the priors
        lp = lp + lnprior_TP + lnprior_gamma
            
        # Check whether outside the rectangular prior space
        for i in range(len(param)):
            if param[i]<self.para[i].low or param[i]>self.para[i].high:
                lp=-np.inf  #zero prior if outside prior parameter space)

        # Check whether outside the prior space defined in splitParamVector
        if np.isfinite(lp):     
            astroparams,systparams,scatter,insidePrior=self.splitParamVector(param)                                   
            if not insidePrior:
                lp=-np.inf  #zero prior if outside prior parameter space)
                            
        return lp, lnprior_TP, lnprior_gamma


    #%% Split Param Vector

    def splitParamVector(self,param):

        insidePrior=True
        
        #Copy default values in para
        para=deepcopy(self.defaultValueForPara)
        
        #Update values with fitting parameters (overwrite default values from defaultValueForPara)
        for i in range(self.paraBounds[0][0]):
            para[self.para[i].symbol] = param[i]

        #---Build astroparams as inputs to the atmosphere forward model (from scratch)-----------
        astroparams=dict()
        astroparams['Rstar']        =  para['Rstar']  
        astroparams['Teffstar']     =  para['Teffstar']  
        if 'Rp' in para:
            astroparams['Rp']=para['Rp']
        else:
            astroparams['Rp']=para['Rstar']*para['RpRs']   
        astroparams['Mp']            =  para['MpMearth']*Mearth
        astroparams['ap']            =  para['ap']  
        astroparams['Tint']          =  para['Tint']  
        astroparams['HeatDistFactor']=  para['HeatDistFactor']  
        astroparams['BondAlbedo']    =  para['BondAlbedo']  
        astroparams['GroundAlbedo']  =  para['GroundAlbedo']  
        
        #--ComposType---------        
        if self.modelSetting['ComposType']=='ChemEqui':
            astroparams['Metallicity'] = 10**para['logM']
            if ('CtoO' in para) and ('StretchCtoO' in para):
                raise ValueError('ERROR!!! Both CtoO and StretchCtoO were set. You can only set one of them.')
            elif 'CtoO' in para:
                astroparams['CtoO']=para['CtoO']
            elif 'StretchCtoO' in para:
                astroparams['CtoO']=ut.Stretch_to_CtoO(para['StretchCtoO'])
            astroparams['pQuench'] = para['pQuench']

        elif self.modelSetting['ComposType']=='WellMixed':
            #Copy default values for each molecule
            astroparams['qmol'] = para['qmol']

            #Update with fitting parameters
            for MolName in para['qmol'].keys():            
                if 'log'+MolName in para.keys():
                    astroparams['qmol'][MolName] = 10**para['log'+MolName]
                elif 'log('+MolName+'/H2O)' in para.keys():
                    astroparams['qmol'][MolName] = 10**para['log('+MolName+'/H2O)'] * 10**para['logH2O']   #log(CH4/H2O) --> CH4/H2O --> CH4
            
            #Update H2 and He mixing ratio to get a sum of 1
            astroparams['qmol']['H2']=0
            astroparams['qmol']['He']=0
            qH2He = 1 - np.sum([astroparams['qmol'][MolName] for MolName in astroparams['qmol'].keys()])
            astroparams['qmol']['H2']=qH2He/(1+0.157)                   #Jupiters He/H2 = 0.157
            astroparams['qmol']['He']=qH2He - astroparams['qmol']['H2'] #Jupiters He/H2 = 0.157

            #ut.dispdict(astroparams['qmol'])
            if qH2He<0.0001:
                insidePrior=False
                print('outside Prior!!! (qH2He<0.0001)')

        #--TempType---------
        if self.modelSetting['TempType']=='parameters':
            astroparams['Tprof'] = para['Tprof']
        elif self.modelSetting['TempType']=='FreeUniform':
            astroparams['Temp']  =  para['T00']
        elif self.modelSetting['TempType']=='TpTwoVis':
            astroparams['kappaIR']=10**para['logKappaIR']
            astroparams['gamma1'] =10**para['logGamma1']
            astroparams['gamma2'] =10**para['logGamma2']
            astroparams['alpha']  =para['alpha']
            astroparams['beta']   =para['beta']
        elif self.modelSetting['TempType']=='TintHeatDist':
            if self.modelSetting['ComposType']=='ChemEqui':
                astroparams['Tprof'] = self.interpTpLUT(self.TpLUT,Metallicity=astroparams['Metallicity'],Tint=astroparams['Tint'],HeatDistFactor=astroparams['HeatDistFactor'])
            elif self.modelSetting['ComposType']=='WellMixed':
                solarVal=dict()
                solarVal['H2O']=7.918E-04; solarVal['CH4']=4.305E-04; solarVal['CO']=5.450E-06; solarVal['CO2']=9.326E-09
                approxMetallicity = (10**para['logH2O'] + 10**para['logCH4'] + 10**para['logCO'] + 10**para['logCO2']) / (solarVal['H2O']+solarVal['CH4']+solarVal['CO']+solarVal['CO2'])
                astroparams['Tprof'] = self.interpTpLUT(self.TpLUT,Metallicity=approxMetallicity,Tint=astroparams['Tint'],HeatDistFactor=astroparams['HeatDistFactor'])
            #print(self.modelSetting['TempType'],astroparams['Tint'],astroparams['HeatDistFactor'],astroparams['Tprof'][1][0])
        
        elif self.modelSetting['TempType'][-10:-2]=='FreeProf':
            for i in range(int(self.modelSetting['TempType'][-2:])):
                if len(str(i))==1:  stri = '0'+str(i)
                else:               stri = str(i)
                astroparams['T'+stri]    =para['T'+stri]
            if 'gamma' in para.keys():
                astroparams['gamma']    = para['gamma']
            elif 'loggamma' in para.keys():
                astroparams['gamma']    = 10**para['loggamma']
        
        elif self.modelSetting['TempType']=='FreeTPcustomLay':
            for i in range(self.modelSetting['TPcustomLay'].size):
                if len(str(i))==1:  stri = '0'+str(i)
                else:               stri = str(i)
                astroparams['T'+stri]    =para['T'+stri]
            
        #--Geometric Albedo-------
        if 'SetGeometricAlbedo' in self.modelSetting['albedoOnly']:
            astroparams['GeometricAlbedo']=para['GeometricAlbedo']
            
        #--CloudTypes-------------
        astroparams['pCloud']       =  10**para['logpCloud']
        astroparams['cHaze']        =  10**para['logcHaze']
        if self.modelSetting['CloudTypes'][2]==1:
            astroparams['mieRpart']          = 10**para['logMieRpart']
            astroparams['miePAtTau1']        = 10**para['logMiePAtTau1']
            astroparams['mieRelScaleHeight'] = 10**para['logMieRelScaleHeight']
        if self.modelSetting['CloudTypes'][3]==1:
            astroparams['carmaFile']    =  para['carmaFile'] 

        if 'resPower' in para:
            astroparams['resPower']       =  para['resPower']

        for p in para:
            if 'offset' in p:
                astroparams[p] = para[p]
            
        #---Systematics Parameters for each light curve--------------------------------------
        systparams=[]
        for ispec in range(self.nspecs):
            sysModelPara=dict()
            for i in range(self.paraBounds[ispec][0],self.paraBounds[ispec][1]):
                sysModelPara[self.para[i].symbol[:-2]]=param[i]   #make dictionary from sysModel parameters for sysModel
            systparams.append(sysModelPara)

        #---Scatter--------------------------------------------------------------------------
        scatter=np.ones(self.nspecs)
        
        return astroparams,systparams,scatter,insidePrior
        

    def interpTpLUT(self,TpLUT,Metallicity,Tint,HeatDistFactor):
        temp=ut.interp3bb(np.log(TpLUT['Metallicities']),TpLUT['Tints'],TpLUT['HeatDistFactors'],TpLUT['temp'],np.log(Metallicity),Tint,HeatDistFactor)
        Tprof = np.vstack([TpLUT['pres'], temp])
        return Tprof
    


    #%%        
        
    def lnlike(self,param,plotting=False,bestfitrun=False):

        print('param = ',param)
        
        astroparams,systparams,scatter,insidePrior=self.splitParamVector(param)

        chi2,npoints,redchi2,lnlikelihood=np.zeros(self.nspecs),np.zeros(self.nspecs),np.zeros(self.nspecs),np.zeros(self.nspecs)

        #Run Model Atmosphere
        if insidePrior:
            
            #-- Apply offset to certain instruments if fitted
            for i,spec in enumerate(self.specs):
                if len(self.modelSetting['specOffsets']):
                    instrnames  = set(spec['instrname'])
                    for instr in list(instrnames):
                        if instr in self.modelSetting['specOffsets']:
                            ind = np.where(spec['instrname']==instr)
                            spec['yval'][ind] = self.specs0[i]['yval'][ind]+astroparams['offset_'+instr]

            astromodel = self.simulateInstrOutput(astroparams,plotting=plotting)
    
            systematicsmodel,model=[],[]
            for i,spec in enumerate(self.specs):
    
                #-Instrument Systematics Model (calls sysModel specified by user)
                if spec.meta['sysModel']['sysModel'] is None:
                    systematicsmodel.append(  np.ones(len(spec))  )
                else:
                    systematicsmodel.append(  spec.meta['sysModel']['sysModel'](spec,systparams[i])  )
    
                #-Total model
                model.append(astromodel[i]*systematicsmodel[i])
               
                #-Calculate lnlike------------------
                chi2[i] = calcChi2(spec['yval'],spec['yerrLow'],model[i])
                npoints[i] = len(spec)
                redchi2[i] = chi2[i] / npoints[i]
                lnlikelihood[i] = calclnlike(spec['yval'],spec['yerrLow'],model[i])
    
            totalchi2=np.sum(chi2)
            totalredchi2 = np.sum(chi2) / np.sum(npoints)
            totallnlikelihood=np.sum(lnlikelihood)
    
            print(str(self.istep)+'/'+str(self.nsteps),' {:f}  (totalchi2={:f})'.format(totalredchi2,totalchi2))
    
            #--Plotting---------------------------------
            if plotting or bestfitrun:
                #copy values to keep last results accessible
                self.lastRun=dict()
                self.lastRun['astroparams']=astroparams
                self.lastRun['systparams']=systparams
                self.lastRun['scatter']=scatter
                self.lastRun['astromodel']=astromodel
                self.lastRun['systematicsmodel']=systematicsmodel
                self.lastRun['model']=model
                residuals=[]
                for i,spec in enumerate(self.specs):
                    residuals.append(spec['yval']-model[i])
                self.lastRun['residuals']=residuals
    
            #--Saving to bestfitRun----------------------
            if bestfitrun:
                self.bestfitRun=self.lastRun
                #update corrected flux in light curves
                for i,spec in enumerate(self.specs):
                    spec.meta['bestfitRun']=dict()
                    spec.meta['bestfitRun']['astroparams']=astroparams
                    spec.meta['bestfitRun']['systparams']=systparams[i]
                    spec.meta['bestfitRun']['systematicsmodel']=systematicsmodel[i]
                    spec.meta['bestfitRun']['astromodel']=astromodel[i]
                    spec.meta['bestfitRun']['model']=model[i]
                    spec.meta['bestfitRun']['residuals']=residuals[i]
                    spec.meta['bestfitRun']['scatter']=scatter[i]
                    spec.meta['bestfitRun']['corrflux']=spec['yval']/systematicsmodel[i]
                    spec.meta['sysModel']['bestfit']=systparams[i]
    
            blob = dict()
            blob['lnlike']=totallnlikelihood
            blob['chi2']=totalchi2
            blob['redchi2']=totalredchi2
    
            blob['T']=self.T   #self.T[self.blobLayers]
            blob['qmol_lay']=self.qmol_lay[self.blobLayers,:][:,:]
            blob['dppm']     =convolve(self.modelDppm     ,self.blobKernel)[::int(self.blobSmoothing)][self.edge:-self.edge]
            blob['secEclppm']=convolve(self.modelSecEclppm,self.blobKernel)[::int(self.blobSmoothing)][self.edge:-self.edge]
            blob['thermal']  =convolve(self.modelThermal  ,self.blobKernel)[::int(self.blobSmoothing)][self.edge:-self.edge]

            blob['outgoingFlux']=self.atm.totalOutgoingFlux()
            blob['MuAve']       =self.atm.MuAve[self.blobLayers][self.indLayToPandas] / uAtom
            blob['grav']        =self.atm.grav[self.blobLayers][self.indLayToPandas]
            blob['scaleHeight' ]=self.atm.scaleHeight[self.blobLayers][self.indLayToPandas]
        
        else:
            totallnlikelihood=-np.inf
            
            blob = dict()
            blob['lnlike']=-np.inf
            blob['chi2']=-np.inf
            blob['redchi2']=-np.inf
    
            nwaveBlob = len(convolve(self.modelDppm,self.blobKernel)[::int(self.blobSmoothing)][self.edge:-self.edge])
            blob['T']=np.full(self.atm.nLay,np.nan)
            blob['qmol_lay']=np.full([len(self.blobLayers),self.atm.nMol],np.nan)
            blob['dppm']     =np.full(nwaveBlob,np.nan)
            blob['secEclppm']=np.full(nwaveBlob,np.nan)
            blob['thermal']  =np.full(nwaveBlob,np.nan)

            blob['outgoingFlux']=np.nan 
            blob['MuAve']       =np.nan
            blob['grav']        =np.nan
            blob['scaleHeight' ]=np.nan

        print('---------------------------------------------------------')
        
        return totallnlikelihood, blob
        
        
    def simulateInstrOutput(self,astroparams,plotting=False):
        '''calls forward model and simulates instrument output'''

        if plotting:        
            print(astroparams)

        #Run atmosphere model with new parameters (results are stored in self.atm)
        if self.fitRadiusInFwdModel==False:
            [self.modelWave,self.modelDppm,self.modelSecEclppm,self.modelThermal,self.T,self.qmol_lay,self.modelRpRs]=self.atm.runModel(self.modelSetting,astroparams,fitRadius=False,specsfitRadius=None,
                                                                                                                                        runName='',disp2terminal=False,returnVals=True,saveToFile=False,shiftDppm=False)
        else:                                                                                            
            [self.modelWave,self.modelDppm,self.modelSecEclppm,self.modelThermal,self.T,self.qmol_lay,self.modelRpRs]=self.atm.runModel(self.modelSetting,astroparams,fitRadius=True,specsfitRadius=[self.specs[0]],
                                                                                                                                        runName='',disp2terminal=False,returnVals=True,saveToFile=False,shiftDppm=True)
        
        #Compute instrument response from last model run
        astromodel=self.atm.instrResp(self.specs)

        return astromodel  #self.modelThermal
          


    def plotLastLikelihoodEvaluation(self,extraTitle='',plotRawData=True,plotResiduals=False,plotOrigScatter=False,plotRange=None,binSize=None,plotErrorbars=True,residualRange=None,residualTicks=None,plotLCsInOneFigure=False,showDataIndex=False,ylim=None,figsize=None,timeAxis='bjd',showScatterValue=True,resPower=1000):
        #astroparams=self.lastRun['astroparams']
        #astromodel=self.lastRun['astromodel']
        #systematicsmodel=self.lastRun['systematicsmodel']
        model=self.lastRun['model']
        #scatter=self.lastRun['scatter']
        

        #kernel = Gaussian1DKernel(10)
        #l = int(kernel.shape[0]/2)


        figs=[];axs=[];
        for i,spec in enumerate(self.specs):
            
            wave=spec['wave']/1000
            waveunit='um'            
            wavelabel='Wavelength [um]'
            
            #--Create figure and axes------------------------------------
            if i==0 or plotLCsInOneFigure==False:
                if plotResiduals==True:
                    fig = plt.figure(figsize=figsize)#figsize=(11, 8))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1])
                    ax=[]
                    ax.append(fig.add_subplot(gs[0]))
                    ax.append(fig.add_subplot(gs[1], sharex=ax[0]))
                elif plotResiduals==False:
                    fig, axtemp = plt.subplots(figsize=figsize)
                    ax=[axtemp]
                
                    
            #--Upper Panel: Plot Raw Data----------------------------------------------------------------
            hline=self.atm.plotSpectrum(ax=ax[0],spectype=spec.meta['spectype'],resPower=resPower,label='model',xscale=None)
            ax[0].plot(wave*1000,model[i],'o',ms=6,linewidth=0.5,markerfacecolor=hline.get_color(),markeredgecolor='black',zorder=100)
            spec.plot(ax=ax[0],color='black',lw=2,xunit=waveunit,showDataIndex=False,label='data',zorder=110)

            ax[0].set_xscale('log')
            ut.xspeclog(ax[0],level=1)

            if ylim is not None:
                ax[0].set_ylim(ylim)


            #ax[0].set_xlim([time[0],time[-1]])
            #ax[0].plot(self.modelWave,self.modelDppm,'-r',lw=0.5,label='model')
            #smoothedSpec = convolve(self.modelDppm, kernel)
            #ax[0].plot(self.modelWave[l:-l],smoothedSpec[l:-l],'-r',lw=0.5,label='model')
            


#            #--Upper Panel: Plot Corrected Data and Transit Model----------------------------------------------------------------
#            if binSize is None:
#                if plotErrorbars==False:
#                    ax[0].plot(time,transmodel[i],'bo',ms=2)
#                    ax[0].plot(time,lc.flux/systematicsmodel[i],'ro',ms=4,alpha=0.5)
#                else:
#                    ax[0].errorbar(time,lc.flux/systematicsmodel[i],yerr=lc.scatter,fmt='o', lw=1, ms=4,label='',alpha=1, color='k')
#            else:
#                if plotErrorbars==False:
#                    ax[0].plot(binning(time,binSize),binning(lc.flux/systematicsmodel[i],binSize),'o',ms=4,alpha=0.5)
#                else:
#                    ax[0].errorbar(binning(time,binSize),binning(lc.flux/systematicsmodel[i],binSize),yerr=binning(lc.scatter/np.sqrt(binSize),binSize),fmt='ro', lw=1,ms=4,label='',alpha=1, color='k')


            #--Lower Panel: Plot Residuals----------------------------------------------------------------
            if plotResiduals:
                plt.setp(ax[0].get_xticklabels(),visible=False)
#                ax[1].axhline(0,ls='-',color='gray',zorder=-1000)
#                if binSize is None:
#                    if plotOrigScatter==False:
#                        ax[1].errorbar(time,   (lc.flux-model[i])*1e6,  yerr=scatter[i]*np.ones(lc.bjd.shape)*1e6,fmt='o', lw=1, ms=4,label='',alpha=1, color='b')
#                    else:
#                        ax[1].errorbar(time,   (lc.flux-model[i])*1e6,  yerr=lc.scatter*1e6,fmt='o', lw=1, ms=4,label='',alpha=1, color='k')
#                else:
#                    if plotOrigScatter==False:
#                        ax[1].errorbar(binning(time,binSize),binning((lc.flux-model[i])*1e6,binSize),yerr=binning(scatter[i]*np.ones(lc.bjd.shape)*1e6/np.sqrt(binSize),binSize),fmt='o', lw=1, ms=4,label='',alpha=1, color='b')
#                    else:
#                        ax[1].errorbar(binning(time,binSize),binning((lc.flux-model[i])*1e6,binSize),yerr=binning(lc.scatter*1e6/np.sqrt(binSize),binSize),fmt='o', lw=1, ms=4,label='',alpha=1, color='k')
#                ax[1].set_ylabel('Residuals [ppm]')
#
#                ax[1].yaxis.set_major_locator(plt.MaxNLocator(6))
#                if residualRange is not None:
#                    ax[1].set_ylim(residualRange)
#                #plt.locator_params(nbins=4)
#                if residualTicks is not None:
#                    ax[1].set_yticks(residualTicks)
#
#                if showScatterValue:
#                    ax[1].text(0.01, 0.98,'%.1f ppm'%(scatter[i]*1e6), fontsize=10, ha='left', va='top', transform=ax[1].transAxes)
#
#                if showDataIndex:
#                    for j in range(len(lc.bjd)):
#                        ax[1].text(time[j],(lc.flux[j]-model[i][j]+lc.scatter[j])*1e6,str(lc.index[j]), fontsize=6, va='bottom', ha='center')
#                        #ax[1].text(time,(lc.flux-model[i]+scatter[i]*1.1)*1e6,str(lc.index[j]), fontsize=8)


            #------------------------------------------------------------------
#            for i,a in enumerate(ax):
#                ax[i].minorticks_on()

            #if plotRange is not None:
            #    ax[0].set_xlim(transitparams[i].t0+plotRange-jdref)


            ax[-1].set_xlabel(wavelabel)
            fig.suptitle(extraTitle + ' ' + spec.meta['name'])

            #fig.savefig(self.filebase+'LastLikelihoodEvaluation_LC' + str(i) + '.png')
#            plt.show()
            figs.append(fig)
            axs.append(ax)

        return figs,axs
                

    #---PLOTS FROM ATMOSPHERE MODEL------------------------------------------------------------------------

    def makeAtmPlotsForLastRun(self,addToFileBase=''):
        atm=self.atm
        specs=self.specs
        modelSetting=self.modelSetting
        origFilebase=deepcopy(atm.filebase)
        atm.filebase=atm.filebase+addToFileBase

        #--Plotting--------------------------------------------------------------------
        fig,ax=atm.plotComp(save=True)#; plt.close(fig)
        fig,ax=atm.plotTp(save=True) #; plt.close(fig)

        #--Plotting--------------------------------------------------------------------
        xscale='log';  resPower=500
        fig,ax=atm.plotSpectrum(save=True,spectype='dppm',specs=specs,resPower=resPower,xscale=xscale)
        if modelSetting['thermalOnly']!=[] and modelSetting['thermalOnly'][0]!='calcEmissionMolByMol':
            fig,ax=atm.plotSpectrum(save=True,spectype='thermal',specs=specs,resPower=resPower,xscale=xscale)
            fig,ax=atm.plotSpectrum(save=True,spectype='thermalSecEclppm',specs=specs,resPower=resPower,xscale=xscale)
            if 'MuObs' in modelSetting['thermalOnly']:
                fig,ax=atm.plotThermalMuObs(save=True,resPower=resPower,xscale=xscale)
                fig,ax=atm.plotSpectrum(save=True,spectype='thermalMuObs',specs=specs,resPower=resPower,xscale=xscale)
        if modelSetting['albedoOnly']!=[]:
            fig,ax=atm.plotSpectrum(save=True,spectype='albedo',specs=specs,resPower=resPower,xscale=xscale)
            fig,ax=atm.plotSpectrum(save=True,spectype='albedoSecEclppm',specs=specs,resPower=resPower,xscale=xscale)
        if 'calcEmissionMolByMol' in modelSetting['thermalOnly']:
            fig,ax=atm.plotSpectraByMol(save=True,spectype='dppm',resPower=resPower,xscale=xscale)
        if 'dppmMol' in modelSetting['transit']:
            fig,ax=atm.plotSpectraByMol(save=True,spectype='thermal',resPower=resPower,xscale=xscale,setylim=2)

#        xscale='linear';  resPower=500
#        #atm.save(saveSettings=np.array([1,0,0,1,0,0]),kernel=kernel)
#        if modelSetting['calcSettings'][0]:
#            fig,ax=atm.plotSpectrum(save=True,spectype='dppm',specs=specs,resPower=resPower,xscale=xscale)
#        if modelSetting['calcSettings'][1]:
#            fig,ax=atm.plotSpectrum(save=True,spectype='thermal',specs=specs,resPower=resPower,xscale=xscale)
#            fig,ax=atm.plotSpectrum(save=True,spectype='thermalSecEclppm',specs=specs,resPower=resPower,xscale=xscale)
#            if modelSetting['calcSettings'][1]==2:
#                fig,ax=atm.plotThermalMuObs(save=True,resPower=resPower,xscale=xscale)
#        if modelSetting['calcSettings'][2]:
#            #fig,ax=atm.plotSpectrum(save=True,spectype='albedo',specs=specs,kernel=kernel,xscale=xscale)
#            fig,ax=atm.plotSpectrum(save=True,spectype='albedoSecEclppm',specs=specs,resPower=resPower,xscale=xscale)
#            if modelSetting['calcSettings'][1]==2:
#                fig,ax=atm.plotSpectrum(save=True,spectype='thermalMuObs',specs=specs,resPower=resPower,xscale=xscale)
#        if modelSetting['calcSettings'][3]:
#            fig,ax=atm.plotSpectraByMol(save=True,spectype='dppm',specs=specs,resPower=resPower,xscale=xscale)
#        if modelSetting['calcSettings'][4]:
#            fig,ax=atm.plotSpectraByMol(save=True,spectype='thermal',resPower=resPower,xscale=xscale,setylim=0.25)
           
        plt.close('all')
        atm.filebase=origFilebase


    #---POST-PROCESSING------------------------------------------------------------------------
    
    def transferBlobsFull(self):

        #--Extract single value parameters from blobs------------------------------------------------
        if self.lastSaveToPanda==-1:
            nNewStepsInBlobs = self.istep
        else:
            nNewStepsInBlobs = self.istep-self.lastSaveToPanda #number of new steps in self.sampler.blobs
        
        lnprior      = np.zeros([self.nwalkers,nNewStepsInBlobs])
        lnprior_gamma= np.zeros([self.nwalkers,nNewStepsInBlobs])
        lnprior_TP   = np.zeros([self.nwalkers,nNewStepsInBlobs])
        lnlike       = np.zeros([self.nwalkers,nNewStepsInBlobs])
        chi2         = np.zeros([self.nwalkers,nNewStepsInBlobs])
        redchi2      = np.zeros([self.nwalkers,nNewStepsInBlobs])
        outgoingFlux = np.zeros([self.nwalkers,nNewStepsInBlobs])
        MuAve        = np.zeros([self.nwalkers,nNewStepsInBlobs])
        grav         = np.zeros([self.nwalkers,nNewStepsInBlobs])
        scaleHeight  = np.zeros([self.nwalkers,nNewStepsInBlobs])

        for i in range(nNewStepsInBlobs):
            lnprior[:,i] = [blob['lnprior'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            lnprior_gamma[:,i] = [blob['lnprior_gamma'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            lnprior_TP[:,i] = [blob['lnprior_TP'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            lnlike[:,i] = [blob['lnlike'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            chi2[:,i] = [blob['chi2'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            redchi2[:,i] = [blob['redchi2'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            outgoingFlux[:,i] = [blob['outgoingFlux'] for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            MuAve[:,i]        = [blob['MuAve']        for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            grav[:,i]         = [blob['grav']         for blob in self.sampler.blobs[i-nNewStepsInBlobs]]
            scaleHeight[:,i]  = [blob['scaleHeight']  for blob in self.sampler.blobs[i-nNewStepsInBlobs]]


        #--Build pandas data frame 
        self.newPanda = pd.DataFrame(self.sampler.lnprobability[:,np.max([self.lastSaveToPanda,0]):self.istep].reshape(-1), columns=['lnprobability'])

        #Add step and walker to data frame
        step=np.zeros([self.nwalkers,nNewStepsInBlobs]).astype(int)
        walker=np.zeros([self.nwalkers,nNewStepsInBlobs]).astype(int)
        for s in range(nNewStepsInBlobs):
            if self.lastSaveToPanda==-1: step[:,s]=s
            else: step[:,s]=s+self.lastSaveToPanda
        for w in range(self.nwalkers):
            walker[w,:]=w
        self.newPanda['step'] = step.reshape(-1)
        self.newPanda['walker'] = walker.reshape(-1)

        #Add goodness-of-fit values to panda
        self.newPanda['lnprior'] = lnprior.reshape(-1)
        self.newPanda['lnprior_TP'] = lnprior_TP.reshape(-1)
        self.newPanda['lnprior_gamma'] = lnprior_gamma.reshape(-1)
        self.newPanda['lnlike'] = lnlike.reshape(-1)
        self.newPanda['chi2'] = chi2.reshape(-1)
        self.newPanda['redchi2'] = redchi2.reshape(-1)

        #Add fitting parameters to pandas data frame
        samples = pd.DataFrame(self.sampler.chain[:,np.max([self.lastSaveToPanda,0]):self.istep,:].reshape((-1, self.ndim)), columns=[x.symbol for x in self.para])
        self.newPanda = pd.concat([self.newPanda,samples], axis=1)

        #Add additional single value parameters to panda
        self.newPanda['outgoingFlux'] = outgoingFlux.reshape(-1)
        self.newPanda['MuAve']        = MuAve.reshape(-1)
        self.newPanda['grav']         = grav.reshape(-1)
        self.newPanda['scaleHeight']  = scaleHeight.reshape(-1)
                
        #Best fit (everything is contained in self.bestfit now!!!)
        
        lastStepInBlobs = self.newPanda['step'][len(self.newPanda['step'])-1]
        firstStepInBlobs = lastStepInBlobs-len(self.sampler.blobs)+1
        
        if np.max(self.newPanda.lnprobability)>self.MaxLnProb:#if there is a new best fit
            self.paraBestFit = ut.samp2bestfit(self.newPanda)

            ind = self.newPanda.lnprobability.argmax()
            self.MaxLnProb = self.newPanda.lnprobability[ind]
            self.paraMaxLnProb = self.panda2para(self.newPanda.iloc[ind])
            self.pandaMaxLnProbFull = self.newPanda.iloc[ind]
            self.bestfit = self.newPanda.iloc[ind]
            self.bestProbBlob = self.sampler.blobs[int(self.bestfit['step']-firstStepInBlobs)][int(self.bestfit['walker'])]

        if np.max(self.newPanda.lnlike)>self.MaxLnLike:
            ind = self.newPanda.lnlike.argmax()
            self.MaxLnLike = self.newPanda.lnlike[ind]
            self.paraMaxLnLike = self.panda2para(self.newPanda.iloc[ind])
            self.pandaMaxLnProb = self.newPanda.iloc[ind]#where the ln(likelihood) is max, not ln prob ! (the name could be misleading)
            self.bestLikeBlob = self.sampler.blobs[int(self.pandaMaxLnProb['step']-firstStepInBlobs)][int(self.pandaMaxLnProb['walker'])]
        
        #Add derived values to panda
        
        #self.burnin = int(self.nsteps*burninFrac)    

        #Additional derived values
        if 'RpRs' in self.newPanda:
        	self.newPanda['Dppm'] = self.newPanda['RpRs']**2 * 1e6

        #Readability
        if 'StretchCtoO' in self.newPanda:
            self.newPanda['C/O'] = ut.Stretch_to_CtoO(self.newPanda['StretchCtoO'])
        if 'logpCloud' in self.newPanda:
            self.newPanda['pCloud [mbar]'] = 10**self.newPanda['logpCloud'] / 100


        self.indLayToPandas = np.searchsorted(self.atm.p[self.blobLayers],self.indLayToPandas,side='right')
        print("Pressure Level: " +  str(self.atm.p[self.blobLayers[self.indLayToPandas]] / 100) + ' mbar')

        #Add temperature to panda (at level defined by self.indLayToPandas)
        
        temp_arr = np.full([nNewStepsInBlobs,self.nwalkers,self.atm.nLay],np.nan)
        compos_arr = np.full([nNewStepsInBlobs,self.nwalkers,len(self.blobLayers),self.atm.nMol],np.nan)
        for i in range(nNewStepsInBlobs):
            for j in range(self.nwalkers):
               # if 'T' in self.sampler.blobs[i-nNewStepsInBlobs][j].keys(): 
                    try:
                        temp_arr[i,j]=self.sampler.blobs[i-nNewStepsInBlobs][j]['T']
                    except Exception: 
                        pass
                #if 'qmol_lay' in self.sampler.blobs[i-nNewStepsInBlobs][j].keys(): 
                    try:
                        compos_arr[i,j]=self.sampler.blobs[i-nNewStepsInBlobs][j]['qmol_lay']
                    except Exception:
                        pass

        self.newPanda['Temp'] = temp_arr[:,:,self.indLayToPandas].transpose().reshape(-1)

        #Add all the molecular abundances to panda
        for iMol,MolName in enumerate(self.atm.MolNames):
            self.newPanda[MolName] = np.log10(compos_arr[:,:,self.indLayToPandas,iMol].transpose().reshape(-1))

        #H2/He and heavy Mols fractions
        self.newPanda['H2He']   =10**self.newPanda['H2']+10**self.newPanda['He']
        self.newPanda['logH2He']=np.log10(self.newPanda['H2He'])
        self.newPanda['heavyMols']=1-self.newPanda['H2He']
        self.newPanda['logHeavyMols']=np.log10(self.newPanda['heavyMols'])
          
        #Remove unnecessary information in self.sampler.blobs
        for i in range(-len(self.sampler.blobs),-np.min([self.nSaveBlobs,len(self.sampler.blobs)])):self.sampler.blobs.remove(self.sampler.blobs[i]) 
        if self.lastSaveToPanda==-1:
            self.panda = self.newPanda
        else: 
            self.panda = pd.DataFrame(pd.concat([self.panda, self.newPanda])).sort_values(['walker','step'], ascending=True)
            
        self.lastSaveToPanda = self.istep


    def transferBlobs(self):
        blobs = self.sampler.get_blobs()

        #--Build pandas data frame 
        self.panda = pd.DataFrame(self.sampler.lnprobability.reshape(-1), columns=['lnprobability'])

        #Add step and walker to data frame        
        step = np.zeros([self.nwalkers,self.nsubsteps]).astype(int)
        walker = np.zeros([self.nwalkers,self.nsubsteps]).astype(int)
        
        step[:] = np.arange(self.istep-self.nsubsteps,self.istep)
        
        for w in range(self.nwalkers):
            walker[w,:] = w
        
        self.panda['step'] = step.reshape(-1)
        self.panda['walker'] = walker.reshape(-1)
    
        #Add goodness-of-fit values to panda
        self.panda['lnprior'] = np.transpose(blobs['lnprior']).reshape(-1)
        self.panda['lnprior_TP'] = np.transpose(blobs['lnprior_TP']).reshape(-1)
        self.panda['lnprior_gamma'] = np.transpose(blobs['lnprior_gamma']).reshape(-1)
        self.panda['lnlike'] = np.transpose(blobs['lnlike']).reshape(-1)
        self.panda['chi2'] = np.transpose(blobs['chi2']).reshape(-1)
        self.panda['redchi2'] = np.transpose(blobs['redchi2']).reshape(-1)
        
        #Add fitting parameters to pandas data frame
        samples = pd.DataFrame(self.sampler.chain.reshape((-1, self.ndim)), columns=[x.symbol for x in self.para])
        self.panda = pd.concat([self.panda,samples], axis=1)
        
        #Add additional single value parameters to panda
        self.panda['outgoingFlux'] = np.transpose(blobs['outgoingFlux']).reshape(-1)
        self.panda['MuAve']        = np.transpose(blobs['MuAve']).reshape(-1)
        self.panda['grav']         = np.transpose(blobs['grav']).reshape(-1)
        self.panda['scaleHeight']  = np.transpose(blobs['scaleHeight']).reshape(-1)
        
        #Best fit (everything is contained in self.bestfit now!!!)
        if np.max(self.panda.lnprobability)>self.MaxLnProb:#if there is a new max lnprob
            self.paraBestFit = ut.samp2bestfit(self.panda)
    
            ind = self.panda.lnprobability.argmax()
            self.MaxLnProb = self.panda.lnprobability[ind]
            self.paraMaxLnProb = self.panda2para(self.panda.iloc[ind])
            self.pandaMaxLnProbFull = self.panda.iloc[ind]
            self.bestfit = self.panda.iloc[ind]
            self.bestProbBlob = blobs[int(self.bestfit['step']-np.min(self.panda['step']))][int(self.bestfit['walker'])]
        
        if np.max(self.panda.lnlike)>self.MaxLnLike:
            ind = self.panda.lnlike.argmax()
            self.MaxLnLike = self.panda.lnlike[ind]
            self.paraMaxLnLike = self.panda2para(self.panda.iloc[ind])
            self.pandaMaxLnProb = self.panda.iloc[ind]#where the ln(likelihood) is max, not ln prob ! (the name could be misleading)
            self.bestLikeBlob = blobs[int(self.pandaMaxLnProb['step']-np.min(self.panda['step']))][int(self.pandaMaxLnProb['walker'])]
        
        #Add derived values to panda
        
        #self.burnin = int(self.nsteps*burninFrac)    

        #Additional derived values
        if 'RpRs' in self.panda:
            	self.panda['Dppm'] = self.panda['RpRs']**2 * 1e6

        #Readability
        if 'StretchCtoO' in self.panda:
            self.panda['C/O'] = ut.Stretch_to_CtoO(self.panda['StretchCtoO'])
        if 'logpCloud' in self.panda:
            self.panda['pCloud [mbar]'] = 10**self.panda['logpCloud'] / 100


        self.indLayToPandas = np.searchsorted(self.atm.p[self.blobLayers],self.indLayToPandas,side='right')
        print("Pressure Level: " +  str(self.atm.p[self.blobLayers[self.indLayToPandas]] / 100) + ' mbar')

        #Add temperature to panda (at level defined by self.indLayToPandas)
        if not self.DoNotBlobArrays:
            temp_arr = np.full([self.nsubsteps,self.nwalkers,self.atm.nLay],np.nan)
            compos_arr = np.full([self.nsubsteps,self.nwalkers,len(self.blobLayers),self.atm.nMol],np.nan)
            for i in range(self.nsubsteps):
                for j in range(self.nwalkers):
                    if 'T' in self.blobKeys: 
                        temp_arr[i,j]=blobs['T'][i][j]
                    if 'qmol_lay' in self.blobKeys: 
                        compos_arr[i,j]=blobs['qmol_lay'][i][j]
    
            self.panda['Temp'] = temp_arr[:,:,self.indLayToPandas].transpose().reshape(-1)
    
            #Add all the molecular abundances to panda
            for iMol,MolName in enumerate(self.atm.MolNames):
                self.panda[MolName] = np.log10(compos_arr[:,:,self.indLayToPandas,iMol].transpose().reshape(-1))
    
            #H2/He and heavy Mols fractions
            self.panda['H2He']   =10**self.panda['H2']+10**self.panda['He']
            self.panda['logH2He']=np.log10(self.panda['H2He'])
            self.panda['heavyMols']=1-self.panda['H2He']
            self.panda['logHeavyMols']=np.log10(self.panda['heavyMols'])
          
        self.lastSaveToPanda = self.istep   
    
        print('Memory usage of blobs and self.panda:')
        print('blobs:'+str(blobs.nbytes/float(1e9))+'Gb = '+str(blobs.nbytes/(self.istep*float(1e6)))+'Gb per 1000 steps')     
        print('self.panda:'+str(sys.getsizeof(self.panda)/float(1e9))+'Gb = '+str(sys.getsizeof(self.panda)/(self.istep*float(1e6)))+'Gb per 1000 steps')        
   
        
    def panda2para(self,pandaRow):
        return pandaRow[9:9+len(self.para)]


    def showParaEstimates(self):
        txt1 = ut.pd2latex(self.panda)+'\n'
        txt2 = ut.mcmcstats(self.panda)
        print(txt1)
        print(txt2)
        text_file = open(self.filebase+'BestFit.txt', 'w')
        text_file.write('defaultValueForPara:\n')
        text_file.write(str(self.defaultValueForPara)+'\n\n')
        text_file.write(txt1)
        text_file.write(txt2)
        text_file.close()
                            #        txt1 = ut.pd2latex(self.panda)+'\n'
                            #        txt2 = ut.mcmcstats(self.panda)
                            #        print(txt1)
                            #        print(txt2)
                            #        text_file = open(self.filebase+'BestFit.txt', 'w')
                            #        text_file.write(txt1)
                            #        text_file.write(txt2)
                            #        text_file.close()

        #Write machine readable best fit file in ecsv format
        self.paraEstimate = ut.samp2bestfit(self.panda)
        self.bestfitTab = ut.convertBestFitPandasToTable(self.paraEstimate)
        self.bestfitTab.meta['defaultValueForPara']=self.defaultValueForPara
        self.bestfitTab.write(self.filebase+'BestFit.csv',format='ascii.ecsv',delimiter=',',overwrite=True)


    def runModelwithBestFit(self):
        
        print('Max lnlikelihood = ', self.MaxLnLike)
        print(self.paraMaxLnLike)

        #Run with max probability scenario
        self.lnprob(self.paraMaxLnLike.values,bestfitrun=True)

        #Assign best fit values to each light curve object        
        for i,spec in enumerate(self.specs):
            spec.meta['bestfit'] = self.paraBestFit


    def runModelwithPandaRow(self,pandaRow):
        para=pandaRow[9:9+len(self.para)]
        self.lnprob(para,plotting=True)

    def runModelwithPara(self,paraset):
        self.lnprob(paraset,plotting=True)



    def chainplot(self,istep=None,save=True):
        if istep is None:
            fig,axes=ut.chainplot(self.sampler.chain,labels=self.parasymbols)
            if save:
                fig.savefig(self.filebase+"chainplot.png")
        else:
            if self.sampler is not None:
                fig,axes=ut.chainplot(self.sampler.chain[:,:istep,:],labels=self.parasymbols)
            else:
                fig,axes=ut.chainplot(self.sampler.chain[:,:istep,:],labels=self.parasymbols)
            if save:
                fig.savefig(self.filebase+"chainplot_temp.png")
            plt.close(fig)
        return fig
            

                    #    def triangleplot(self,burnin=None,plot_datapoints=True,bins=50,showTickLabel=False,dpi=200,save=True):
                    #        if burnin is None:
                    #            burnin = int(self.nsteps*3./5)    #only take the last 40%
                    #
                    #        samp = self.sampler.chain[:, burnin:, :].reshape((-1, self.ndim))
                    #        symbols = [x.symbol for x in self.para]
                    #        labelTexts = [x.label for x in self.para]
                    #
                    #        #Remove the "_1" in symbols (otherwise latex breaks because of double underscore)
                    #        n0 = self.paraBounds[0][0]
                    #        for i in range(n0,len(symbols)):
                    #            symbols[i] = symbols[i][:-2]
                    #        
                    #        fig1 = triangle.corner(samp, labels=symbols, labelTexts=labelTexts,plot_contours=True,quantiles=[0.16,0.5,0.84],plot_datapoints=plot_datapoints,bins=bins,showTickLabel=showTickLabel,fontsize=30)
                    #        if save:
                    #            fig1.savefig(self.filebase+"triangle_AllParameters.png",dpi=dpi)
                    #        return fig1

    #%%
    
    def save_hdf5_v04(self,filebase=None,blobKeys=None,burnin=None,nsamp=10000):

        self.saveFormat = 'v04'
        print("Saving! saveFormat = '"+self.saveFormat+"'")
        
        blobKeys = self.blobKeys
        
        #Defaults
        if filebase is None:
            filebase=self.filebase
        if burnin is None:
            self.burnin=int(0.6*self.istep)
        if blobKeys==None:
            blobKeys=['T','qmol_lay','dppm','secEclppm','thermal']

        self.wavesm=self.atm.wave[::int(self.blobSmoothing)][self.edge:-self.edge]

        #--Make self.bestFitModel------------------------------------------------------
        self.bestfitModel=dict()    
        self.bestfitModel['wave']    =self.wavesm
                    #        shape_spectrum= np.min([self.istep, self.nSaveBlobs])

        if 'dppm' in blobKeys:
            self.bestfitModel['dppm']        =self.bestProbBlob['dppm']
        if 'secEclppm' in blobKeys:
            self.bestfitModel['secEclppm']   =self.bestProbBlob['secEclppm']
        if 'thermal' in blobKeys:
            self.bestfitModel['thermal']     =self.bestProbBlob['thermal']

        #if 'T' in self.bestProbBlob.keys():
        try:
            self.bestfitModel['T']=self.bestProbBlob['T'][:]
        except Exception:
            self.bestfitModel['T']=np.full(self.atm.nLay, np.nan)

        #if 'qmol_lay' in self.bestProbBlob.keys():
        try:
            self.bestfitModel['qmol_lay']=self.bestProbBlob['qmol_lay'][:,:]
        except Exception:
            self.bestfitModel['qmol_lay'] = np.full([len(self.blobLayers),self.atm.nMol],np.nan)
            
        #--Save atmosphere object---------------------------------------------------------
        self.atm.save(filename=filebase+self.saveFormat+'_RetrievalObj_atm.pkl')


        #--Save random sample of model results from blobs---------------------------------
        print('Saving random samples of blobs:')        
        
        #check if some burnin steps are still in self.sampler.blobs
        if self.nsubsteps == None:
            if self.istep <= self.nSaveBlobs:
                nburnin_still_in_blobs = self.burnin
                sizeForSampleChoice = (self.istep-nburnin_still_in_blobs)*self.nwalkers
            elif (self.nSaveBlobs + self.burnin) >self.istep :
                nburnin_still_in_blobs = self.nSaveBlobs + self.burnin - self.istep
                sizeForSampleChoice = (self.nSaveBlobs-nburnin_still_in_blobs)*self.nwalkers
            else : 
                nburnin_still_in_blobs = 0
                sizeForSampleChoice = (self.nSaveBlobs)*self.nwalkers
            nsamp=np.min([nsamp,(self.istep-nburnin_still_in_blobs)*self.nwalkers,(self.nSaveBlobs-nburnin_still_in_blobs)*self.nwalkers]) ## review
        
        else:
            if self.nSaveBlobs>self.nsubsteps:
                if self.nsubsteps>10:
                    self.nSaveBlobs = int(0.3*self.nsubsteps)
                else:
                    self.nSaveBlobs = int(1)
            nsamp = self.nSaveBlobs * self.nwalkers
            sizeForSampleChoice = self.nsubsteps * self.nwalkers

        
        #--Make Astropy Table from random sample--------------------------------------------------------
        self.sampInd=np.sort(np.random.choice(np.arange(sizeForSampleChoice),size=nsamp,replace=False))
        sampSteps, sampWalkers = divmod(self.sampInd, self.nwalkers)
        
        if self.nsubsteps == None:
            sampSteps = sampSteps + nburnin_still_in_blobs
        
            if self.istep <= self.nSaveBlobs:
                stepAndWalker_s=list(sampSteps)
            else:
                stepAndWalker_s=list(sampSteps+(self.istep-self.nSaveBlobs))
                
        else:
            stepAndWalker_s=list(sampSteps+(self.istep-self.nsubsteps))
            
        stepAndWalker_w=list(sampWalkers)


                    #        #---Make astropy table from steps and walkers (SHOULD BECOME UNNECESSARY) ---------------------------
                    #        stepAndWalkerTable = table.Table([stepAndWalker_s,stepAndWalker_w], names=['step','walker'])
                    #        #Make astropy table with parameters
                    #        symbols = [x.symbol for x in self.para]
                    #        parameters = np.full((len(symbols),len(stepAndWalkerTable['step'])),np.nan)
                    #        for i in range(len(symbols)):
                    #            for j in range(len(stepAndWalkerTable['step'])):
                    #                parameters[i,j]=self.panda.iloc[np.where((self.panda['step']==stepAndWalkerTable['step'][j])*(self.panda['walker']==stepAndWalkerTable['walker'][j]))][symbols[i]]
                    #        parametersTable = table.Table(list(parameters), names=symbols)
                    #        #Samples Table
                    #        samples = table.hstack([stepAndWalkerTable,parametersTable])
                    #        samples.write(filebase+self.saveFormat+'_samples_old.hdf5',path='samples',overwrite=True)
                    #        #---------------------------------------------------------------------------------------------------

        #Samples Table directly from self.panda
        if self.nsubsteps == None:    
            ind = self.istep * np.array(stepAndWalker_w)+stepAndWalker_s
            
        else:
            ind = self.nsubsteps * np.array(stepAndWalker_w)+(np.array(stepAndWalker_s)-(self.istep-self.nsubsteps))
            
        samples = table.Table.from_pandas(self.panda.iloc[ind])
        samples.write(filebase+self.saveFormat+'_samples.hdf5',path='samples',overwrite=True)
        

        #Add blobs to table (from self.sampler.blobs)
        listOfBlobsInSample = [] 
        for i in range(sampSteps.size):
                current_blob = dict()
                #current_blob_keys = self.sampler.blobs[sampSteps[i]][sampWalkers[i]].keys()
                current_blob_keys = self.blobKeys
                nwaveBlob = len(convolve(self.modelDppm,self.blobKernel)[::int(self.blobSmoothing)][self.edge:-self.edge])
                if not self.DoNotBlobArrays:
                    for key in ['dppm','secEclppm','thermal']:
                        if key not in current_blob_keys : 
                            current_blob[key]=np.full(nwaveBlob,np.nan)
                    if 'T' not in current_blob_keys: 
                        current_blob['T']=np.full(self.atm.nLay,np.nan)
                    if 'qmol_lay' not in current_blob_keys: 
                        current_blob['qmol_lay']=np.full([len(self.blobLayers),self.atm.nMol],np.nan)
                for key in current_blob_keys:
                    current_blob[key]=self.sampler.blobs[sampSteps[i]][sampWalkers[i]][key]
                listOfBlobsInSample.append(current_blob)
        self.samples = table.hstack([samples,table.Table(listOfBlobsInSample)])
        
                                        #        #delete unnecessary information from self.sampler.blobs
                                        #        for colname in self.samples.colnames:
                                        #            if colname not in blobKeys: del self.samples[colname]

        for key in ['chi2_1','lnprior_1','lnprior_TP_1','lnprior_gamma_1','lnlike_1','chi_1','redchi2_1','outgoingFlux_1','MuAve_1','grav_1','scaleHeight_1']:
            if 'chi2_1' in self.samples.columns:
                self.samples.rename_column(key,key[:-2])
        
        #Save full astropy table to hdf5 file
        print('Saving self.samples to samplesInclBlobs.hdf5 :')
        print(self.samples.info)
        self.samples.write(filebase+self.saveFormat+'_samplesInclBlobs.hdf5',path='samples',overwrite=True)
                
                #        #--save info in self.samp to make the h5 files (SHOULD BECOME UNNECESSARY)----------------------------------------
                #        self.samp=dict()
                #        for key in blobKeys:
                #            self.samp[key]=samples[key]
                #            if key in ['dppm','secEclppm','thermal']:
                #                self.samp[key] = self.samp[key][:,self.edge:-self.edge]  #cut the ends
                
                        #        start=time.time()
                        #        for key in blobKeys:
                        #            with h5py.File(filebase+self.saveFormat+'_RetrievalObj_'+key+'.h5', 'w') as hf:
                        #                hf.create_dataset(key, data=self.samp[key])
                        #        print('samp hdf5 save time = ',time.time()-start,'sec')

        #--Save panda chain to file (created in transferBlobs)-----------------------------------------
        start=time.time()
        if self.nsubsteps==None:
            self.panda.to_hdf(filebase+self.saveFormat+'_RetrievalObj_panda.h5','table')  #,append=True)
        else:
            if self.istep == self.nsubsteps:
                self.panda.to_hdf(filebase+self.saveFormat+'_RetrievalObj_panda.h5','table',format='table')
            else:
                self.panda.to_hdf(filebase+self.saveFormat+'_RetrievalObj_panda.h5','table',append=True,format='table')
        print('Setting number of numexpr threads back to 1')
        ne.utils.set_vml_num_threads(1)   #ne.set_num_threads(1)
        print('panda hdf5 save time = ',time.time()-start,'sec')
        
        #--Saves core object to pickle without atm, blobs, etc.----------------------------------------
        obj=dict()        
        start=time.time()
        for key in self.__dict__.keys():
            if key!='sampler' and key!='atm' and key!='blobs' and key!='samp' and key!='panda' and key!='samples':
                obj[key]=self.__dict__[key]
        obj['nsteps']=self.istep
        obj['sampler']=dict()
        if self.nsubsteps == None:
            obj['sampler']['chain']         =self.sampler.chain[:,:self.istep,:]
            obj['sampler']['lnprobability'] =self.sampler.lnprobability[:,:self.istep]
        else:
            obj['sampler']['chain']         =self.sampler.chain
            obj['sampler']['lnprobability'] =self.sampler.lnprobability
            
        lastsave = ut.savepickle(obj, filebase+self.saveFormat+'_RetrievalObj.pkl')
        print('core object pickle save time = ',time.time()-start,'sec')
        
       # self.memoryUsage(file_end='sizeOfVarInSave')
        
        #save last position of the walkers in a txt file
        np.savetxt(filebase+self.saveFormat+'_LastWalkerPositions.txt',self.sampler.get_last_sample().coords)
        return lastsave
    
    
    def load_hdf5_v04(self,filename,inclBlobs=False):
        
        print('Loading object pickle data to self')
        obj=ut.loadpickle(filename)
        for key in obj.keys():
            self.__dict__[key] = obj[key]
        if 'sampler' in obj.keys():
            self.sampler = emcee.sampler.Sampler(2,2)
            self.sampler._chain  = obj['sampler']['chain']
            self.sampler._lnprob = obj['sampler']['lnprobability']

        print('Loading atmosphere pickle data to self.atm')
        atmFile=filename[:-4]+'_atm.pkl'
        self.atm=scarlet.loadAtm(atmFile)

        hdfFile=filename[:-4]+'_panda.h5'
        print('Loading pandas data frame to self.panda (size='+ut.file_size(hdfFile)+')')
        self.panda = pd.read_hdf(hdfFile,'table')
        
        #Read samples
        if inclBlobs:
            samplesFile=filename[:-16]+'samplesInclBlobs.hdf5'
        else:
            samplesFile=filename[:-16]+'samples.hdf5'
        print('Loading pandas data frame to self.samples (size='+ut.file_size(samplesFile)+')')
        self.samples = table.Table.read(samplesFile)

        #Correction because file saving was not correct: remove '_1' from some keys if needed:
        for key in ['chi2_1','lnprior_1','lnprior_TP_1','lnprior_gamma_1','lnlike_1','chi_1','redchi2_1','outgoingFlux_1','MuAve_1','grav_1','scaleHeight_1']:
            if key in self.samples.columns:
                self.samples.rename_column(key,key[:-2])
        
        if inclBlobs:
            if self.samples['dppm'].shape[1] > self.wavesm.shape[0]:
                #Correction because file saving was not correct:
                for key in ['dppm','secEclppm','thermal']:
                    self.samples[key]=self.samples[key][:,self.edge:-self.edge]

        self.filebase = os.path.dirname(filename) + '/' + self.filename + '_'
        print('filebase = '+self.filebase)    

        print('Setting number of numexpr threads back to 1')
        ne.utils.set_vml_num_threads(1)   #ne.set_num_threads(1)


    def load_nestle(self,filename,loadAtm=True,loadChain=True,loadSamples=True,loadRandomSamples=True):

        print('Loading object pickle data to self')
        obj=ut.loadpickle(filename)
        for key in obj.keys():
            self.__dict__[key] = obj[key]

                                #        if 'sampler' in obj.keys():
                                #            self.sampler = emcee.sampler.Sampler(2,2)
                                #            self.sampler._chain  = obj['sampler']['chain']
                                #            self.sampler._lnprob = obj['sampler']['lnprobability']

        self.filebase = os.path.dirname(filename) + '/' + self.filename + '_' + self.nestleName + '_'
        #self.filebase = os.path.dirname(filename) + '/' + self.filename + '_' 
        print('filebase = '+self.filebase)    


        #Load atm object
        if loadAtm:
            print('Loading atmosphere pickle data to self.atm')
            atmFile=filename[:-4]+'_atm.pkl'
            self.atm=scarlet.loadAtm(atmFile)
            self.wavesm=self.atm.wave[::int(self.blobSmoothing)][self.edge:-self.edge]
        
        #--Loading chain and results from machine-readable h5 file-------------------------------
        if loadSamples:
            self.nestleresults=dict()
            hf = h5py.File(self.filebase + 'results_with_samples.h5', 'r')
            for key in hf.keys():
                self.nestleresults[key]=hf[key][()]
            hf.close()
        else:
            self.nestleresults=dict()
            hf = h5py.File(self.filebase + 'results.h5', 'r')
            for key in hf.keys():
                self.nestleresults[key]=hf[key][()]
            hf.close()

        #--Loading samples that contain the blob information
        if loadRandomSamples:
            if os.path.isfile(self.filebase+'RandomSamples.h5'):
                self.samples=table.Table.read(self.filebase+'RandomSamples.h5')
            else:
                print('!!! Could not find RandomSamples.h5 file:   {0:s}'.format(self.filebase+'RandomSamples.h5'))


    def memoryUsage(self,file_end ='sizeOfVariables'):
        
        variable_sizes=[self.istep]
        list_in_scope = dir()
        for x in list_in_scope:
            if x.endswith('__'):
                list_in_scope.remove(x)
            else:
                variable_sizes.append(getsize(x))
        list_self_variables = self.__dict__.keys()
        for svar in list_self_variables:
            variable_sizes.append(getsize(getattr(self,svar)))
        list_self_sampler_variables = ['blobs','chain','lnprobability']
        variable_sizes.append(getsize(self.sampler.blobs))
        variable_sizes.append(getsize(self.sampler.chain))
        variable_sizes.append(getsize(self.sampler.lnprobability))
        list_var_names = ['self_istep_varSize']+list_in_scope+['self_'+ s for s in list_self_variables]+['self_sampler_'+ s for s in list_self_sampler_variables]
            
        table_var_size = table.Table([list_var_names,variable_sizes], names=['name','size'])
        table_var_size.sort('size')
        table_var_size.reverse()
        table_var_size['formatted'] = [ut.convert_bytes(x) for x in table_var_size['size']]
        
        table_var_size.write(self.filebase+self.saveFormat+'_'+file_end+'.txt',format='ascii.fixed_width',overwrite=True)
        #table_var_size.write(self.filebase+self.saveFormat+'_'+file_end+'it{0:06d}'.format(self.istep)+'.txt',format='ascii.fixed_width',overwrite=True)
        #table_var_size.write(self.filebase+self.saveFormat+'_'+file_end+'.hdf5',path='varsize',overwrite=True)


    def openpath(self):
        subprocess.call(('open', os.path.dirname(self.filebase)))
      

#%%
class fitparam(object):
    '''
    '''
    def __init__(self,symbol,guess,low,high,srange=0.01,label=None,samp=None):
        self.symbol=symbol  #formula name
        if label is None:
            self.label=self.symbol   #text label
        else:
            self.label=label
        self.guess=guess
        self.low=low
        self.high=high
        self.srange=srange
        self.samp=samp

    def calcMaxLike(self,lnprobability):
        #---Max Likelihood point-------------
        ind = np.unravel_index(lnprobability.argmax(), lnprobability.shape)
        self.xMaxProb = self.samp[ind]

    def paraStat(self):
        line = "{:30s}: {:16.6f} {:+14.6f} {:+14.6f}".format(self.symbol, self.xMean, -self.dx1Low, self.dx1Upp)
        return line

#%%

#calculate prior on T profile using the second derivative (independant of the number of knots)
def lnpriorT(gamma,N,TProfile):
    step15=np.linspace(np.log10(1e-8),np.log10(1e5),num=15,retstep=True)[1]
    stepN=np.linspace(np.log10(1e-8),np.log10(1e5),num=N,retstep=True)[1]
    
    return (-1.0/(2.0*gamma)) * ((step15/stepN)**4) * np.sum((TProfile[2:] - 2*TProfile[1:-1] + TProfile[:-2])**2) - 0.5 * np.log(2*np.pi*gamma)

#calculate prior on gamma for TpFreeProf or TtauFreeProf
def lnpriorGamma(gamma,mode): #gammaMode = mode of the gamma distribution for prior calculation
                
    #parameters of inverse gamma distribution used as prior for gamma in Line et al. (2015)
    alphaDist   = 1.0  
    betaDist    = 5.0e-5
    mode_initial_distri = 2.5e-5
    
    #re-scale gamma and calculate the prior
    gammaTilde  = gamma * (mode_initial_distri/mode)
    prior_gamma = np.exp(-betaDist/gammaTilde)*(betaDist/gammaTilde)**alphaDist /gammaTilde/math.gamma(alphaDist)
    
    return np.log(prior_gamma)

def lnpriorGamma_alt(gamma,slope):
    
    prior_log10gamma = np.exp(-slope* np.log10(gamma))
    return np.log(prior_log10gamma)
    

def lnpriorGamma_alt2(gamma, slope): #inverse gamma distribution with custom mode and slope
    
    if slope==3:#for slope of -3, mode at 10
        alpha   = 0.31
        beta    = 13.08
    elif slope==4:#for slope of -4, mode at 10
        alpha   = 0.74
        beta    = 17.44
    elif slope==5:#for slope of -5, mode at 10
        alpha   = 1.18
        beta    = 21.80
    elif slope==6:#for slope of -6, mode at 10
        alpha   = 1.62
        beta    = 26.16
    elif slope==8:#for slope of -8, mode at 10
        alpha   = 2.49
        beta    = 34.88
    elif slope==10:#for slope of -10, mode at 10
        alpha   = 3.36
        beta    = 43.60
    elif slope==20:#for slope of -20, mode at 10
        alpha   = 7.72
        beta    = 87.20
    elif slope==30:#for slope of -30, mode at 10
        alpha   = 12.08
        beta    = 130.80
    elif slope==35:#for slope of -35, mode at 10         
        alpha   = 14.26  
        beta    = 152.60
    elif slope==40:#for slope of -40, mode at 10
        alpha   = 16.44
        beta    = 174.40
    else :
        raise ValueError('ERROR: There is no custom alpha and beta for this slope !!')
        
    prior_gamma = (beta**alpha /math.gamma(alpha)) * (gamma**(-alpha-1)) * np.exp(-beta/gamma)
    return np.log(prior_gamma)