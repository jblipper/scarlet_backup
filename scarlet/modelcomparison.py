#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 01:16:09 2020

@author: bbenneke
"""


from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np
import numexpr as ne
ne.set_num_threads(1) #ne.utils.set_vml_num_threads(1)

import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import os
import astropy.io.fits as pf

import auxbenneke.utilities as ut
from auxbenneke.constants import pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom
from auxbenneke.quickcmds import clf
from auxbenneke import pyplanet, pyspectrum


from copy import deepcopy
import sys
#from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel

#from IPython.display import display, HTML
#import auxbenneke.plotprob as pr

#from bisect import bisect
#import pandas as pd
#pd.set_option('display.height', 100)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
#from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator

from astropy import table
#mpl.style.use('setborn')

#import time

import scarlet
#from scarlet import retrievalPlotTools
#import pyspectrum

import corner

#from PyPDF2 import PdfFileMerger
#import nestle

import h5py
import glob


#%%

def calcBayesFactors(basepath,nsamples=None):

    filenames= os.listdir(basepath) # get all files' and folders' names in the current directory
    
    result = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(basepath,filename)): # check whether the current object is a folder or not
            result.append(filename)
            
    result.sort()
    
    runs=table.Table([np.array(result[:-1])],names=['runName'])
    nruns=len(runs)
    
    refInd=[-1]
    
    
    #%%
    
    runs['label']=np.array('',dtype='S100')
    runs['npoints']=np.zeros([nruns],dtype=int)
    runs['logz']=np.zeros([nruns])
    runs['logzerr']=np.zeros([nruns])
    runs['MaxLnLike']=np.zeros([nruns])

    runs['MolNamesToFit']=np.array('',dtype='S100')
    runs['removed']=np.array('',dtype='S100')
    runs['nMols']=np.zeros([nruns],dtype=int)

    runs['nCloudTypes']=np.zeros([nruns],dtype=int)
    runs['CloudTypes']=np.zeros([nruns],dtype=int)
    runs['GrayClouds']=np.zeros([nruns],dtype=int)
    runs['Hazes']=np.zeros([nruns],dtype=int)

    
    for irun,run in enumerate(runs):
    
        print(run['runName'])

        if nsamples is None:
            #Find lastest result file    
            filenamePattern=os.path.join(basepath,run['runName'],run['runName']+'_zz_nestle*_RetrievalObj.pkl')
            resultFiles = glob.glob(filenamePattern)
            
            if resultFiles:
                resultFile=sorted(resultFiles)[-1]
            else:
                resultFile=None
        else:
            #Take the file for the specified nsamples
            resultFile=os.path.join(basepath,run['runName'],run['runName']+'_zz_nestle{0:05d}_RetrievalObj.pkl'.format(nsamples))
        
        print(resultFile)
        if resultFile is not None:
            #Load
            fit=scarlet.retrieval('empty',None)
            try:
                fit.load_nestle(resultFile,loadAtm=False,loadChain=False,loadSamples=True,loadRandomSamples=False)
            except:
                fit=None
                run['label']     = 'Opening failed.'

            if fit is not None:
                fit.label=fit.autoLabel()
                run['label']     =fit.label
                run['npoints']   =fit.nestleNPoints
                
                if 'MolNamesToFit' in fit.__dict__.keys():
                    run['MolNamesToFit']   =','.join(fit.MolNamesToFit)
                    run['nMols']           =len(fit.MolNamesToFit)
                
                run['nCloudTypes']     =np.sum(fit.modelSetting['CloudTypes'])
                run['CloudTypes']      =int(np.array2string(fit.modelSetting['CloudTypes'],separator='')[1:-1])
                run['GrayClouds']      =fit.modelSetting['CloudTypes'][0]
                run['Hazes']           =fit.modelSetting['CloudTypes'][1]
                
                run['logz']      =fit.nestleresults['logz']
                run['logzerr']   =fit.nestleresults['logzerr']
                run['MaxLnLike'] =np.max(fit.nestleresults['logl'])


        else:
            run['label']     = 'Not found.'
        


    
    #%%

    runs.sort(['nCloudTypes','nMols'])
    runs.reverse()
    refInd=0
    
    
    allMols=np.array(runs[refInd]['MolNamesToFit'].split(','))
    for run in runs:
        run['removed']=', '.join([mol for mol in np.setdiff1d(allMols,np.array(run['MolNamesToFit'].split(',')))])
        if runs['GrayClouds'][refInd] and not run['GrayClouds']:
            if run['removed']:
                run['removed']=', '.join([run['removed'],'no clouds'])
            else:
                run['removed']='clouds'
        if runs['Hazes'][refInd] and not run['Hazes']:
            if run['removed']:
                run['removed']=', '.join([run['removed'],'no hazes'])
            else:
                run['removed']='hazes'
    
    runs['dLogZ'] = runs['logz'][refInd] - runs['logz']
    
    runs['ln(Z)']=np.array('',dtype='S30')
    runs['Deltaln(Z)']=np.array('',dtype='S30')
    for run in runs:
        #run['ln(Z)']      = ut.meanUnc2latex('',run['logz'] ,run['logzerr']           ,forceplus=True)
        run['ln(Z)']      = ut.meanUnc2latex('',run['logz'] ,0.03                     ,forceplus=True)
        run['Deltaln(Z)'] = ut.meanUnc2latex('',run['dLogZ'],run['logzerr']*np.sqrt(2),forceplus=True)


    runs['BayesFac'] = np.exp(runs['dLogZ'])
    runs['sigma'] = ut.bayesFacToSigma(runs['BayesFac'])
    
    runs['BayesFacWithSigma']=np.array('',dtype='S30')
    for run in runs:
        if run['sigma']>2.2:
            run['BayesFacWithSigma'] = '{0:.2f} ({1:.2f}$\sigma$)'.format(run['BayesFac'],run['sigma'])    
        else:
            run['BayesFacWithSigma'] = '{0:.2f}'.format(run['BayesFac'])    
            
    
    #runs['BayesFac2'] = np.exp(runs['logz']) / np.exp(runs['logz'][refInd])
    
    runs['dMaxLnLike'] = runs['MaxLnLike'][refInd] - runs['MaxLnLike']
    runs['MaxLikeRatio'] = np.exp(runs['dMaxLnLike'])
    
    #runs.sort(keys=['BayesFac'])
    
    #%%


    
    #ut.printext(runs)
    runs.write(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTableFull.txt'),format='ascii.fixed_width',overwrite=True)
    #ut.printext(runs)



    runsTable = runs[['label','nMols','CloudTypes','removed','dLogZ','BayesFac','MaxLikeRatio']]
    #print(runsTable)
    runsTable.write(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable.txt'),format='ascii.fixed_width',overwrite=True)
    #runsTable.write(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable.tex'),format='ascii.aastex')
    #ut.openext(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable.txt'))

    
    fmt='7.2f'

    runs['dLogZ'].info.format = fmt
    runs['BayesFac'].info.format = fmt
    runs['sigma'].info.format = fmt
    runs['MaxLikeRatio'].info.format = fmt

    
    #%%
    
    runs['removed'][refInd]='full'
    runs['BayesFacWithSigma'][refInd]='--'
    
    t = runs
    t['Scenario']=ut.chemlatex(t['removed'])

    #t.write(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable.tex'),format='ascii.aastex')

    
    
    f=open(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable2.tex'), 'w')   
    print(r'\begin{table}[]',file=f)
    print(r'\caption{Bayesian Evidence for SCARLET Atmospheric Retrievals}',file=f)
    print(r'\begin{tabular}{ccc}',file=f)
    print(r'\hline',file=f)
    print(r'\hline',file=f)
    print(r'{0:20s} & {1:20s} & {2:20s} \\'.format('Scenario','ln(Z)','Bayes factor for'),file=f)    
    print(r'{0:20s} & {1:20s} & {2:20s} \\'.format('','','molecule present'),file=f)    
    print(r'\hline',file=f)
    for row in t:
        print(r'{0:20s} & {1:20s} & {2:20s} \\'.format('no $'+row['Scenario']+'$',row['ln(Z)'],row['BayesFacWithSigma']),file=f)    
    print(r'\hline',file=f)
    print(r'\end{tabular}',file=f)
    print(r'\end{table}',file=f)
    f.close()
    

    f=open(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable2.tex'), 'w')   
    print(r'{\renewcommand{\arraystretch}{1.4}',file=f)
    print(r'\begin{table*}[]',file=f)
    print(r'\caption{Bayesian Evidence for SCARLET Atmospheric Retrievals}',file=f)
    print(r'\centering',file=f)
    print(r'\begin{tabular}{cccl}',file=f)
    print(r'\hline',file=f)
    print(r'\hline',file=f)
    print(r'{0:20s} & {1:20s} & {2:20s} & {3:20s} \\'.format('Retrieval Model','Evidence'            ,'Max Like','Bayes Factor'),file=f)    
    print(r'{0:20s} & {1:20s} & {2:20s} & {3:20s} \\'.format(''               ,'$\ln(\mathcal{Z}_i$)','$\ln(\mathcal{L}_i$)','$B_i = \mathcal{Z}_\mathrm{full}/\mathcal{Z}_i$'),file=f)    
    print(r'\hline',file=f)
    for row in t:
        print(r'{0:20s} & {1:20s} & {2:20.2f} & {3:20s} \\'.format(
                'no $'+row['Scenario']+'$',
                row['ln(Z)'],
                row['MaxLnLike'],
                '$B_{'+row['Scenario']+'}$ = '+row['BayesFacWithSigma']),file=f)
    print(r'\hline',file=f)
    print(r'\end{tabular}',file=f)
    print(r'\end{table*}',file=f)
    print(r'}',file=f)
    f.close()
    




    f=open(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable3.tex'), 'w')   
    print(r'\begin{table}[]',file=f)
    print(r'\caption{Bayesian Evidence for SCARLET Atmospheric Retrievals}',file=f)
    print(r'\begin{tabular}{ccc}',file=f)
    print(r'\hline',file=f)
    print(r'\hline',file=f)
    print(r'{0:20s} & {1:20s} & {2:20s} \\'.format('Scenario','$\Delta\,ln(Z)$','Bayes factor for'),file=f)    
    print(r'{0:20s} & {1:20s} & {2:20s} \\'.format('','','molecule present'),file=f)    
    print(r'\hline',file=f)
    for row in t:
        print(r'{0:20s} & {1:20s} & {2:20s} \\'.format('no $'+row['Scenario']+'$',row['Deltaln(Z)'],row['BayesFacWithSigma']),file=f)    
    print(r'\hline',file=f)
    print(r'\end{tabular}',file=f)
    print(r'\end{table}',file=f)
    f.close()
        

    #ut.openext(os.path.join(basepath,os.path.basename(basepath)+'BayesFactorTable2.tex'))
    
    return runs
    



    

