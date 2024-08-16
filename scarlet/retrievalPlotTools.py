# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:42:29 2015
@author: bbenneke
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl

import numpy as np
import pdb
import os
import astropy.io.fits as pf

import auxbenneke.utilities as ut
from auxbenneke.constants import day, Rearth, Mearth, Rjup, Mjup, mbar, bar
from auxbenneke import pyplanet
from copy import deepcopy
import sys
from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel
from scipy.interpolate import make_interp_spline

import auxbenneke.plotprob as pr

from scipy import stats

import pandas as pd
#pd.set_option('display.height', 100)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)


import time

import scarlet
from scarlet import radutils
#import pyspectrum

import corner

from PyPDF2 import PdfFileMerger
from bisect import bisect


#%%


def loadRetrievalObj(path,saveFormat='v04',inclBlobs=True,keys=['T','qmol_lay','dppm','thermal','secEclppm','outgoingFlux']):

    print('-------------------------------------------------------')

    casename=os.path.basename(path)
    baseDir=os.path.dirname(path)
    
    fileEnd='_'+saveFormat+'_RetrievalObj.pkl'
    filename = os.path.join(baseDir,casename,casename+fileEnd)
    print(filename)

    start=time.time()
    print('Loading results...')

    if saveFormat=='v04':
        fit=scarlet.retrieval('empty',None)
        fit.load_hdf5_v04(filename,inclBlobs=inclBlobs)

    if saveFormat=='v0.3':
        fit=scarlet.retrieval('empty',None)
        fileEnd='_'+saveFormat+'_RetrievalObj.pkl'
        filename = os.path.join(baseDir,casename,casename+fileEnd)
        fit.load_hdf5_v03(filename,blobKeys=keys)
        fit.save_hdf5_v04()

    print('loadtime = ',time.time()-start,'sec')

    return fit

    

#%%
def makeCleanedChain(fit,figFiles=[],plotting=True):

    
    #Additional variables in fit.panda (will later also be in df)
    fit.panda['index']=np.arange(len(fit.panda))
    fit.nDataPoints=np.sum([len(spec) for spec in fit.specs])
    fit.panda['sigmaSign'] = (fit.panda['chi2'] - fit.nDataPoints) / np.sqrt(2*fit.nDataPoints)

    
    df = deepcopy(fit.panda)
    listOfPara=[para.symbol for para in fit.para]
    print('listOfPara: ', listOfPara)

    stepRange = None
    #--Plot chain before removing burn-in and bad walkers
    if fit.nsubsteps != None and fit.nsubsteps != 0:
        if len(fit.panda)==fit.nwalkers*fit.nsubsteps:
            stepRange=[fit.istep-fit.nsubsteps,fit.istep-1]
    
    if plotting:
        figFiles.append(pr.chainplot(df[['lnprior','lnprobability','chi2','redchi2', 'lnprior_gamma','lnprior_TP']+listOfPara],nwalkers=fit.nwalkers,fontsize=8,saveFileBase=fit.filebase,paraStr='AllFitPara',stepRange=stepRange))
    
    #--Remove burn-in
    if fit.nsubsteps == None or fit.nsubsteps == 0:
        burnin=int(0.6*fit.istep)
        df = df[df['step'] >= burnin]

    #--Remove bad walkers
    redchi2=df['chi2'].to_numpy()
    redchi2=redchi2.reshape([fit.nwalkers,int(len(df)/fit.nwalkers)])
    
    redchi2Filled = deepcopy(redchi2)
    redchi2Filled[np.isnan(redchi2Filled)]=1e30
    minRedChi2OfWalker=np.min(redchi2Filled,1)
    medianRedChi2OfWalker=np.median(redchi2Filled,1)
    
    if plotting:
        fig,ax=plt.subplots()
        ax.plot(minRedChi2OfWalker,label='minimum')
        ax.plot(medianRedChi2OfWalker,label='median')
        ax.set_xlabel('index of walker')
        ax.set_ylabel('redchi2')
        ax.legend()
    
    y=medianRedChi2OfWalker
    med = np.median(y); std = np.std(y); sigma=3
    removeWalkers = np.where(y > med+sigma*std)[0]
    print('removeWalkers: ', removeWalkers)

    if plotting:
        figFiles[-1][0].suptitle('RemWalkers: '+str(removeWalkers))

    keptWalkers=np.arange(fit.nwalkers)
    for removeWalker in removeWalkers:
        df = df[df['walker']!=removeWalker]
        #fit.nwalkers=fit.nwalkers-1
        keptWalkers=np.delete(keptWalkers,np.where(keptWalkers==removeWalker-1))


    #--Chains after removing burn-in and bad walkers
    if plotting:
        try:
            figFiles.append(pr.chainplot(df[['lnprior','lnprobability','chi2','redchi2', 'lnprior_gamma','lnprior_TP']+listOfPara],nwalkers=len(keptWalkers),fontsize=8,saveFileBase=fit.filebase+'burnInRem',paraStr='AllFitPara',stepRange=stepRange))
        except:
            print('Could not create chainplot after burnInRem!')

    fit.listOfPara   =listOfPara
#    fit.keptWalkers  =keptWalkers
#    fit.removeWalkers=removeWalkers

    return df,listOfPara,keptWalkers,removeWalkers



def plotChainAndCorner(fit,figFiles=[]):
    '''
    Corner and Panel1d plots
    '''
    
    df,listOfPara,keptWalkers,removeWalkers = makeCleanedChain(fit,figFiles,plotting=True)

    print(fit.bestfit)
    figFiles.append(pr.panel1d(df[listOfPara],saveFileBase=fit.filebase,paraStr='FitParas'))
    figFiles.append(pr.panel1d(df[7:]        ,saveFileBase=fit.filebase,paraStr='AllPanda'))
    if fit.istep>10:
        figFiles.append(pr.triangle(df[listOfPara],bins=20,saveFileBase=fit.filebase,paraStr='AllFitPara',plot_datapoints=False))
 
    return figFiles
     



#%%

def calcDerivedQuantities(fit):
    '''Compute additional parameters in df'''

    df=fit.df
    
    if 'logpCloud' in df:
        df['logpCloudmbar']=df['logpCloud']-2

    if 'logMiePAtTau1' in df:
        df['logMiePAtTau1mbar']=df['logMiePAtTau1']-2


    #df=df[df['StretchCtoO'] < 8]
    #df['H2Osolar']=np.log10(10**df['H2O']/302e-6)
    #df['COsolar']=np.log10(10**df['CO']/414e-6)

    df['log [CH4]/[H2O]']=np.log10(10**df['CH4']/10**df['H2O'])
    df['log [CO]/[H2O]']=np.log10(10**df['CO']/10**df['H2O'])
    df['log [CO2]/[H2O]']=np.log10(10**df['CO2']/10**df['H2O'])


    if 'logpCloud' in df:
        df['logpCloudmbar']=df['logpCloud']-2
    if 'logMiePAtTau1' in df:
        df['logMiePAtTau1mbar']=df['logMiePAtTau1']-2

    if 'CtoO' not in df:
        df['CtoO']=(10**df['CO']+10**df['CO2']+10**df['CH4']) / (10**df['H2O']+10**df['CO']+2*10**df['CO2'])
    if 'logCtoO' not in df:
        df['logCtoO']=np.log10(df['CtoO'])
    if 'StretchCtoO' not in df:
        df['StretchCtoO']=ut.CtoO_to_Stretch(df['CtoO'].values)

def plotTP(p_layers, T_samples, ax=None, Tlabel='bottom', axethickness = 2, fontsize = 15, tickthickness = 2, tickfontsize = 13, direction = 'in', ticklength = 3):

    #Making figure
    if ax is None:
        fig,ax=plt.subplots()
    else:
        fig=None

#    subplot_width = axes[0,0].get_position().width
#    subplot_height = axes[0,0].get_position().height
#    subplot_spacing = axes[0,1].get_position().x0 - (axes[0,0].get_position().x0 + axes[0,0].get_position().width)
#    
#    TP_ax_0 = axes[int(len(plot_params)/2),int(len(plot_params)/2)].get_position().x0
#    TP_ay_0 = axes[int(len(plot_params)/2),int(len(plot_params)/2)].get_position().y0 + subplot_height + subplot_spacing
#        
#    ax = fig.add_axes( [TP_ax_0, TP_ay_0, 2*subplot_width-0.05, 2*subplot_height-0.02] )
    try:
        a=T_samples.shape[1]
        levels=[2.5,16,50,84,97.5]
        perc = np.nanpercentile(T_samples, levels, axis=0).T
    except:
        perc=T_samples
#    perc=np.zeros([len(p_layers),len(levels)])
#    for iLay in range(len(p_layers)):
#        perc[iLay,:] = np.nanpercentile(T_samples[:,iLay],levels)    
    
    #fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
    ax.set_xlabel('Temperature [K]', fontsize=fontsize)
    ax.set_ylabel('Pressure [bar]', fontsize=fontsize)
    ax.set_yscale('log')
#    ax.fill_betweenx(p_layers/1e5,perc[:,0],perc[:,4],color=[1,0.8,0.8],zorder=-10)
#    ax.fill_betweenx(p_layers/1e5,perc[:,0],perc[:,4],color=[1,0.5,0.5],zorder=-10)
#    ax.fill_betweenx(p_layers/1e5,perc[:,0],perc[:,4],color='firebrick',alpha=0.4,zorder=-10)
#    ax.fill_betweenx(p_layers/1e5,perc[:,0],perc[:,4],color='red',zorder=-10)
#    ax.fill_betweenx(p_layers/1e5,perc[:,1],perc[:,3],color=[1,0.75,0.75],zorder=-9)
#    ax.fill_betweenx(p_layers/1e5,perc[:,1],perc[:,3],color='firebrick',zorder=-9)
    try:
        ax.fill_betweenx(p_layers/1e5,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
        ax.fill_betweenx(p_layers/1e5,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
        ax.plot(perc[:,2],p_layers/1e5,color='blue',zorder=-8)

    except:
        ax.plot(perc,p_layers/1e5,color='blue',zorder=-8)

#    retrieval_obj.atm.Teq = retrieval_obj.atm.Teffstar*(retrieval_obj.atm.Rstar/retrieval_obj.atm.ap)**(0.5)*   (retrieval_obj.atm.HeatDistFactor*(1-retrieval_obj.atm.BondAlbedo))**(0.25)
    #plt.axvline(x=fit.atm.Teq,color='k',linestyle='--')
    #ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-7)
    ax.set_ylim([1e-9,1e3])
    ax.invert_yaxis()
    #ax.minorticks_on()
   # ax.yaxis.set_label_position("left")
   # if Tlabel == 'top':
    #    ax.xaxis.set_label_position("top")
     #   ax.tick_params(labelleft='off', labelright='on', labelbottom='off', labeltop='on', top = 'on', right = 'on', bottom='on', left='on', width=tickthickness, labelsize=tickfontsize, direction=direction, length=ticklength )
   # else:
    #    ax.tick_params(labelleft='off', labelright='on', labelbottom='on', labeltop='off', top = 'on', right = 'on', bottom='on', left='on', width=tickthickness, labelsize=tickfontsize, direction=direction, length=ticklength )

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axethickness)
                
    return fig, ax



def plotTP_v2(p_layers, T_samples, ax=None, xlim=[None,None], ylim=[1e-7,1e2], Tlabel='bottom', Plabel='left', color=None, axethickness = 2, fontsize = 16, tickthickness = 2, tickfontsize = 16, direction = 'in', ticklength = 3, labelpad=0, smooth = 40, alpha=1, showTPpoints=None,TPmarkersize=5, ylabel_coord=None):

    #Making figure
    if ax is None:
        fig,ax=plt.subplots()
    else:
        fig=None

    levels=[0.5,12,50,88,99.5]
    perc = np.nanpercentile(T_samples, levels, axis=0).T

    xnew = np.linspace(np.log10(p_layers).min(), np.log10(p_layers).max(), smooth)
    xnew2 = np.linspace(np.log10(p_layers).min(), np.log10(p_layers).max(), 1000)
    smooth_perc = np.zeros([smooth,5])
    smooth_perc2 = np.zeros([1000,5])

    for i in np.arange(5):
        spl = make_interp_spline(np.log10(p_layers), perc[:,i], k=3)  # type: BSpline
        smooth_perc[:,i] = spl(xnew)
        smooth_perc2[:,i] = spl(xnew2)

    # find nearest to 1 bar and 1 milibar and calculate slope
    # np.where()

    #fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
    ax.set_xlabel('Temperature [K]', fontsize=fontsize)
    ax.set_ylabel('Pressure [bar]', fontsize=fontsize, labelpad=labelpad)
    
    try:
        ax.yaxis.set_label_coords(ylabel_coord[0], ylabel_coord[1])
    except:
        pass

    ax.set_yscale('log')
    
    if color:
        ax.fill_betweenx(10**xnew/1e5,smooth_perc[:,0],smooth_perc[:,4],color=color, alpha = alpha-0.2,zorder=-10)
        ax.fill_betweenx(10**xnew/1e5,smooth_perc[:,1],smooth_perc[:,3],color=color, alpha = alpha,zorder=-9)
        ax.plot(smooth_perc[:,2],10**xnew/1e5,color=color,zorder=-8)
    else:
        ax.fill_betweenx(10**xnew/1e5,smooth_perc[:,0],smooth_perc[:,4],color=[0.9,0.9,1], alpha = alpha,zorder=-10)
        ax.fill_betweenx(10**xnew/1e5,smooth_perc[:,1],smooth_perc[:,3],color=[0.7,0.7,1], alpha = alpha,zorder=-9)
        ax.plot(smooth_perc[:,2],10**xnew/1e5,color='blue',zorder=-8)

    
    if np.any(showTPpoints):
        # inds = np.zeros(len(showTPpoints))
        inds = []
        for i, Ppoint in enumerate(showTPpoints):
            # inds[i] = int((np.abs(xnew-Ppoint)).argmin())
            ind = int((np.abs(xnew2-Ppoint)).argmin())
            inds.append(ind)
        
        ax.plot(smooth_perc2[:,2][inds],10**xnew2[inds]/1e5, linestyle='', markersize=TPmarkersize, marker='o',color='k', zorder=99)
        # ax.plot(smooth_perc2[:,2][inds],10**xnew2[inds]/1e5, linestyle='', markersize=TPmarkersize, marker='o', zorder=99)
        # line = ax.plot(smooth_perc[:,2][inds],10**xnew[inds]/1e5, linestyle='', markersize=TPmarkersize, marker='o',color=color, zorder=99)
        # line[0].set_clip_on(False)

    #ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    if Tlabel == 'top':
        ax.xaxis.set_label_position("top")
        ax.tick_params(labelbottom=False, labeltop=True, top = 'on', right = 'on', bottom='on', left='on', width=tickthickness, labelsize=tickfontsize, direction=direction, length=ticklength )
    else:
        # ax.tick_params(labelleft=None, labelright=True, labelbottom=True, labeltop=None, top = 'on', right = 'on', bottom='on', left='on', width=tickthickness, labelsize=tickfontsize, direction=direction, length=ticklength )
        ax.tick_params(labelbottom=True, labeltop=False, top = 'on', right = 'on', bottom='on', left='on', width=tickthickness, labelsize=tickfontsize, direction=direction, length=ticklength )

    if Plabel == 'right':
        ax.yaxis.set_label_position("right")
        ax.tick_params(labelleft=False, labelright=True, top = 'on', right = 'on', bottom='on', left='on', width=tickthickness, labelsize=tickfontsize, direction=direction, length=ticklength )


    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axethickness)

    ax.minorticks_off()
    
    # yticks = ax.get_yticks()
    # ytick_str = []
    # for tick in yticks:
    #     if tick == 1:
    #         ytick_str.append(r'1')
    #     else:
    #         ytick_str.append(r'10$^{%s}$' % int(np.log10(tick)))
    # ax.set_yticklabels(ytick_str)

    # ax.set_xticklabels(ax.get_xticks().astype(int), rotation=35 )

    # locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 ))
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    return fig, ax





#%%
        
#def prepSampPanda(fit):
#    '''make sampPanda from fit.samp (only 10000 samples, but allows get the Tp structure etc.)'''
#    
#    
#    fit.sampPanda=fit.panda.iloc[fit.sampInd]
#    
#    fit.sampPanda['index'] = fit.sampPanda.index
#    fit.sampPanda['ind']=np.arange(len(fit.sampPanda))
#    
#    fit.sampPanda['outgoingFluxTest']=fit.samp['outgoingFlux']
#    fit.sampPanda['T_100mbar']=fit.samp['T'][:,bisect(fit.atm.p,100*mbar)]
#    fit.sampPanda['T_0.1mbar']=fit.samp['T'][:,bisect(fit.atm.p,0.1*mbar)]
#    fit.sampPanda['DeltaTPhot'] = fit.sampPanda['T_100mbar'] - fit.sampPanda['T_0.1mbar']
#
#
#
#    if 'log(CH4/H2O)' in fit.panda.columns:
#        fit.sampPanda['log(CH4/H2O)'] = fit.panda.iloc[fit.sampInd]['log(CH4/H2O)'].values
#    if 'CH4' in fit.panda.columns:
#        fit.sampPanda['CH4'] = fit.panda.iloc[fit.sampInd]['CH4'].values
#    if 'H2O' in fit.panda.columns:
#        fit.sampPanda['H2O'] = fit.panda.iloc[fit.sampInd]['H2O'].values
#
#    fit.sampPanda.set_index('ind',inplace=True)
#
#    #samples = fit.panda.ix[fit.sampInd]
#

def PlotTPAndContribution(atm, T_samples,Specs,spectype='secEclppm',modelT=None,fig=None, Tlabel='bottom', axethickness = 2, fontsize = 15, tickthickness = 2, tickfontsize = 13, direction = 'in', ticklength = 3):
        specs=deepcopy(Specs)
        cFunctionList=[]
        specsCleaned=[]
        for i,spec in enumerate(specs):
            if spec.meta['spectype']==spectype:
                cFunctions=atm.contribution_function([spec])
                specsCleaned.append(specs[i])
                cFunctionList.append(cFunctions)
       
        
        if fig is None:
        
            fig=plt.figure(figsize=(30,23))
            gs=gridspec.GridSpec(1,5,hspace=0,wspace=0)
        else:
            gs0=gridspec.GridSpec(1,6,figure=fig)
            gs=gridspec.GridSpecFromSubplotSpec(1,5,subplot_spec=gs0[5])
        ax=fig.add_subplot(gs[0,:3])
        ax2=fig.add_subplot(gs[0,3])
        ax3=fig.add_subplot(gs[0,4])
        fontsize=15
        ax2.set_yscale('log')
        ax2.set_ylim([1e-9,1e3])
        ax2.invert_yaxis()

        wave=[]
        for spec in specsCleaned:
            wave.extend(spec['wave'])

        
    
        norm = mpl.colors.Normalize(vmin=min(wave), vmax=max(wave))
        colors=pl.cm.jet(norm(wave))
        pressure=atm.p/1e5
        j=0
        for cFunctions in cFunctionList:
            n=cFunctions.shape[1]  
            for i in range(0,n):
                mass=np.trapz(y=cFunctions[:,i],x=np.log10(pressure))
                cFunctions[:,i]=cFunctions[:,i]/mass
                ax2.plot(cFunctions[:,i],pressure,color=colors[i+j])
            j=j+n


        plt.setp(ax2.get_yticklabels(),visible=False)

        cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=pl.cm.jet,norm=norm)
        cb1.set_label(r'Wavelength $[\mu m]$')
        ax=plotTP(p_layers=atm.p, T_samples=T_samples, ax=ax,Tlabel='top',fontsize=fontsize)

        if modelT is not None:
            ax[1].plot(modelT,p_layers/1e5,color='r',label='Best fit')
            ax[1].legend()
        return fig




#%%

def plotSpecTpSamplesInOne(fit,randomInds=None,figFiles=[],
                          style=[1,1,0],xscale=['log','waveZoom'],bestFit=[1,1,1],
                          xlim=[None,None,None],ylim=[None,None,[1e-10,1e2]],
                          colorPara='',alpha=0.03,nSampPlotted=300,presAxis=False,
                          jointPlot=False, withTpModel=False,Tspline=False,figsize=[10.3375,9.625],save=True):


    if jointPlot:        
        fig = plt.figure(figsize=figsize,dpi=100)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1])
        axSpec=[]
        axSpec.append(fig.add_subplot(gs[0,:]))
        axSpec.append(fig.add_subplot(gs[1,0]))
        axTp=fig.add_subplot(gs[1,1])
        
        figSpec=[]
        figSpec.append(fig)
        figSpec.append(fig)
        figTp=fig
        
    else:
        axSpec=[]; figSpec=[]
        for ispec,spec in enumerate(fit.specs):
            if spec.meta['spectype']!='totalOutgoingFlux':
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_xlabel(r'Wavelength [$\mu$m]')
                figSpec.append(fig)
                axSpec.append(ax)
        figTp, axTp = plt.subplots(figsize=[8,10])

    
    #----------------------------------------
    if randomInds is None:
        size=np.min([nSampPlotted,len(fit.samples)])
        randomInds=np.sort(np.random.choice(np.arange(0,fit.samples['T'].shape[0]),size=size,replace=False))

    #if logWaveAxis is None:
    #    logWaveAxis = (fit.atm.waveRange[1]/fit.atm.waveRange[0])>2   

    #----------------------------------------
    p=fit.atm.p
    wave=fit.wavesm  
    T       =fit.samples['T']
    #qmol_lay=fit.samples['qmol_lay']
    if 'dppm' in fit.samples.keys():    
        dppm=fit.samples['dppm']
        axSpec[0].set_ylabel(r'Transit depth [ppm]')
    if 'secEclppm' in fit.samples.keys():    
        secEclppm=fit.samples['secEclppm']
    if 'thermal' in fit.samples.keys():    
        thermal=fit.samples['thermal']

    if colorPara!='':
        cval = fit.samples[colorPara].data
        sm=ut.makeColorScale(cval[np.isnan(cval)==False],cm=mpl.cm.jet)
    colorParaForPath = colorPara.replace("/", "").replace("(", "").replace(")", "").replace(":", "")   

     

    #----------------------------------------
    for ispec,spec in enumerate(fit.specs):

        spectype=spec.meta['spectype']
        if spectype!='totalOutgoingFlux':

            waveZoom=np.array([np.min(spec['waveMin']),np.max(spec['waveMax'])])
            extraWave=(waveZoom[1]-waveZoom[0])*0.2
            waveZoom=np.array([waveZoom[0]-extraWave,waveZoom[1]+extraWave])
            
            #Style 0
            if style[ispec]==0:

                fig=figSpec[ispec];  ax=axSpec[ispec]
                
                spec.plot(ax=ax,pretty=1)
                
                levels=[2.5,16,50,84,97.5]
                perc=np.zeros([len(wave),len(levels)])
                for iwave in range(len(wave)):
                    if spectype=='dppm':
                        perc[iwave,:] = np.nanpercentile(dppm[:,iwave],levels)
                    elif spectype=='secEclppm':
                        perc[iwave,:] = np.nanpercentile(secEclppm[:,iwave],levels)
                    elif spectype=='thermal':
                        perc[iwave,:] = np.nanpercentile(thermal[:,iwave],levels)
                ax.fill_between(wave,perc[:,0],perc[:,4],color=[0.8,0.8,1],zorder=-10)
                ax.fill_between(wave,perc[:,1],perc[:,3],color=[0.6,0.6,1],zorder=-9)
                ax.plot(wave,perc[:,2],color='blue',zorder=-8)   #Plot median spectrum
                if bestFit[ispec]:
                    ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)

                if xlim[ispec] is not None:
                    ax.set_xlim(xlim[ispec])
                if ylim[ispec] is not None:
                    ax.set_ylim(ylim[ispec])
                    
                if xscale[ispec]=='linear':
                    if not jointPlot and save:
                        filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_0_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
                elif xscale[ispec]=='waveZoom':
                    ax.set_xlim(waveZoom)
                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
                    if len(list(yy1))==1:
                        ax.set_ylim( yy1, yy2)
                    else:
                        ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
                        ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
                    if not jointPlot and save:
                        filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_0_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
                elif xscale[ispec]=='log':
                    #ax.set_xlim(auto=True);ax.set_ylim(ylim)
                    ax.set_xscale("log"); ut.xspeclog(ax,level=1)
                    if not jointPlot and save:
                        filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_0_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

                if spectype=='dppm':
                    ax.set_xlabel(r'Wavelength [$\mu$m]')
                    ax.set_ylabel('Transit Depth [ppm]')
                elif spectype=='secEclppm':
                    ax.set_xlabel(r'Wavelength [$\mu$m]')
                    ax.set_ylabel('Sec. Eclipse Depth [ppm]')
            
            #Style 1
            elif style[ispec]==1:

                fig=figSpec[ispec];  ax=axSpec[ispec]
                
                spec.plot(ax=ax,pretty=1)

                for i in randomInds:

                    #Select color
                    if colorPara!='':
                        color = sm.to_rgba(cval[i])
                    else:
                        color='blue'

                    #Plotting
                    if spectype=='dppm':
                        ax.plot(wave,dppm[i,:],color=color,zorder=-7,alpha=alpha)
                        ax.set_xlabel(r'Wavelength [$\mu$m]')
                        ax.set_ylabel('Transit Depth [ppm]')
                    elif spectype=='secEclppm':
                        ax.plot(wave,secEclppm[i,:],color=color,zorder=-7,alpha=alpha)
                        ax.set_xlabel(r'Wavelength [$\mu$m]')
                        ax.set_ylabel('Sec. Eclipse Depth [ppm]')
                    elif spectype=='thermal':
                        ax.plot(wave,thermal[i,:],color=color,zorder=-7,alpha=alpha)

                if colorPara!='':
                    plt.colorbar(sm).set_label(colorPara)
                if bestFit[ispec]:
                    ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
                    
                if xlim[ispec] is not None:
                    ax.set_xlim(xlim[ispec])
                if ylim[ispec] is not None:
                    ax.set_ylim(ylim[ispec])

                #ylim=ax.get_ylim()
                if xscale[ispec]=='linear':
                    if not jointPlot and save:
                        filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_'+colorParaForPath+'_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
                elif xscale[ispec]=='waveZoom':
                    ax.set_xlim(waveZoom)
                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
                    ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
                    #ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
                    if not jointPlot and save:
                        filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_'+colorParaForPath+'_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
                elif xscale[ispec]=='log':
                    #ax.set_xlim(auto=True);ax.set_ylim(ylim)
                    #fig.set_size_inches(10,6,forward=True);
                    ax.set_xscale("log"); ut.xspeclog(ax,level=1)
                    if not jointPlot and save:
                        filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_'+colorParaForPath+'_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

            if spectype=='dppm' and presAxis:
                print('Making pressure axis')
                ut.makeSecYAxis(ax,(fit.atm.RpBase+fit.atm.z)**2/fit.atm.Rstar**2 *1e6,fit.atm.p/100,np.hstack([np.array([1])*10**i for i in np.arange(-3.0,5.0)]),label='Pressure [mbar]')#,yminorticks2=np.hstack([np.array([1,2,4,6,8])*10**i for i in np.arange(-3.0,5.0)]))
                    
    #--Tp--------------------------------------
    if style[2]==0:
        fig=figTp; ax=axTp
        
        levels=[2.5,16,50,84,97.5]
        perc=np.zeros([len(p),len(levels)])
        for iLay in range(len(p)):
            perc[iLay,:] = np.nanpercentile(T[:,iLay],levels)     
        
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [bar]')
        ax.set_yscale('log')
        ax.fill_betweenx(p/1e5,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
        ax.fill_betweenx(p/1e5,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
        fit.atm.Teq = fit.atm.Teffstar*(fit.atm.Rstar/fit.atm.ap)**(0.5)*   (fit.atm.HeatDistFactor*(1-fit.atm.BondAlbedo))**(0.25)
        ax.axvline(x=fit.atm.Teq,color='k',linestyle='--')
        ax.plot(perc[:,2],p/1e5,color='blue',zorder=-8)
        if bestFit[2]:
            ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-7)
        ax.set_ylim()

        if xlim[2] is not None:
            ax.set_xlim(xlim[2])
        if ylim[2] is not None:
            ax.set_ylim(ylim[2])
            
        if withTpModel :
##        fileModel='../../scarlet_results/FwdRuns/HD_209458_b/HD_209458_b_Metallicity10.0_CtoO0.54_pCloud100000.0mbar_cHaze1e-10_pQuench1e-99_TpNonGrayConvTint75.0f0.25_atm.pkl' ####     
            fileModel='../../scarlet_results/FwdRuns/HD_209458_b/HD_209458_b_WellMixed_H2_0.763_He_0.23_CO_0.004_R2_atm.pkl'     
            atmMod = scarlet.loadAtm(fileModel)
            ax.plot(atmMod.T,atmMod.p/1e5,'darkmagenta', marker='x')
            ax.set_ylim(1e-7,1e2)

        ax.invert_yaxis()
        ax.minorticks_on()
        if Tspline:
            ax.set_xlim(0,3000)
        if not jointPlot and save:
            filename=fit.filebase+'Tp_1.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

    elif style[2]==1:
        fig=figTp; ax=axTp

        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [bar]')
        ax.set_yscale('log')
        for i in randomInds:
            #Choose color
            if colorPara!='':
                color = sm.to_rgba(cval[i])
            else:
                color='black'
            #Plot
            ax.plot(T[i,:],p/1e5,color=color,zorder=-7,alpha=alpha)

        if colorPara!='':
            plt.colorbar(sm).set_label(colorPara)
        else:
            ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-6)

        if xlim[2] is not None:
            ax.set_xlim(xlim[2])
        if ylim[2] is not None:
            ax.set_ylim(ylim[2])

        ax.invert_yaxis()
        ax.minorticks_on()
        if Tspline:
            ax.set_xlim(0,3000)
        if not jointPlot and save:
            filename=fit.filebase+'Tp_2_'+colorParaForPath+'.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

    if jointPlot and save:        
        figSpec[0].tight_layout()
        filename=fit.filebase+'SpecTpSamplesInOne.pdf'; figSpec[0].savefig(filename); figFiles.append([figSpec[0],filename])


    return figSpec,axSpec,figTp,axTp





















#%% OBSOLETE!!! DO NOT USE ANYMORE

def plotSpecSamples(fit,randomInds=None,figFiles=[],whichPlots=[0,1],logWaveAxis=None,plotTp=True,plotContribution=False,plotTtau=False,Tspline=False):

    #----------------------------------------
    if randomInds is None:
        size=np.min([300,len(fit.sampInd)])
        randomInds=np.sort(np.random.choice(np.arange(0,fit.samples['T'].shape[0]),size=size,replace=False))
#        randomInds=np.sort(np.random.choice(np.arange(0,len(fit.sampInd)),size=size,replace=False)) ## like previous version
    if logWaveAxis is None:
        logWaveAxis = (fit.atm.waveRange[1]/fit.atm.waveRange[0])>2   

    #----------------------------------------
    p=fit.atm.p
    wave=fit.wavesm  
    T       =fit.samples['T']
    # qmol_lay=fit.samples['qmol_lay']
    specs=deepcopy(fit.specs)
    if 'dppm' in fit.samples.colnames:    
        dppm=fit.samples['dppm']
    if 'secEclppm' in fit.samples.colnames:    
        secEclppm=fit.samples['secEclppm']
    if 'thermal' in fit.samples.colnames: 
        thermal=fit.samples['thermal']
        TBright=radutils.calcTBright(thermal,wave)
        fit.bestfitModel['Tbright']=radutils.calcTBright(fit.bestfitModel['thermal'],wave)


    for spec in specs:
        if spec.meta['spectype']=='secEclppm':
            thermalConversion=fit.atm.convertSecEclppmToThermal(spec)
            TBrightSpec=fit.atm.convertThermalToTBright(thermalConversion)

            specs.append(TBrightSpec)



    #----------------------------------------
    for ispec,spec in enumerate(specs):

        spectype=spec.meta['spectype']
        if spectype!='totalOutgoingFlux' and spectype!='highres' and fit.DoNotBlobArrays!=True:

            waveZoom=np.array([np.min(spec['waveMin']),np.max(spec['waveMax'])])
            extraWave=(waveZoom[1]-waveZoom[0])*0.2
            waveZoom=np.array([waveZoom[0]-extraWave,waveZoom[1]+extraWave])
            
            if whichPlots[0]:
                fig,ax=spec.plot(color='black')
                fig.set_size_inches(8,6,forward=True); 
                levels=[2.5,16,50,84,97.5]
                perc=np.zeros([len(wave),len(levels)])
                for iwave in range(len(wave)):
                    if spectype=='dppm':
                        perc[iwave,:] = np.nanpercentile(dppm[:,iwave],levels)
                        ax.set_ylabel('Transit depth [ppm]')
                    elif spectype=='secEclppm':
                        perc[iwave,:] = np.nanpercentile(secEclppm[:,iwave],levels)  
                        ax.set_ylabel('Eclipse depth [ppm]')
                    elif spectype=='thermal':
                        perc[iwave,:] = np.nanpercentile(thermal[:,iwave],levels)
                        ax.set_ylabel('Thermal Flux')
                    elif spectype=='Tbright':
                        perc[iwave,:] = np.nanpercentile(TBright[:,iwave],levels)
                        ax.set_ylabel('Brightness Temperature')
                #ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
                ax.fill_between(wave,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
                ax.fill_between(wave,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
                ax.plot(wave,perc[:,2],color='blue',zorder=-8)
                # ax.set_xlim([1.99,6.01])
                # ax.set_ylim([0.0,1600.0])
                ylim=ax.get_ylim()
                ax.set_title('')
                
                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
    
                if waveZoom is not None:
                    ax.set_xlim(waveZoom)
                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
                    if len(list(yy1))==1:
                        ax.set_ylim( yy1, yy2)
                    else:
                        ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
                        ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
    
                #ax.set_xlim(auto=True);ax.set_ylim(ylim)
                if logWaveAxis:
                    ax.set_xlim([2.49,6.01]); ax.set_ylim(ylim)
                    fig.set_size_inches(8,6,forward=True); ax.set_xscale("log"); ut.xspeclog(ax,level=1)
                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
        
            
            if whichPlots[1]:
                fig,ax=spec.plot(color='blue')
                for i in randomInds:
                    if spectype=='dppm':
                        ax.plot(wave,dppm[i,:],color='black',zorder=-7,alpha=0.1)
                    elif spectype=='secEclppm':
                        ax.plot(wave,secEclppm[i,:],color='black',zorder=-7,alpha=0.1)
                    elif spectype=='thermal':
                        ax.plot(wave,thermal[i,:],color='black',zorder=-7,alpha=0.1)
                    elif spectype=='Tbright':
                        ax.plot(wave,TBright[i,:],color='black',zorder=-7,alpha=0.1)
                ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
                ylim=ax.get_ylim()
                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

                if waveZoom is not None:
                    ax.set_xlim(waveZoom)
                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
                    if len(list(yy1))==1:
                        ax.set_ylim( yy1, yy2)
                    else:
                        ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
                        ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
                
                ax.set_xlim(auto=True);ax.set_ylim(ylim)
                if logWaveAxis:
                    fig.set_size_inches(10,6,forward=True); ax.set_xscale("log"); ut.xspeclog(ax,level=1)
                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])



    #----------------------------------------
    if plotTp:
        
        if whichPlots[0]:
            
            levels=[2.5,16,50,84,97.5]
            perc=np.zeros([len(p),len(levels)])
            for iLay in range(len(p)):
                perc[iLay,:] = np.nanpercentile(T[:,iLay],levels)    
            
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            ax.set_yscale('log')
            ax.fill_betweenx(p/1e5,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
            ax.fill_betweenx(p/1e5,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
            fit.atm.Teq = fit.atm.Teffstar*(fit.atm.Rstar/fit.atm.ap)**(0.5)*   (fit.atm.HeatDistFactor*(1-fit.atm.BondAlbedo))**(0.25)
            plt.axvline(x=fit.atm.Teq,color='k',linestyle='--')
            ax.plot(perc[:,2],p/1e5,color='blue',zorder=-8)
            ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-7)
            ax.set_ylim([1e-10,1e2])
            ax.invert_yaxis()
            ax.minorticks_on()
            if Tspline:
                ax.set_xlim(0,3000)
            filename=fit.filebase+'Tp_1.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

            if plotContribution==True:
                if specs[0].meta['spectype'] != 'highres':
                    fig=PlotTPAndContribution(fit.atm,T,specs,spectype=specs[0].meta['spectype']) 
                    filename=fit.filebase+'Tp_Contributions_'+specs[0].meta['spectype']+'.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            ax.set_yscale('log')
            for i in randomInds:
                ax.plot(T[i,:],p/1e5,color='black',zorder=-7,alpha=0.1)
            ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-6)
            ax.set_ylim([1e-10,1e2])
            ax.invert_yaxis()
            ax.minorticks_on()
            if Tspline:
                ax.set_xlim(0,3000)
            filename=fit.filebase+'Tp_2.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

    #----------------------------------------
    if plotTtau:
        
        if whichPlots[0]:   
            
            nTFreeLayers = int(fit.atm.modelSetting['TempType'][-2:])
            log_tau_layers = np.linspace(np.log10(1e-8),np.log10(1e5),nTFreeLayers) #nTFreeLayers from tau=XX to XX equally spaced in log

            T_bestFit = []
            symbols = [x.symbol for x in fit.para if x.symbol[0]=='T']
            for s in symbols:
                T_bestFit.append(fit.panda.iloc[np.where((fit.panda['step']==fit.bestfit['step'])*(fit.panda['walker']==fit.bestfit['walker']))][s].tolist()[0])
            T_bestFit = np.array(T_bestFit)
            
            tau = 10**log_tau_layers
            levels=[2.5,16,50,84,97.5]
            perc=np.zeros([len(log_tau_layers),len(levels)])
            for i in range(len(log_tau_layers)):
                if len(str(i))==1 : istr = '0'+str(i)
                else : istr = str(i)
                perc[i,:] = np.nanpercentile(fit.panda['T'+istr][:],levels)    
            
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Optical depth')
            ax.set_yscale('log')
            ax.fill_betweenx(tau,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
            ax.fill_betweenx(tau,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
            fit.atm.Teq = fit.atm.Teffstar*(fit.atm.Rstar/fit.atm.ap)**(0.5)*   (fit.atm.HeatDistFactor*(1-fit.atm.BondAlbedo))**(0.25)
            plt.axvline(x=fit.atm.Teq,color='k',linestyle='--')
            ax.plot(perc[:,2],tau,color='blue',zorder=-8)
            ax.plot(T_bestFit,tau,'red',lw=1.5,zorder=-7)
            ax.set_ylim([1e-8,1e5])
            ax.invert_yaxis()
            ax.minorticks_on()
            filename=fit.filebase+'Ttau_1.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

    return figFiles



















#%%

#def plotSpecSamples2(fit,randomInds=None,figFiles=[],whichPlots=[0,1],logWaveAxis=None,
#                    plotTp=True,colorPara='',alpha=0.3,filterVal=None,nSampPlotted=300,bestFit=False):
#
#    #----------------------------------------
#    if bestFit==False:
#        if randomInds is None:
#            size=np.min([nSampPlotted,len(fit.sampInd)])
#            randomInds=np.sort(np.random.choice(np.arange(0,len(fit.sampInd)),size=size,replace=False))
#    else:
#        randomInds=[800]    
#
#    if logWaveAxis is None:
#        logWaveAxis = (fit.atm.waveRange[1]/fit.atm.waveRange[0])>2   
#
#    #----------------------------------------
#    p=fit.atm.p
#    wave=fit.wavesm  
#    T       =fit.samp['T']
#    #qmol_lay=fit.samp['qmol_lay']
#    if 'dppm' in fit.samp.keys():    
#        dppm=fit.samp['dppm']
#    if 'secEclppm' in fit.samp.keys():    
#        secEclppm=fit.samp['secEclppm']
#    if 'thermal' in fit.samp.keys():    
#        thermal=fit.samp['thermal']
#
#    if filterVal is None:
#        filterVal=np.full(len(fit.sampInd), True, dtype=bool)
#
#    if colorPara!='':
#        cval = fit.panda.iloc[fit.sampInd][colorPara].values  #fit.samp['T'][:,39]
#        sm=ut.makeColorScale(cval[np.logical_and(np.isnan(cval)==False,filterVal)],cm=mpl.cm.jet)
#        
#    colorParaForPath = colorPara.replace("/", "").replace("(", "").replace(")", "").replace(":", "")   
#
#     
#
#    #----------------------------------------
#    for ispec,spec in enumerate(fit.specs):
#
#        spectype=spec.meta['spectype']
#        if spectype!='totalOutgoingFlux':
#
#            waveZoom=np.array([np.min(spec['waveMin']),np.max(spec['waveMax'])])
#            extraWave=(waveZoom[1]-waveZoom[0])*0.2
#            waveZoom=np.array([waveZoom[0]-extraWave,waveZoom[1]+extraWave])
#            
#            if whichPlots[0]:
#                fig,ax=spec.plot(color='black')
#                levels=[2.5,16,50,84,97.5]
#                perc=np.zeros([len(wave),len(levels)])
#                for iwave in range(len(wave)):
#                    if spectype=='dppm':
#                        perc[iwave,:] = np.percentile(dppm[:,iwave],levels)
#                    elif spectype=='secEclppm':
#                        perc[iwave,:] = np.percentile(secEclppm[:,iwave],levels)        
#                    elif spectype=='thermal':
#                        perc[iwave,:] = np.percentile(thermal[:,iwave],levels)        
#                ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
#                ax.fill_between(wave,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
#                ax.fill_between(wave,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
#                ax.plot(wave,perc[:,2],color='blue',zorder=-8)
#                ylim=ax.get_ylim()
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#    
#                if waveZoom is not None:
#                    ax.set_xlim(waveZoom)
#                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
#                    ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
#                    ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
#                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#    
#                ax.set_xlim(auto=True);ax.set_ylim(ylim)
#                if logWaveAxis:
#                    fig.set_size_inches(10,6,forward=True); ax.set_xscale("log", nonposx='clip'); ut.xspeclog(ax,level=1)
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#        
#            
#            if whichPlots[1]:
#                fig,ax=spec.plot(color='blue')
#                for i in randomInds:
#                    if colorPara!='':
#                        color = sm.to_rgba(cval[i])
#                    else:
#                        color='black'
#
#                    if filterVal[i]:
#                        if spectype=='dppm':
#                            ax.plot(wave,dppm[i,:],color=color,zorder=-7,alpha=alpha)
#                        elif spectype=='secEclppm':
#                            ax.plot(wave,secEclppm[i,:],color=color,zorder=-7,alpha=0.1)
#                        elif spectype=='thermal':
#                            ax.plot(wave,thermal[i,:],color=color,zorder=-7,alpha=alpha)
#
#                if colorPara!='':
#                    plt.colorbar(sm).set_label(colorPara)
#                else:
#                    ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
#                ylim=ax.get_ylim()
#
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_'+colorParaForPath+'_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#                if waveZoom is not None:
#                    ax.set_xlim(waveZoom)
#                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
#                    ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
#                    ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
#                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_'+colorParaForPath+'_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#                
#                ax.set_xlim(auto=True);ax.set_ylim(ylim)
#                if logWaveAxis:
#                    fig.set_size_inches(10,6,forward=True); ax.set_xscale("log", nonposx='clip'); ut.xspeclog(ax,level=1)
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_'+colorParaForPath+'_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#
#
#    #----------------------------------------
#    if plotTp:
#        
#        if whichPlots[0]:
#            
#            levels=[2.5,16,50,84,97.5]
#            perc=np.zeros([len(p),len(levels)])
#            for iLay in range(len(p)):
#                perc[iLay,:] = np.percentile(T[:,iLay],levels)        
#            
#            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
#            ax.set_xlabel('Temperature [K]')
#            ax.set_ylabel('Pressure [bar]')
#            ax.set_yscale('log')
#            ax.fill_betweenx(p/1e5,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
#            ax.fill_betweenx(p/1e5,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
#            fit.atm.Teq = fit.atm.Teffstar*(fit.atm.Rstar/fit.atm.ap)**(0.5)*   (fit.atm.fdash*(1-fit.atm.BondAlbedo))**(0.25)
#            plt.axvline(x=fit.atm.Teq,color='k',linestyle='--')
#            ax.plot(perc[:,2],p/1e5,color='blue',zorder=-8)
#            ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-7)
#            ax.set_ylim([1e-10,1e2])
#            ax.invert_yaxis()
#            ax.minorticks_on()
#            filename=fit.filebase+'Tp_1.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#
#        if whichPlots[1]:
#            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
#            ax.set_xlabel('Temperature [K]')
#            ax.set_ylabel('Pressure [bar]')
#            ax.set_yscale('log')
#            for i in randomInds:
#                if colorPara!='':
#                    color = sm.to_rgba(cval[i])
#                else:
#                    color='black'
#                if filterVal[i]:
#                    ax.plot(T[i,:],p/1e5,color=color,zorder=-7,alpha=alpha)
#
#            if colorPara!='':
#                plt.colorbar(sm).set_label(colorPara)
#            else:
#                ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-6)
#
#            ax.set_ylim([1e-10,1e2])
#            ax.invert_yaxis()
#            ax.minorticks_on()
#            filename=fit.filebase+'Tp_2_'+colorParaForPath+'.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#    return figFiles
#















#%%
  
#def plotSpecSamplesV2(fit,randomInds=None,figFiles=[],whichPlots=[0,1],logWaveAxis=None,
#                    plotTp=True,colorPara='',alpha=0.3,filterVal=None,nSampPlotted=300):
#
#    #----------------------------------------
#    if randomInds is None:
#        size=np.min([nSampPlotted,len(fit.sampPanda)])
#        randomInds=np.sort(np.random.choice(np.arange(0,len(fit.sampPanda)),size=size,replace=False))
#    if logWaveAxis is None:
#        logWaveAxis = (fit.atm.waveRange[1]/fit.atm.waveRange[0])>2   
#
#    #----------------------------------------
#    p=fit.atm.p
#    wave=fit.wavesm  
#    T       =fit.samp['T']
#    #qmol_lay=fit.samp['qmol_lay']
#    if 'dppm' in fit.samp.keys():    
#        dppm=fit.samp['dppm']
#    if 'secEclppm' in fit.samp.keys():    
#        secEclppm=fit.samp['secEclppm']
#    if 'thermal' in fit.samp.keys():    
#        thermal=fit.samp['thermal']
#
#    if filterVal is None:
#        filterVal=np.full(len(fit.sampPanda), True, dtype=bool)
#
#    if colorPara!='':
#        cval = fit.sampPanda[colorPara].values
#        sm=ut.makeColorScale(cval,cm=mpl.cm.jet)
#        
#    colorParaForPath = colorPara.replace("/", "").replace("(", "").replace(")", "").replace(":", "")   
#
#     
#
#    #----------------------------------------
#    for ispec,spec in enumerate(fit.specs):
#
#        spectype=spec.meta['spectype']
#        if spectype!='totalOutgoingFlux':
#
#            waveZoom=np.array([np.min(spec['waveMin']),np.max(spec['waveMax'])])
#            extraWave=(waveZoom[1]-waveZoom[0])*0.2
#            waveZoom=np.array([waveZoom[0]-extraWave,waveZoom[1]+extraWave])
#            
#            if whichPlots[0]:
#                fig,ax=spec.plot(color='black')
#                levels=[2.5,16,50,84,97.5]
#                perc=np.zeros([len(wave),len(levels)])
#                for iwave in range(len(wave)):
#                    if spectype=='dppm':
#                        perc[iwave,:] = np.percentile(dppm[:,iwave],levels)
#                    elif spectype=='secEclppm':
#                        perc[iwave,:] = np.percentile(secEclppm[:,iwave],levels)        
#                    elif spectype=='thermal':
#                        perc[iwave,:] = np.percentile(thermal[:,iwave],levels)        
#                #ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
#                ax.fill_between(wave,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
#                ax.fill_between(wave,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
#                ax.plot(wave,perc[:,2],color='blue',zorder=-8)
#                ylim=ax.get_ylim()
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#    
#                if waveZoom is not None:
#                    ax.set_xlim(waveZoom)
#                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
#                    ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
#                    ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
#                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#    
#                ax.set_xlim(auto=True);ax.set_ylim(ylim)
#                if logWaveAxis:
#                    fig.set_size_inches(10,6,forward=True); ax.set_xscale("log", nonposx='clip'); ut.xspeclog(ax,level=1)
#                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_1_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#        
#            
#            if whichPlots[1]:
#                fig,ax=spec.plot(color='blue')
#                for i in randomInds:
#                    if colorPara!='':
#                        color = sm.to_rgba(cval[i])
#                    else:
#                        color='black'
#
#                    if filterVal[i]:
#                        if spectype=='dppm':
#                            ax.plot(wave,dppm[i,:],color=color,zorder=-7,alpha=alpha)
#                        elif spectype=='secEclppm':
#                            ax.plot(wave,secEclppm[i,:],color=color,zorder=-7,alpha=0.1)
#                        elif spectype=='thermal':
#                            ax.plot(wave,thermal[i,:],color=color,zorder=-7,alpha=alpha)
#
#                if colorPara!='':
#                    plt.colorbar(sm).set_label(colorPara)
#                else:
#                    ax.plot(wave,fit.bestfitModel[spectype],'red',lw=1.5,zorder=-7)
#                ylim=ax.get_ylim()
#
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_'+colorParaForPath+'_lin.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#                if waveZoom is not None:
#                    ax.set_xlim(waveZoom)
#                    xx=spec['wave']; yy1=spec['yval']-1.5*spec['yerrLow']; yy2=spec['yval']+1.5*spec['yerrUpp']
#                    ind = np.where( (xx > waveZoom[0]) &  (xx < waveZoom[1]) )[0]
#                    ax.set_ylim( yy1[ind].min(), yy2[ind].max() )
#                    filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_'+colorParaForPath+'_zoom.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#                
#                ax.set_xlim(auto=True);ax.set_ylim(ylim)
#                if logWaveAxis:
#                    fig.set_size_inches(10,6,forward=True); ax.set_xscale("log", nonposx='clip'); ut.xspeclog(ax,level=1)
#                filename=fit.filebase+'SpectraFit_'+str(ispec)+'_'+spectype+'_2_'+colorParaForPath+'_log.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#
#
#    #----------------------------------------
#    if plotTp:
#        
#        if whichPlots[0]:
#            
#            levels=[2.5,16,50,84,97.5]
#            perc=np.zeros([len(p),len(levels)])
#            for iLay in range(len(p)):
#                perc[iLay,:] = np.percentile(T[filterVal,iLay],levels)        
#            
#            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
#            ax.set_xlabel('Temperature [K]')
#            ax.set_ylabel('Pressure [bar]')
#            ax.set_yscale('log')
#            ax.fill_betweenx(p/1e5,perc[:,0],perc[:,4],color=[0.9,0.9,1],zorder=-10)
#            ax.fill_betweenx(p/1e5,perc[:,1],perc[:,3],color=[0.7,0.7,1],zorder=-9)
#            fit.atm.Teq = fit.atm.Teffstar*(fit.atm.Rstar/fit.atm.ap)**(0.5)*   (fit.atm.fdash*(1-fit.atm.BondAlbedo))**(0.25)
#            plt.axvline(x=fit.atm.Teq,color='k',linestyle='--')
#            ax.plot(perc[:,2],p/1e5,color='blue',zorder=-8)
#            ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-7)
#            ax.set_ylim([1e-10,1e2])
#            ax.invert_yaxis()
#            ax.minorticks_on()
#            filename=fit.filebase+'Tp_1.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#
#        if whichPlots[1]:
#            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
#            ax.set_xlabel('Temperature [K]')
#            ax.set_ylabel('Pressure [bar]')
#            ax.set_yscale('log')
#            for i in randomInds:
#                if colorPara!='':
#                    color = sm.to_rgba(cval[i])
#                else:
#                    color='black'
#                if filterVal[i]:
#                    ax.plot(T[i,:],p/1e5,color=color,zorder=-7,alpha=alpha)
#
#            if colorPara!='':
#                plt.colorbar(sm).set_label(colorPara)
#            else:
#                ax.plot(fit.bestfitModel['T'],p/1e5,'red',lw=1.5,zorder=-6)
#
#            ax.set_ylim([1e-10,1e2])
#            ax.invert_yaxis()
#            ax.minorticks_on()
#            filename=fit.filebase+'Tp_2_'+colorParaForPath+'.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
#
#    return figFiles
#

   

#%% Likelihood_Panel1D 
    
def likelihoodRatioPanel1D(fit,figFiles=[]):
    '''
    Makes likelihood panel from fit.panda
    '''
    
    keys=fit.panda.keys()[7:]

    nx,ny = np.int(np.sqrt(len(keys)-1)+1),  np.int((len(keys)-1)/(np.int(np.sqrt(len(keys)-1)+1)))+1
    fig,axs=plt.subplots(nx,ny,figsize=[30,20],dpi=40)

    axs=np.array(axs).flatten()

    maxLnLike = np.nanmax(fit.panda['lnlike'].values)

    for ikey,key in enumerate(keys):
        ax=axs[ikey]
        
        x=fit.panda[key].values
        
        LR = np.exp(maxLnLike-fit.panda['lnlike'].values)
        
        y=LR #fit.panda['chi2'].values/np.min(fit.panda['chi2'])

#            if ikey>=2:
#                ax.set_xlim([-9,0])
#                borders=np.linspace(-9,0,40)
#            else:
        borders=np.linspace(np.percentile(ut.nanfree(x),0.001),np.percentile(ut.nanfree(x),99.99),40)

        centers=0.5*(borders[:-1]+borders[1:])
        besty=np.zeros_like(centers)
        for i in range(len(borders)-1):
            besty[i]=np.nanmin(np.r_[1e30,y[np.logical_and(x>borders[i],x<borders[i+1])]])
             
#        if key=='logH2O':
#            pdb.set_trace()
            
        ax.step(centers,besty,where='mid', label='mid')
        ax.set_xlabel(key)
        ax.set_ylabel(r'Likelihood ratio')
        ax.set_ylim([-0.4,30])
        
        
        for sigma in [1,2,3]:
            ax.axhline(stats.norm.pdf(0)/stats.norm.pdf(sigma),color='k',ls='--')
        
        ax.invert_yaxis()
    
    if mpl.get_backend()!='pdf':
        plt.tight_layout()    

    filename=fit.filebase+'LikelihoodPanel1D_AllPanda.pdf'; fig.savefig(filename); figFiles.append([fig,filename])






#%% Likelihood_Panel1D 
    
def likelihoodPanel1D(fit,figFiles=[]):
    '''
    Makes likelihood panel from fit.panda
    '''
    
    keys=fit.panda.keys()[7:]

    nx,ny = np.int(np.sqrt(len(keys)-1)+1),  np.int((len(keys)-1)/(np.int(np.sqrt(len(keys)-1)+1)))+1
    fig,axs=plt.subplots(nx,ny,figsize=[30,20],dpi=50)

    axs=np.array(axs).flatten()

    for ikey,key in enumerate(keys):
        ax=axs[ikey]
        
        x=fit.panda[key].values
        y=fit.panda['chi2'].values-np.min(fit.panda['chi2'])

#            if ikey>=2:
#                ax.set_xlim([-9,0])
#                borders=np.linspace(-9,0,40)
#            else:
        borders=np.linspace(np.percentile(ut.nanfree(x),0.001),np.percentile(ut.nanfree(x),99.99),40)

        centers=0.5*(borders[:-1]+borders[1:])
        bestchi2=np.zeros_like(centers)
        for i in range(len(borders)-1):
            bestchi2[i]=np.nanmin(np.r_[1e30,y[np.logical_and(x>borders[i],x<borders[i+1])]])
        ax.step(centers,bestchi2,where='mid', label='mid')
        ax.set_xlabel(key)
        ax.set_ylabel(r'$\chi^{2} - \chi^{2}_{min}$')
        ax.set_ylim([-0.4,30])
        ax.axhline(9,color='k',ls='--')
        ax.invert_yaxis()
    
    if mpl.get_backend()!='pdf':
        plt.tight_layout()    

    filename=fit.filebase+'LikelihoodPanel1D_AllPanda.pdf'; fig.savefig(filename); figFiles.append([fig,filename])



  

#%% Bayesian Probability versus Profile Likelihood (Panel)


def bayesianProb_vs_ProfLike(fit,figFiles=[],solarVal=dict(),keys=None):
    #keys=['logH2O']
   
    if 'df' in fit.__dict__.keys():
        df=fit.df
        listOfPara=fit.listOfPara
    else:
        df,listOfPara,keptWalkers,removeWalkers=makeCleanedChain(fit,figFiles,plotting=False)   #could be sped up but not doing it again

    figsize=[24,24]
    if keys is None:
        keys=listOfPara 
    elif keys=='all':
        keys=df.keys()[7:]
#        keys=[]
#        for key in df.keys()[7:]:
#            np.any(np.isfinite(fit.df['H2S']))
        figsize=[40,24]

    nx,ny = np.int(np.sqrt(len(keys)-1)+1),  np.int((len(keys)-1)/(np.int(np.sqrt(len(keys)-1)+1)))+1
    if len(keys)>1:
        fig,axs=plt.subplots(nx,ny,figsize=figsize,dpi=40); axs=np.array(axs).flatten()
    else:
        fig,axs=plt.subplots(1,1); axs=np.array(axs).flatten()

    fig.suptitle(fit.pla.Name + '  ' + str([spec.meta['spectype'] for spec in fit.specs]))

    for ikey,key in enumerate(keys):
        

        
        ax=axs[ikey]
        x=df[key].values
        
        #----Bayesian Posterior--------------------------------------
        if len(np.unique(df[key]))>1:
            maxn=pr.hist1d(x,ax=ax,labels=[key],bins=30,quantiles=[0.163,0.5,0.847],show_titles=True,
                               showQuantilesBar=True,fig=fig,color='C0',stacked=True)
            #        if ikey<7:
            #            ax.set_xlim([-9,-1])
    
            ax.set_xlabel(key)
            #ax.set_ylabel(r'Probability')
            #ax.invert_yaxis()
            
            if key in solarVal:
                ax.axvline(solarVal[key],ls='--',color='r',lw=1)
    
        
            #---Profile Likelihood-----------------------------------------
            y=np.exp( df['lnlike'].values - np.max(df['lnlike']) ) 
            borders=np.linspace(np.percentile(x,0),np.percentile(x,100),40)
            centers=0.5*(borders[:-1]+borders[1:])
            bestLike=np.zeros_like(centers)
            for i in range(len(borders)-1):
                bestLike[i]=np.nanmax(np.r_[-99,y[np.logical_and(x>borders[i],x<borders[i+1])]])
    
            #        if ikey<7:
            #            ax.set_xlim([-9,-1])
    
            ax.step(centers,bestLike * maxn,where='mid',color='C2',lw=1,zorder=-10)
        
            if ikey==0:
                print(bestLike)
        
            ax.set_xlabel(key)
            #ax.set_ylabel(r'$\chi^{2} - \chi^{2}_{min}$')
            ax.set_ylim([0,maxn])
            #ax.axhline(9,color='k',ls='--')
        
        if key in solarVal:
            ax.axvline(solarVal[key],ls='--',color='r',lw=1)   
    
    
            
    ax.legend(['Prof Like','Bayesian post.'])

    if mpl.get_backend()!='pdf':
        plt.tight_layout()    

    filename=fit.filebase+'Panel1D_BayesianProb_vs_ProfLike.pdf'; fig.savefig(filename); figFiles.append([fig,filename])




#%%
#%%
    

#%% plots from df
    
def plotPanel1d_fromdf(fit,figFiles=[]):
    figFiles.append(pr.panel1d(fit.df[fit.listOfPara],saveFileBase=fit.filebase,paraStr='FitParas'))
    figFiles.append(pr.panel1d(fit.df[7:]            ,saveFileBase=fit.filebase,paraStr='AllPanda'))
    return figFiles


def plotTriangle_fromdf(fit,figFiles=[]):
    if fit.istep>10:
        figFiles.append(pr.triangle(fit.df[fit.listOfPara],bins=20,saveFileBase=fit.filebase,paraStr='AllFitPara',plot_datapoints=False,dpi=30))
    return figFiles

def plotCorner_fromdf(fit,figFiles=[]):
    if fit.istep>10:
        figFiles.append(pr.triangle(fit.df[fit.listOfPara],bins=20,saveFileBase=fit.filebase,paraStr='AllFitPara',plot_datapoints=False,dpi=30,smooth=1.0))
    return figFiles




def plotChain_fromdf(fit,figFiles=[]):
    figFiles.append(pr.chainplot(fit.df[['chi2','redchi2']+fit.listOfPara],nwalkers=len(fit.keptWalkers),fontsize=8,saveFileBase=fit.filebase+'burnInRem',paraStr='AllFitPara'))
    return figFiles



#%% Likelihood_Panel1D All Cases in One
    
def plotAllCasesLikelihoodPanel1D_fromdf(fits,solarVal=dict(),figFiles=[],keys=None,savefilebase=None):
    '''
    #keys=['logH2O']
    '''
    
    figsize=[24,24]
    if keys is None:
        keys=fits[0].listOfPara 
        fileExt=''
    elif keys=='all':
        keys=fits[0].df.keys()[7:]
        figsize=[40,24]
        fileExt='all'
    else:
        fileExt=keys[0]
        
    nx,ny = np.int(np.sqrt(len(keys)-1)+1),  np.int((len(keys)-1)/(np.int(np.sqrt(len(keys)-1)+1)))+1
    if len(keys)>4:
        fig,axs=plt.subplots(nx,ny,figsize=figsize,dpi=40); axs=np.array(axs).flatten()
    else:
        fig,axs=plt.subplots(1,1); axs=np.array(axs).flatten()

    fig.suptitle(fits[0].pla.Name + '  AllCases LikelihoodPanel1D_fromdf' )

    for fit in fits:
        for ikey,key in enumerate(keys):
            ax=axs[ikey]
            
            x=fit.df[key].values
            y=np.exp( fit.df['lnlike'].values - np.max(fit.df['lnlike']) )  #fit.df['chi2'].values-np.min(fit.df['chi2'])
            borders=np.linspace(np.percentile(x,0),np.percentile(x,100),40)
            centers=0.5*(borders[:-1]+borders[1:])
            bestLike=np.zeros_like(centers)
            for i in range(len(borders)-1):
                bestLike[i]=np.nanmax(np.r_[-99,y[np.logical_and(x>borders[i],x<borders[i+1])]])

            if len(keys)>6 and ikey<6:
                ax.set_xlim([-9,-0.5])

            ax.step(centers,bestLike,where='mid',label=fit.label)
            ax.set_xlabel(key)
            #ax.set_ylabel(r'$\chi^{2} - \chi^{2}_{min}$')
            ax.set_ylim([0,1])
            #ax.axhline(9,color='k',ls='--')
            
            if key in solarVal:
                ax.axvline(solarVal[key],ls='--',color='r',lw=1)

            if ikey==len(keys)-1:
                ax.legend(fontsize=8)
                
        plt.tight_layout()

    if savefilebase is not None:
        filename=savefilebase+'Likelihood_Panel1D_'+fileExt+'.pdf'; fig.savefig(filename); figFiles.append([fig,filename])

    return fig,axs





#%% Bayesian Posterior Panel1D All Cases in One

def plotAllCasesBayesianPanel1D_fromdf(fits,solarVal=dict(),figFiles=[],keys=None,savefilebase=None,
                                       showQuantilesBar=True):
    '''
    #keys=['logH2O']
    '''
    
    
    if keys is None:
        keys=fits[0].listOfPara 
        figsize=[24,24];dpi=40
        fileExt=''
    elif keys=='all':
        keys=fits[0].df.keys()[7:]
        figsize=[40,24];dpi=40
        fileExt='all'
    else:
        figsize=[8,4];dpi=100
        fileExt=keys[0]

    print(figsize)

    nx,ny = np.int(np.sqrt(len(keys)-1)+1),  np.int((len(keys)-1)/(np.int(np.sqrt(len(keys)-1)+1)))+1
    if len(keys)>4:
        fig,axs=plt.subplots(nx,ny,figsize=figsize,dpi=dpi); axs=np.array(axs).flatten()
    else:
        fig,axs=plt.subplots(1,1); axs=np.array(axs).flatten()

    fig.suptitle(fits[0].pla.Name + '  AllCases BayesianPanel1D_fromdf' )

    for ifit,fit in enumerate(fits):
        
        for ikey,key in enumerate(keys):
            ax=axs[ikey]
            
            x=fit.df[key].values
            
            if len(np.unique(fit.df[key]))>1:
                pr.hist1d(x,ax=ax,labels=[key],bins=30,quantiles=[0.163,0.5,0.847],show_titles=True,showQuantilesBar=showQuantilesBar,fig=fig,color='C'+str(ifit),hist_kwargs={'label':fit.label})
                #color='C'+str(2-ifit)
            if len(keys)>6 and ikey<6:
                ax.set_xlim([-9,0.5])

            ax.set_xlabel(key)
            #ax.set_ylabel(r'Probability')
            #ax.invert_yaxis()
            
            #            [np.percentile(x[np.isfinite(x)],0),np.percentile(x[np.isfinite(x)],100)]
            #            ax.set_xlim()
            
            
            if (ifit==0) and (key in solarVal):
                ax.axvline(solarVal[key],ls='--',color='r',lw=1,label='solar')

            if ikey==len(keys)-1:
                ax.legend(loc='auto',fontsize=8)
        
        plt.tight_layout()

    if savefilebase is not None:
        filename=savefilebase+'Posterior_Panel1D_'+ut.str2path(fileExt)+'.pdf'; fig.savefig(filename); figFiles.append([fig,filename])
        fig.filename=filename

    return fig,axs
    








