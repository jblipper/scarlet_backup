import numexpr as ne
import pdb
import scarlet
from scarlet import radutils as rad
from numba import jit, vectorize
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10
import matplotlib.pyplot as plt
from auxbenneke.constants import unitfac, pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom
from copy import deepcopy
import pickle

# Specify the path to your pickle file
pickle_file_path = 'modelSetting_and_params.pkl'

with open(pickle_file_path, 'rb') as f:
    # Load the data from the pickle file
    modelSetting,params = pickle.load(f)
firstIter=False
LucyUnsold=False
runConvection=True

if __name__ == "__main__":
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240703_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240709_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    
atm.readInAbunLUT()

atm.sigmaAtm = np.zeros([atm.nLay,atm.nAbsMol,atm.nWave], dtype=atm.numerical_precision)

atm.T=np.ones(60)*2000

def calcNonGrayTpTradProf(atm,modelSetting,params,firstIter,LucyUnsold,runConvection):   
    
    print('running TpTrad')
    print(f'modelSetting: {modelSetting}')
    print(f'params: {params}')
    print(f'firstIter: {firstIter:}')
    print(f'LucyUnsold: {LucyUnsold}')
    print(f'runConvection: {runConvection}')
    
    T = atm.Teq*np.ones_like(atm.p, dtype=atm.numerical_precision)*1.4
    N=atm.nLay
    levels=deepcopy(T)    
    atm.TList =np.array([np.array([level]) for level in deepcopy(levels)])
    
    #atm.calcAtmosphere(modelSetting,params,updateOnlyDppm=False,updateTp=False,disp2terminal=False,returnOpac=False,
    #               thermalReflCombined=None,thermalOnly=None,albedoOnly=None, low_res_mode=False)
    
    ###atm.qmol_lay                                                                               = atm.calcComposition(modelSetting,params,levels, firstIter=False)
    #atm.executeOption()
   ### atm.z,atm.dz,atm.grav,atm.ntot,atm.nmol,atm.MuAve,atm.scaleHeight,atm.RpBase,atm.r = atm.calcHydroEqui(modelSetting,params,levels)
   ### atm.extinctCoef,atm.absorbCoef, atm.scatCoef                                             = atm.calcOpacities(modelSetting,params,levels,saveOpac=atm.saveOpac, low_res_mode=False)
    
    extinctCoef=atm.extinctCoef.copy()
    scatCoef=atm.scatCoef.copy()       

    #------------------------------------------------------------------------#
    #---------------Calculate temperature perturbations----------------------#
    #------------------------------------------------------------------------#
    
    deltas=np.ones(N)*11
    count=0
    while np.max(np.abs(deltas))>0.2*20: #0.5:
        deltas,levels=atm.take_step_tp_trad(atm.TList,levels,N,extinctCoef,scatCoef,loop=False)
        atm.TList = np.c_[atm.TList,levels]
        
        #atm.T=levels 
        #atm.calcAtmosphere(modelSetting,params,updateOnlyDppm=False,updateTp=False,disp2terminal=False,returnOpac=False,
        #           thermalReflCombined=None,thermalOnly=None,albedoOnly=None, low_res_mode=False)
        #extinctCoef=atm.extinctCoef.copy()
        #scatCoef=atm.scatCoef.copy()   
        
       ### atm.qmol_lay                                                                               = atm.calcComposition(modelSetting,params,levels, firstIter=False)
        #if atm.verbose: print('Going to executeOption(), calcHydroEqui and calcOpacities in calcAtmosphere()\n')
        #atm.executeOption()
      ###  atm.z,atm.dz,atm.grav,atm.ntot,atm.nmol,atm.MuAve,atm.scaleHeight,atm.RpBase,atm.r = atm.calcHydroEqui(modelSetting,params,levels)
      ###  atm.extinctCoef,atm.absorbCoef, atm.scatCoef                                             = atm.calcOpacities(modelSetting,params,levels,saveOpac=atm.saveOpac, low_res_mode=False) 
        
        extinctCoef=atm.extinctCoef.copy()
        scatCoef=atm.scatCoef.copy()
        
        count+=1
        print(count)         
    
    print('Temperature Profile Converged')
    
    return levels  

def take_step_tp_trad(atm,TList,in_levels,N,extinctCoef,scatCoef,loop=False): #*m
    #plotTp(forceLabelAx=True)
    #plotTpChanges(TList)
    toon= atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,in_levels)

    if atm.plotTpChangesEveryIteration:
        atm.plotTpChanges(save=True,close=True,loop=loop)
    
    flux0=toon[0]-toon[1] 
    delflux=-1*flux0
    
    A=np.zeros((N,N),dtype=atm.numerical_precision)

    for level_i in range(N):
        print('*****')
        print(level_i)
        print('*****')
        deltaT=0.001
        levels_ptb=in_levels.copy()
        levels_ptb[level_i]=in_levels[level_i]+deltaT
        flux_ptb_all=atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,levels_ptb)
    
        flux_ptb=flux_ptb_all[0]-flux_ptb_all[1]        
        #flux_ptb_up=flux_ptb_all[0]
        #flux_ptb_down=flux_ptb_all[1]
        A_level_i=(flux_ptb-flux0)/deltaT
        for layer_i in range(N):
            A[layer_i][level_i]=np.trapz(x=atm.wave,y=A_level_i[layer_i])        

    deltafluxsum=np.zeros(N)
    for i in range(N):
        deltafluxsum[i] = np.trapz(x=atm.wave,y=delflux[i])
    delta_levels_lin = np.linalg.lstsq(A, sigmaSB*atm.params['Tint']**4+deltafluxsum, rcond=None)[0]
    delta_levels_reduced=delta_levels_lin*0.10
    maxabs=np.max(np.abs(delta_levels_reduced))
    if maxabs>250:
        delta_levels_reduced=delta_levels_reduced*250/maxabs
    new_levels=in_levels+delta_levels_reduced   

    return delta_levels_reduced,new_levels
    
T = calcNonGrayTpTradProf(atm,modelSetting,params,firstIter,LucyUnsold,runConvection)