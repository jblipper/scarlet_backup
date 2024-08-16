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


if __name__ == "__main__": 
        
    #planet='HD_209458_b'
    #atm1 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240621_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm2 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240703_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm3 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240711_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')

    #planet='TOI_270_d'
    #atm1 = scarlet.loadAtm('does not converge')
    #atm2 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240702_0.3_100.0_64_nLay60/TOI_270_d/TOI_270_d_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint20.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm3 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240709_0.3_100.0_64_nLay60/TOI_270_d/TOI_270_d_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint20.0f0.25A0.1_pCloud100000.0mbar.atm')    
    
    planet='Trappist_1_f'
    #atm1 = scarlet.loadAtm('does not converge')
    atm2 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240703_0.3_100.0_64_nLay60/TRAPPIST_1_f/TRAPPIST_1_f_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint10.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm3 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240712_0.3_100.0_64_nLay60/TRAPPIST_1_f/TRAPPIST_1_f_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint10.0f0.25A0.1_pCloud100000.0mbar.atm') 

fig,ax=plt.subplots(1)

#atm1.plotTp(ax=ax,forceLabelAx=True,label='nongray',color='orange')
atm2.plotTp(ax=ax,forceLabelAx=True,label='trad_uncorrected_asym',color='blue')
atm3.plotTp(ax=ax,forceLabelAx=True,label='corrected_asym',color='limegreen')
plt.legend()

plt.savefig(f'{planet}_combo_tp.png')




