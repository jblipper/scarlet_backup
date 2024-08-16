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
    #planet='TOI_270_d'
    atm1 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240725_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayConvTint180.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm2 = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240725_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTint180.0f0.25A0.1_pCloud100000.0mbar.atm')
#width=140
#height=100
#plt.figure(figsize=(width, height))

plt.figure(figsize=(9, 6))

plt.yscale('log')

#plt.xlim(200,700)
#plt.ylim(1e0,1e6)
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (Pa)')
plt.plot(atm1.T,atm1.p/1e5,label='NonGrayConv')
plt.plot(atm2.T,atm2.p/1e5,label='NonGray')
#*101325/1e5
plt.gca().invert_yaxis()
plt.legend()

plt.savefig('convcompare.png')