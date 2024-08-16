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


if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240630_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')

extinctCoef=atm.opac['extinctCoef']
scatCoef=atm.opac['scatCoef']

toon=atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,atm.T,ref=False)
emiss=atm.calcEmissionSpectrum(extinctCoef)
plt.plot(atm.wave,toon[0][0],label='toon top layer')
plt.xlabel('Wavelength (um)')
plt.ylabel('Spectral Power Density (W/m^2/um)')
plt.xscale('log')
plt.plot (atm.wave,emiss*np.pi,linestyle='dotted',label='emiss*pi',alpha=0.7)
plt.xscale('log')
plt.legend()
plt.savefig('compare_noref.png')

