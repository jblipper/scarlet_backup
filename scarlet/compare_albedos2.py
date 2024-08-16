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
    
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240709_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')    
        
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240711_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/gitHub/scarlet_results/FwdRuns20240709_0.3_100.0_64_nLay60/TOI_270_d/TOI_270_d_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint20.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240711_0.3_100.0_64_nLay60/TRAPPIST_1_f/TRAPPIST_1_f_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTradTint20.0f0.25A0.1_pCloud100000.0mbar.atm')

fig,axs=plt.subplots(2,gridspec_kw={'height_ratios': [2,1]})

atm.plotSpectrum(ax=axs[0],spectype='totalFluxToon',resPower=200,label='toon')

atm.plotSpectrum(ax=axs[0],spectype='totalFluxFeautrier',resPower=200,label='feautrier')

axs[0].set_ylabel(r'Total Emiss. TOA $\left[\mathrm{W/m^{2}\mu m}\right]$')

totalFluxFeautrier0=deepcopy(atm.totalFluxFeautrier)

atm.totalFluxFeautrier=atm.totalFluxToon-atm.totalFluxFeautrier


for ax in axs:
    ax.label_outer()

atm.plotSpectrum(ax=axs[1],spectype='totalFluxFeautrier',resPower=200,label='residuals')

axs[1].set_ylabel('Residuals')
#plt.plot(atm.wave,atm.totalFluxFeautrier,label='feautrier')
#plt.xlabel('Wavelength (um)')
#plt.ylabel('Albedo')
#plt.xscale('log')
#plt.plot (atm.wave,Fupw[0],linestyle='dotted',label='toon',alpha=0.7)
#plt.xscale('log')
#plt.legend()
plt.savefig('compare_albedos.png')

#plt.clf()

#atm.totalFluxToon=get_fluxes_toon(atm,atm.IrradStar,extinctCoef,scatCoef,therm=False,ref=True,asym=0.5)[0][0]

#atm.totalFluxFeautrier=totalFluxFeautrier0-atm.thermalFeautrier

#atm.totalFluxDisort=atm.totalFluxToon/atm.totalFluxFeautrier

#atm.plotSpectrum(spectype='totalFluxDisort',resPower=200,label='ratio')
#plt.plot(atm.wave,atm.totalFluxToon/atm.totalFluxFeautrier,label='feautrier')
#plt.ylim(-10,10)
#plt.plot(atm.wave,atm.totalFluxToon,linestyle='dotted',label='toon',alpha=0.7)
#plt.xscale('log')
#plt.yscale('log')
#plt.legend()
#plt.plot(atm.wave,atm.totalFluxFeautrier)


#plt.savefig('compare_albedos_ratio.png')



