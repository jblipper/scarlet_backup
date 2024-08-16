import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
import matplotlib.pyplot as plt
from auxbenneke.constants import unitfac, pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
#print(atm.p)
#print(atm.wave)
#print(dir(atm))

atm.T=np.ones(60)*500


extinctCoef=atm.opac['extinctCoef']
#x=thermalemission(shave==nwave)
thermal_standard=atm.calcEmissionSpectrum(extinctCoef)
#print(thermal_standard)
result_calcEmissionSpectrum=np.trapz(y=thermal_standard*np.pi,x=atm.wave)
print(f'calcEmissionSpectrum: {result_calcEmissionSpectrum}')


atm.extinctCoef=atm.opac['extinctCoef'].copy()
atm.scatCoef=atm.opac['scatCoef'].copy()
#extinctCoef = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:])
#scatCoef = 0.5*(scatCoef[:atm.nLay-1,:]+scatCoef[1:atm.nLay,:]) 
extinctCoef = 0.5*(atm.extinctCoef[:atm.nLay-1,:]+atm.extinctCoef[1:atm.nLay,:])
scatCoef = 0.5*(atm.scatCoef[:atm.nLay-1,:]+atm.scatCoef[1:atm.nLay,:])
#toon= atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = False,thermal = True, calcJ = False, intTopLay=0,w=0)
toon= atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = False,thermal = True,w=0)
#toon_int=rad.convertIntensity(toon,atm.wave,'W/(m**2*um)','um','W/(m**2*Hz)') / np.pi

result_toon=np.trapz(x=atm.wave,y=toon[0][0])
print(f'toon: {result_toon}')
#print(np.shape(toon[0]))
#print(toon_int)
#print(f'toon: {np.trapz(x=atm.wave,y=toon_int[0][0])}')

print(result_toon/result_calcEmissionSpectrum)

# plt.plot(atm.wave,toon[0][0])
# plt.plot(atm.wave,atm.thermal)

#plt.plot(atm.wave[8000:], np.pi*thermal_standard[8000:]/toon[0][0][8000:])

plt.plot(atm.wave, np.pi*thermal_standard/toon[0][0])
plt.ylim(0,10)
plt.xlabel('Wavelength')
plt.ylabel(r'$\frac{π*intensity_{calcEmissionSpectrum}}{intensity_{toon}}$')
#plt.plot(atm.wave, toon[0][0])

print(atm.wave)
print((result_calcEmissionSpectrum/toon[0][0])[10000])
#plt.ylim(-1,5)
#plt.xlim(0,5)
plt.savefig('toontest.png')

plt.clf()
plt.plot(atm.wave,toon[0][0])
plt.xscale('log')
plt.savefig('testnew.png')
plt.clf()






flux_up=toon[0]
flux_down=toon[1]


plt.plot(atm.wave,flux_up[0])
plt.title('Upwards Flux Top Layer')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(-1,500)
plt.savefig('toon_old/clim_toon_up_top.png')
plt.clf()

plt.plot(atm.wave,flux_down[0])
plt.title('Downwards Flux Top Layer')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(-1,4500)
plt.savefig('toon_old/clim_toon_down_top.png')
plt.clf()

plt.plot(atm.wave,flux_up[30])
plt.title('Upwards Flux Layer 30')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(-1,500)
plt.savefig('toon_old/clim_toon_up_mid.png')
plt.clf()

plt.plot(atm.wave,flux_down[30])
plt.title('Downwards Layer 30')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(-1,5000)
plt.savefig('toon_old/clim_toon_down_mid.png')
plt.clf()

plt.plot(atm.wave,flux_up[0]-flux_up[30])
plt.title('flux_up[0]-flux_up[30]')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(-200,300)
plt.savefig('toon_old/clim_toon_up_test.png')
plt.clf()

print(flux_up[0])
result_toon=np.trapz(x=atm.wave,y=flux_up[0])
print(result_toon) 

plt.plot(atm.wave,thermal_standard*2*np.pi)
plt.xlabel('Wavelength (um)')
plt.ylabel('Intensity * 2pi (W/(m^2*s*str))')
plt.title('calcEmissionSpectrum*2pi')
#plt.xscale('log')
plt.savefig('thermal_standard_lin.png')
plt.clf()