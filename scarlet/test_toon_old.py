import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
import matplotlib.pyplot as plt
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


extinctCoef=atm.opac['extinctCoef'].copy()
scatCoef=atm.opac['scatCoef'].copy()
extinctCoef = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:])
scatCoef = 0.5*(scatCoef[:atm.nLay-1,:]+scatCoef[1:atm.nLay,:]) 
toon= atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = False,thermal = True, calcJ = False, intTopLay=0)
#toon_int=rad.convertIntensity(toon,atm.wave,'W/(m**2*um)','um','W/(m**2*Hz)') / np.pi

result_toon=np.trapz(x=atm.wave,y=toon[0][0])
print(f'toon: {result_toon}')
#print(np.shape(toon[0]))
#print(toon_int)
#print(f'toon: {np.trapz(x=atm.wave,y=toon_int[0][0])}')

print(result_toon/result_calcEmissionSpectrum)

# plt.plot(atm.wave,toon[0][0])
# plt.plot(atm.wave,atm.thermal)

plt.plot(atm.wave, np.pi*thermal_standard/toon[0][0])

print(atm.wave)
plt.ylim(-1,5)
#plt.xlim(0,5)
plt.savefig('toontest.png')