import pickle
from auxbenneke.constants import unitfac

import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
from scipy.integrate import RK45
from matplotlib import pyplot as plt
from toonew_funcs import get_fluxes_toon
from copy import deepcopy






import numexpr as ne
import pdb
import scarlet
from scarlet import radutils as rad
from numba import jit, vectorize
import numpy as np
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10
import matplotlib.pyplot as plt
from auxbenneke.constants import unitfac, pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom

if __name__ == "__main__":
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTint100.0f0.25A0.1_pCloud100000.0mbar.atm')

def calcEnergy(T):
    z,dz,grav,ntot,nmol,MuAve,scaleHeight,RpBase,r=calcHydroEqui(T)
    Cp=14300 #[J*kg^-1*K^-1] (H2)
    #MuAve=uAtom*np.sum(atm.MolarMass[np.newaxis,:]*atm.qmol_lay,axis=1)  #[kg]
    MuAve = 0.5*(MuAve[:atm.nLay-1]+MuAve[1:atm.nLay]) #[kg]
    #ntot=atm.p/(kBoltz*T) # [m^-3]
    ntot = 0.5*(ntot[:atm.nLay-1]+ntot[1:atm.nLay]) # [m^-3]
    Mcp=dz*ntot*MuAve*Cp #[J*K^-1*m^-2]
    Q=Mcp*0.5*(T[:atm.nLay-1]+T[1:atm.nLay]) #[J*m^-2]
    #print(Q)
    E=np.sum(Q)
    return E

def calcHydroEqui(T):

        #Note: All values without "grid" in variable name are at cell centers
        
        #array of molar masses (atm.MolarMass) from readMoleculesProperties() above
        MuAve=uAtom*np.sum(atm.MolarMass[np.newaxis,:]*atm.qmol_lay,axis=1)  #[kg]
        
        #don't need to define p since already have atm.p
        ntot=atm.p/(kBoltz*T) # [m^-3]
        
        nmol=((ntot[:,np.newaxis]).dot(np.ones([1,atm.nMol], dtype=atm.numerical_precision)))*atm.qmol_lay #(iLev) in [m^-3]
        
        r               =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        grav            =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        scaleHeight     =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        dz              =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        
        r[atm.iLevRpRef]=deepcopy(atm.Rp)
        
        #Atmosphere below r[atm.iLevRpRef]=atm.Rp to higher pressures:
        for iLay in range(atm.iLevRpRef,atm.nLay-1):
                for repeat in range(1,5):
                    grav[iLay]        = G*atm.Mp / ( r[iLay] - 0.5*dz[iLay] )**2                                     # [m/s^2]
                    scaleHeight[iLay] = (kBoltz*T[iLay])/(MuAve[iLay]*grav[iLay])                 # [m]
                    dz[iLay]          = -scaleHeight[iLay]*np.log(atm.p[iLay]/atm.p[iLay+1]) # [m]
                r[iLay+1]     = r[iLay] - dz[iLay]
        #Atmosphere above r[atm.iLevRpRef]=atm.Rp to lower pressures:
        for iLay in range(atm.iLevRpRef,0,-1):
                for repeat in range(1,5):
                    grav[iLay-1]        = G*atm.Mp / ( r[iLay] + 0.5*dz[iLay-1] )**2                                     # [m/s^2]
                    scaleHeight[iLay-1] = (kBoltz*T[iLay-1])/(MuAve[iLay-1]*grav[iLay-1])         # [m]
                    dz[iLay-1]          = -scaleHeight[iLay-1]*np.log(atm.p[iLay-1]/atm.p[iLay]) # [m]
                r[iLay-1]     = r[iLay] + dz[iLay-1]
                
                   
  

        ## --> at this point it has determined r completely (everything else can now be derived from that)
        
        RpBase=r[-1]
        
        z=r-r[-1]         # [m]
        dz=np.zeros(atm.nLay-1, dtype=atm.numerical_precision)   # [m]
        dz=-np.diff(z)            # [m]   
        
        grav=G*atm.Mp /  r**2        # [m/s^2]
        scaleHeight=(kBoltz*T)/(MuAve*grav) # [m]
        
        #consistency check
#        scaleHeightLay = 0.5*(scaleHeight[:atm.nLay-1]+scaleHeight[1:atm.nLay])
#        dz2=-scaleHeightLay*np.log(atm.p[:-1]/atm.p[1:]) # [m]
#        print (dz2-dz)/dz*100

        return z,dz,grav,ntot,nmol,MuAve,scaleHeight,RpBase,r
        
#T=np.array(ones)*2000
T=np.array([ 801.04726038,  802.06800173,  805.98237113,  810.46787058,  818.47038155,  827.43164902,  839.73617398,  852.45641686,  867.41307381,  880.75136987,  893.33864209,  902.22252756,  910.80014821,  918.48811073,  928.69236472,  938.88333716,  951.69035626,  964.57268834,  979.87861396,  995.35824736, 1013.55341028, 1031.98709707, 1052.87973707, 1071.8743683,  1091.3655835, 1112.35532581, 1135.21706267, 1158.02967468, 1185.67665515, 1217.44437484, 1256.03165636, 1301.02118829, 1354.51825581, 1416.32259537, 1485.65744154, 1555.01349023, 1624.90291458, 1692.75385532, 1756.65506873, 1800.78171282, 1834.85726405, 1859.19141734, 1873.35973387, 1878.08347106, 1878.46252075, 1878.69783238, 1879.21135397, 1880.30695016, 1882.63383997, 1887.62244917, 1897.78171099, 1917.68511478, 1955.63669003, 2024.40178139, 2138.58925498, 2308.09571828, 2530.19339384, 2786.7255833,  3052.65723082, 3313.30311644])

Ts=[]
Es=[]
for i in range(500):
    T[45]-=2
    T[46]-=2
    #T[:]-=2
    Ts.append(T[45])
    Es.append(calcEnergy(T)) 




# Convert lists to numpy arrays for fitting
Ts_np = np.array(Ts)
Es_np = np.array(Es)

# Perform 4th-degree polynomial regression to find coefficients for the fit
coefficients = np.polyfit(Ts_np, Es_np, 4)
fourth_degree_fit = (coefficients[0] * Ts_np**4 +
                     coefficients[1] * Ts_np**3 +
                     coefficients[2] * Ts_np**2 +
                     coefficients[3] * Ts_np +
                     coefficients[4])

# Calculate residuals
residuals = Es_np - fourth_degree_fit

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Main plot: Data points and 4th-degree polynomial fit
ax1.plot(Ts, Es, '.', label='Data Points')
ax1.plot(Ts_np, fourth_degree_fit, '-', color='red', 
         label=('4th-Degree Fit: y = {:.2e}x^4 + {:.2e}x^3 + {:.2e}x^2 + {:.2e}x + {:.2e}'
                .format(*coefficients)))
ax1.set_xlabel('T')
ax1.set_ylabel('E')
ax1.legend()
ax1.set_title('4th-Degree Polynomial Fit')

# Residual plot
ax2.plot(Ts_np, residuals, '.', color='blue', label='Residuals')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel('T')
ax2.set_ylabel('Residuals')
ax2.legend()
ax2.set_title('Residual Plot')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('TTest_with_residuals.png')
plt.show()

### Quadratic

# Ts_np = np.array(Ts)
# Es_np = np.array(Es)
# 
# # Perform quadratic regression to find coefficients for the fit
# coefficients = np.polyfit(Ts_np, Es_np, 2)
# quadratic_fit = coefficients[0] * Ts_np**2 + coefficients[1] * Ts_np + coefficients[2]
# 
# # Calculate residuals
# residuals = Es_np - quadratic_fit
# 
# # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
# 
# # Main plot: Data points and quadratic fit
# ax1.plot(Ts, Es, '.', label='Data Points')
# ax1.plot(Ts_np, quadratic_fit, '-', color='red', label=f'Quadratic Fit: y = {coefficients[0]:.2e}x² + {coefficients[1]:.2e}x + {coefficients[2]:.2e}')
# ax1.set_xlabel('T')
# ax1.set_ylabel('E')
# ax1.legend()
# ax1.set_title('Quadratic Fit')
# 
# # Residual plot
# ax2.plot(Ts_np, residuals, '.', color='blue', label='Residuals')
# ax2.axhline(0, color='black', linewidth=0.8)
# ax2.set_xlabel('T')
# ax2.set_ylabel('Residuals')
# ax2.legend()
# ax2.set_title('Residual Plot')
# 
# # Adjust layout and save the plot
# plt.tight_layout()
# plt.savefig('TTest_with_residuals.png')
# plt.show()

###Linear


#Ts_np = np.array(Ts)
#Es_np = np.array(Es)

# Perform linear regression to find slope and intercept
#slope, intercept = np.polyfit(Ts_np, Es_np, 1)
#fit_line = slope * Ts_np + intercept

# Plot the data points
#plt.plot(Ts, Es, '.', label='Data Points')

# Plot the linear fit
#plt.plot(Ts_np, fit_line, '-', color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')

# Label the axes
#plt.xlabel('T')
#plt.ylabel('E')

# Add a legend
#plt.legend()

# Save and show the plot
#plt.savefig('TTest.png')
#plt.show()

#plt.plot(Ts,Es,'.')
#plt.xlabel('T')
#plt.ylabel('E')
#plt.savefig('TTest.png')

 
print(Es)
        