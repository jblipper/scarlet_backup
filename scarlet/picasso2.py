from picaso import fluxes,climate

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


#def climate( pressure, temperature, dwni,  bb , y2, tp, tmin, tmax ,DTAU, TAU, W0, 
#            COSB,ftau_cld, ftau_ray,GCOS2, DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, 
#            ubar0,ubar1,cos_theta, FOPI, single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
#            wno,nwno,ng,nt, nlevel, ngauss, gauss_wts,reflected, thermal):

flux_net_v_layer, flux_net_v, flux_plus_v, flux_minus_v , flux_net_ir_layer, flux_net_ir, flux_plus_ir, flux_minus_ir=climate.climate( 
atm.p, atm.T, dwni,  bb , y2, tp, tmin, tmax ,DTAU, TAU, W0, COSB,ftau_cld, ftau_ray,GCOS2, 
DTAU_OG, TAU_OG, W0_OG, COSB_OG, W0_no_raman , surf_reflect, ubar0,ubar1,cos_theta, FOPI, 
single_phase,multi_phase,frac_a,frac_b,frac_c,constant_back,constant_forward, tridiagonal , 
wno,nwno,ng,nt, nlevel, ngauss, gauss_wts,reflected, thermal)
    
"""
Program to run RT for climate calculations. Runs the thermal and reflected module.
And combines the results with wavenumber widths.



    
Parameters 
----------
pressure : array 
     Level Pressure  Array
temperature : array
    Opacity class from `justdoit.opannection`
dwni : array 
    IR wavenumber intervals.
bb : array 
    BB flux array. output from set_bb
y2 : array
    output from set_bb
tp : array
    output from set_bb
tmin : float
    Minimum temp upto which interpolation has been done.
tmax : float
    Maximum temp upto which interpolation has been done.
    

reflected : bool 
    Run reflected light
thermal : bool 
    Run thermal emission

        
Return
------
array
    Visible and IR -- net (layer and level), upward (level) and downward (level)  fluxes
"""
    
