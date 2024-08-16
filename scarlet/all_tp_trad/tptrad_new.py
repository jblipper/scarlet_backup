import pickle

import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')

T=np.ones(60)*5000
def tp_trad(T,do_scat=True,negli_flux=0):
    dz0=atm.dz.copy()
    dz=np.zeros(len(T))
    dz[0]=dz0[0]
    for i in range(1,len(T)-1):
        dz[i]=(dz0[i]+dz0[i-1])/2
    dz[len(T)-1]=dz0[len(dz0)-1]
    f_down_top=atm.IrradStarEff.copy()
    extinctCoef=atm.opac['extinctCoef'].copy()
    l=1/extinctCoef #mean free path
    absorbCoef=atm.opac['absorbCoef'].copy()
    scatCoef=atm.opac['scatCoef'].copy() 
    print(scatCoef)
    
    N=len(T)
    levels=T.copy()
    
    
    #------------------------------------------------------------------------#
    #-----Calculate flux at each layer using initial temperature profile-----#
    #------------------------------------------------------------------------#
    
    flux_star=np.zeros((N,len(atm.wave)))
    Istar=f_down_top.copy()
    for layer_i in range(N):
        print('-')
        print(layer_i)
        print('-')
        
        #radiation from star
        flux_star[layer_i]=-Istar.copy()
        Istar=Istar*np.e**(-dz[layer_i]/l[layer_i])
        
        flux_star_scat_layer_i=0        
        if do_scat==True:
            for level_i in range(N):
                print('.')
                #starlight scattered from level_i up to layer_i
                flux_star_scat_layer_i_added=0
                if level_i>=layer_i:
                    cur_lev=0
                    I=f_down_top.copy()
                    while(cur_lev<level_i and I.any()>negli_flux):
                        I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                        cur_lev+=1
                    I=I*0.5*scatCoef[level_i]
                    cur_lev-=1
                    while(cur_lev>=layer_i and I.any()>negli_flux):
                        I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                        cur_lev-=1
                    if(I.any()>negli_flux):
                        flux_star_scat_layer_i_added=I.copy() 
                flux_star_scat_layer_i=flux_star_scat_layer_i+flux_star_scat_layer_i_added
        flux_star[layer_i]=flux_star[layer_i]+flux_star_scat_layer_i
            
    
    
    def get_flux(currlevels,flux_star):
        layers=np.zeros((N,len(atm.wave)))
        T=currlevels.copy() 
        B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,atm.nWave)),np.tile(atm.wave[np.newaxis,:],(len(T),1)))
        emiss=absorbCoef*B
    
        for layer_i in range(N):
            print('-')
            print(layer_i)
            print('-')
            flux=flux_star[layer_i].copy() 
        
            for level_i in range(N):
                print('.')
                flux_added=0
                
                #emission from level_i that is transmitted to layer_i without being scattered
                flux_added_emiss=0
                I0=emiss[level_i].copy() 
                cur_lev=level_i
                if level_i<layer_i:
                    I=-1*I0
                    while(cur_lev<layer_i):
                        I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                        cur_lev+=1
                else:
                    I=I0.copy()
                    while(cur_lev>=layer_i):
                        I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                        cur_lev-=1
                flux_added_emiss=I.copy()
                
                #emission from level_i that is scattered off another level before reaching layer_i
                #multiple scattering is not considered
                flux_added_scat=0
                if do_scat==True:
                    
                    #scattered off lower level and reflected back upwards
                    scat_up=0
                    if level_i<layer_i:
                        cur_scattering_lev=layer_i
                    else: 
                        cur_scattering_lev=level_i
                    while(cur_scattering_lev<N):
                        I=I0.copy()
                        cur_lev=level_i
                        while(cur_lev<cur_scattering_lev and I.any()>negli_flux):
                            I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                            cur_lev+=1
                        I=I*0.5*scatCoef[cur_scattering_lev]
                        cur_lev-=1
                        while(cur_lev>=layer_i and I.any()>negli_flux):
                            I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                            cur_lev-=1
                        if(I.any()>negli_flux):
                            scat_up=scat_up+I 
                        cur_scattering_lev+=1  
                    #scattered off higher level and reflected back downwards     
                    scat_down=0
                    if level_i>=layer_i:
                        cur_scattering_lev=layer_i-1
                    else: 
                        cur_scattering_lev=level_i
                    while(cur_scattering_lev>=0):
                        I=-1*I0
                        cur_lev=level_i
                        while(cur_lev>cur_scattering_lev and I.any()>negli_flux):
                            I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                            cur_lev=cur_lev-1
                        I=I*0.5*scatCoef[cur_scattering_lev]
                        cur_lev+=1
                        while(cur_lev<layer_i and I.any()>negli_flux):
                            I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                            cur_lev=cur_lev+1
                        if(I.any()>negli_flux):
                            scat_down=scat_down+I 
                        cur_scattering_lev-=1 
                    flux_added_scat=scat_up+scat_down
            
                flux_added=flux_added_emiss+flux_added_scat
                flux=flux+flux_added    
            layers[layer_i]=flux.copy()
        #pdb.set_trace()
        return layers
    
    flux0=get_flux(T,flux_star)                
    #------------------------------------------------------------------------#
    #---------------Calculate temperature perturbations----------------------#
    #------------------------------------------------------------------------#
     
    delflux=-1*flux0
    A=np.zeros((N,N))
    
    for level_i in range(N):
         print('*****')
         print(level_i)
         print('*****')
         deltaT=1
         levels_ptb=levels.copy()
         levels_ptb[level_i]=levels[level_i]+deltaT
         flux_ptb=get_flux(levels_ptb,flux_star)
         A_level_i=(flux_ptb-flux0)/deltaT
         for layer_i in range(N):
             A[layer_i][level_i]=np.trapz(x=atm.wave,y=A_level_i[layer_i])
             
    
    pdb.set_trace()
    tolerance = 1e-6
    A_regularized = A + tolerance * np.eye(A.shape[0])
    deltafluxsum=np.zeros(N)
    for i in range(N):
        deltafluxsum[i] = np.trapz(x=atm.wave,y=delflux[i])
    delta_levels=np.linalg.solve(A_regularized, deltafluxsum)
    
    data = {'A': A, 'delflux': delflux, 'levels':levels, 'delfluxsum': deltafluxsum, 'delta_levels': delta_levels}
    with open('data_new.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    #delta_levels=np.linalg.solve(A,delflux)
    levels=levels+delta_levels
    return levels               

#------------------------------------------                
#total = np.trapz(x=self.wave,y=FupAtEachWave)
#------------------------------------------
            
    

    #print(extinctCoef-absorbCoef-scatCoef)
    #print(np.shape(extinctCoef))
    #print(np.shape(extinctCoef[0]))
x=tp_trad(T,do_scat=False,negli_flux=5*10**(-1))
pdb.set_trace() 
