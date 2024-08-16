import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')

T=np.ones(60)*5000
def tp_trad(T,do_scat=True,negli_flux=0):
    dz0=atm.dz
    dz=np.zeros(len(T))
    dz[0]=dz0[0]
    for i in range(1,len(T)-1):
        dz[i]=(dz0[i]+dz0[i-1])/2
    dz[len(T)-1]=dz0[len(dz0)-1]
    f_down_top=atm.IrradStarEff
    extinctCoef=atm.opac['extinctCoef']
    l=1/extinctCoef #mean free path
    absorbCoef=atm.opac['absorbCoef'] 
    scatCoef=atm.opac['scatCoef'] 
    B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,atm.nWave)),np.tile(atm.wave[np.newaxis,:],(len(T),1)))
    emiss=absorbCoef*B
    print(scatCoef)
    
    N=len(T)
    levels=T
    layers=np.zeros((N,len(atm.wave)))
    
    Istar=f_down_top
    for layer_i in range(N):
        print('-')
        print(layer_i)
        print('-')
        flux=np.zeros(len(atm.wave))
        
        #radiation from star
        flux=flux-Istar
        Istar=Istar*np.e**(-dz[layer_i]/l[layer_i])
        flux_star_scat=0
        
        for level_i in range(N):
            print('.')
            flux_added=0
            
            #emission from level_i that is transmitted to layer_i without being scattered
            flux_added_emiss=0
            I0=emiss[level_i]
            cur_lev=level_i
            if level_i<layer_i:
                I=-1*I0
                while(cur_lev<layer_i):
                    I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                    cur_lev+=1
            else:
                I=I0
                while(cur_lev>=layer_i):
                    I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                    cur_lev-=1
            flux_added_emiss=I
            
            #emission from level_i that is scattered off another level before reaching layer_i
            #multiple scattering is not considered
            flux_added_scat=0
            if do_scat==True:
                #starlight scattered from level_i up to layer_i
                flux_star_scat_added=0
                if level_i>=layer_i:
                    cur_lev=0
                    I=f_down_top
                    while(cur_lev<level_i and I.any()>negli_flux):
                        I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                        cur_lev+=1
                    I=I*0.5*scatCoef[level_i]
                    cur_lev-=1
                    while(cur_lev>=layer_i and I.any()>negli_flux):
                        I=I*np.e**(-dz[cur_lev]/l[cur_lev])
                        cur_lev-=1
                    if(I.any()>negli_flux):
                        flux_star_scat_added=I 
                flux_star_scat=flux_star_scat+flux_star_scat_added     
                
                #scattered off lower level and reflected back upwards
                scat_up=0
                if level_i<layer_i:
                    cur_scattering_lev=layer_i
                else: 
                    cur_scattering_lev=level_i
                while(cur_scattering_lev<N):
                    I=I0
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
        layers[layer_i]=flux
    return layers
                    
                
                
                    

                

            
    

    #print(extinctCoef-absorbCoef-scatCoef)
    #print(np.shape(extinctCoef))
    #print(np.shape(extinctCoef[0]))
x=tp_trad(T,do_scat=True,negli_flux=10**(-5))
pdb.set_trace() 
