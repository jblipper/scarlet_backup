import pickle

import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')

T=np.ones(60)*5000
#T=np.array([ 1.77705462e+05, -3.54916625e+05, -1.49773140e+06, -3.75374431e+06,-8.01585268e+06, -1.55647184e+07, -2.58519628e+07, -1.79441072e+07,-1.99180266e+07, -2.10320078e+07, -2.22210297e+07, -2.35205869e+07,-2.49470340e+07, -2.64471817e+07, -2.86939881e+07, -3.10412549e+07,-3.34704627e+07, -3.57640188e+07, -3.81382279e+07, -4.13040850e+07,-4.34702936e+07, -4.32965084e+07, -4.15101782e+07, -3.62437233e+07,-2.91650108e+07, -2.14487072e+07, -1.43691073e+07, -8.89543618e+06,-5.15418976e+06, -2.84872796e+06, -1.49378615e+06, -7.44867920e+05,-3.64711696e+05, -1.69525010e+05, -7.21813088e+04, -2.89619026e+04,-9.17826942e+03, -2.44037612e+02,  3.47733547e+03,  5.01293760e+03,5.64129045e+03,  5.89084996e+03,  5.98535471e+03,  6.02135867e+03,6.03352028e+03,  6.03535983e+03,  6.03402545e+03,  6.03304804e+03,6.03215135e+03,  6.03172047e+03,  6.03155317e+03,  6.03145118e+03,6.03136859e+03,  6.03133279e+03,  6.03134648e+03,  6.03137525e+03,6.03141629e+03,  6.03152507e+03,  6.03190857e+03,  6.03289385e+03])
T=np.array([702.9479696168794,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
def tp_trad(T,do_scat=True,negli_flux=0):
    dz0=-1*atm.dz.copy()
    dz0= -np.diff(atm.z) 
    dz=np.zeros(len(T))
    dz[0]=dz0[0]
    for i in range(1,len(T)-1):
        dz[i]=(dz0[i]+dz0[i-1])/2
    dz[len(T)-1]=dz0[len(dz0)-1]
    f_down_top=atm.IrradStarEff.copy()
    extinctCoef=atm.opac['extinctCoef'].copy()
    l=1/extinctCoef #mean free path
    
    N=len(T)
    
    dtau = 0.5*(extinctCoef[:N-1,:]+extinctCoef[1:N,:]) * np.outer(dz0,np.ones(atm.nWave, dtype=atm.numerical_precision))  
    
    tau=np.vstack([np.zeros(atm.nWave, dtype=atm.numerical_precision),np.cumsum(dtau,axis=0)])
    # Ensure taugrid > 0
    tau*=(1+(np.arange(1,(N+1))[:,np.newaxis])*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-10)+(np.arange(1,(N+1))[:,np.newaxis]*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-99)
    muObs=np.array([0.5773502691896258])
    for i in range(len(muObs)):         # Loop through all muObs    
        transmis=np.exp((-tau)/muObs[i])      # e^(-Tv/u) term
    pdb.set_trace()
    
    absorbCoef=atm.opac['absorbCoef'].copy()
    scatCoef=atm.opac['scatCoef'].copy() 
    print(scatCoef)
    
    N=len(T)
    levels=T.copy()
    
    #----------------------------------------------------------------------------#
    #--Set up tabulation for cumulative transmissions (through multiple layers)--#
    #----------------------------------------------------------------------------# 
    cum_trans=np.ones((N,N,len(atm.wave)))*-1
    def get_cum_trans(shallow_level,deep_level):
        if shallow_level>deep_level:
            out=np.ones(len(atm.wave))
            return out
        elif cum_trans[shallow_level][deep_level][0]!=-1:
            return cum_trans[shallow_level][deep_level]
        elif shallow_level==deep_level:
            out=transmis[shallow_level].copy()
            cum_trans[shallow_level][deep_level]=out.copy()
            print("SJHSNJKNS")
            return out
        else:
            out=transmis[deep_level]*get_cum_trans(shallow_level,deep_level-1)
            cum_trans[shallow_level][deep_level]=out.copy()
            print("SJHSNJKNS")
            return out
    
    
    
    
    
    
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
        Istar=Istar*transmis[layer_i]
        # e^(-Tv/u)
        flux_star_scat_layer_i=0        
        if do_scat==True:
            for level_i in range(N):
                print('.')
                #starlight scattered from level_i up to layer_i
                flux_star_scat_layer_i_added=0
                if level_i>=layer_i:
                    I=f_down_top.copy()*get_cum_trans(0,level_i-1)
                    I=I*0.5*scatCoef[level_i]
                    I=I*get_cum_trans(layer_i,level_i-1)
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
                    I=I*get_cum_trans(level_i+1,layer_i-1)
                else:
                    I=I0.copy()
                    I=I*get_cum_trans(layer_i,level_i-1)
                flux_added_emiss=I.copy()
                
                #emission from level_i that is scattered off another level before reaching layer_i
                #multiple scattering is not considered
                flux_added_scat=0
                if do_scat==True:
                    
                    #scattered off lower level and reflected back upwards
                    scat_up=0
                    if level_i+1<layer_i:
                        cur_scattering_lev=layer_i
                    else: 
                        cur_scattering_lev=level_i+1
                    while(cur_scattering_lev<N):
                        I=I0.copy()
                        I=I*get_cum_trans(level_i+1,cur_scattering_lev-1)
                        I=I*0.5*scatCoef[cur_scattering_lev]
                        I=I*get_cum_trans(layer_i,cur_scattering_lev-1)
                        scat_up=scat_up+I 
                        cur_scattering_lev+=1  
                    #scattered off higher level and reflected back downwards     
                    scat_down=0
                    if level_i-1>=layer_i:
                        cur_scattering_lev=layer_i-1
                    else: 
                        cur_scattering_lev=level_i-1
                    while(cur_scattering_lev>=0):
                        I=-1*I0
                        I=I*get_cum_trans(cur_scattering_lev+1,level_i-1)
                        I=I*0.5*scatCoef[cur_scattering_lev]
                        I=I*get_cum_trans(cur_scattering_lev+1,layer_i-1)
                        scat_down=scat_down+I 
                        cur_scattering_lev-=1 
                    flux_added_scat=scat_up+scat_down
            
                flux_added=flux_added_emiss+flux_added_scat
                flux=flux+flux_added    
            layers[layer_i]=flux.copy()
        #pdb.set_trace()
        return layers
    
    #flux_star=np.zeros((N,len(atm.wave)))
    flux0=get_flux(T,flux_star)  
    pdb.set_trace()               
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
    A_regularized = A #+ tolerance * np.eye(A.shape[0])
    deltafluxsum=np.zeros(N)
    for i in range(N):
        deltafluxsum[i] = np.trapz(x=atm.wave,y=delflux[i])
    #delta_levels=np.linalg.solve(A_regularized, deltafluxsum)
    delta_levels = np.linalg.lstsq(A, deltafluxsum, rcond=None)[0]
    
    data = {'A': A, 'delflux': delflux, 'levels':levels, 'delfluxsum': deltafluxsum, 'delta_levels': delta_levels}
    with open('data.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    #delta_levels=np.linalg.solve(A,delflux)
    levels=levels+delta_levels
    pdb.set_trace() 
    #tp_trad(levels,do_scat=True,negli_flux=0)
    return levels               

#------------------------------------------                
#total = np.trapz(x=self.wave,y=FupAtEachWave)
#------------------------------------------
            
    

    #print(extinctCoef-absorbCoef-scatCoef)
    #print(np.shape(extinctCoef))
    #print(np.shape(extinctCoef[0]))
x=tp_trad(T,do_scat=True,negli_flux=5*10**(-1))
pdb.set_trace() 
