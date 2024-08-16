import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
#print(atm.p)
#print(atm.wave)
#print(dir(atm))
extinctCoef=atm.opac['extinctCoef']
#x=thermalemission(shave==nwave)
def calcEmissionNew(extinctCoef, muObs=np.array([0.5773502691896258]), FluxUnit='W/(m**2*um)'):
    p=atm.p
    T=atm.T
    wave=atm.wave
    atm.dz = -np.diff(atm.z)
    # We define a new grid (Pressure, Temperature, Altitude) which will
    # stop at the cloud deck as if it was our ground. 
        
    # From the cloud pressure pCloud we interpolate to get the cloud temperature TCloud
    # and the cloud altitude zCloud. To do so, we use linear interpolation defined in utilities
    # but with the pressure in log scale.
        
    # Here is the if statement which defines the different grids depending
    # on the presence of clouds. 
        
    includeGrayCloudDeck=atm.includeGrayCloudDeckThermal

    #---Set the local variable (T, z, lay) for the thermal emission integration---------------
    if includeGrayCloudDeck and atm.modelSetting['CloudTypes'][0]:
        TCloud = ut.interp1dEx(np.log(atm.p),atm.T,np.log(atm.params['pCloud']))
        zCloud = ut.interp1dEx(np.log(atm.p),atm.z,np.log(atm.params['pCloud']))
            
        # We are now ready to create our new grid, which will stop at the cloud. In 
        # this calculation, we only need to define it for T and z. 
        T = np.append(atm.T[0:np.searchsorted(atm.p, atm.params['pCloud'], side='left')], TCloud)
        z = np.append(atm.z[0:np.searchsorted(atm.p, atm.params['pCloud'], side='left')], zCloud)
            
        lay = len(T) # This is the length of our new grid
    else:
        T = atm.T
        z = atm.z
        lay = atm.nLay
            
        
    #---Compute thermal emission spectrum------------------------------------------------------

    # Defn req parameters
    TSurf     = T[-1]                       # Surface temp defn'ed at last entry of T -> T[-1]
    dz   = -np.diff(z)        
    #dtau = extinctCoef * np.tile(dz[:,np.newaxis],(1,atm.nWave))

    #dtau = extinctCoef * np.outer(dz,np.ones(atm.nWave))
    dtau = 0.5*(extinctCoef[:lay-1,:]+extinctCoef[1:lay,:]) * np.outer(dz,np.ones(atm.nWave, dtype=atm.numerical_precision))   
    #if (np.array_equal(dtau,dtau2)):
        #print "Success!"
        #alternatively just type this into python console
        
    # Build grid of cumulative sums of tau from TOA to ground/cloud, starting at 0
    tau=np.vstack([np.zeros(atm.nWave, dtype=atm.numerical_precision),np.cumsum(dtau,axis=0)])
    # Ensure taugrid > 0
    tau*=(1+(np.arange(1,(lay+1))[:,np.newaxis])*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-10)+(np.arange(1,(lay+1))[:,np.newaxis]*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-99)

    if atm.doTransit:

        ## Compute black body radiation for T and TSurf with radutils
        B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,atm.nWave)),np.tile(atm.wave[np.newaxis,:],(len(T),1)),'um',FluxUnit,'rad')
        #B = rad.PlanckFct(T,atm.wave,'um',FluxUnit,'rad')
        Bsurf = rad.PlanckFct(TSurf,atm.wave[np.newaxis,:],'um',FluxUnit,'rad')
    else:
        ## Compute black body radiation for T and TSurf with radutils
        B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,atm.nWave)),np.tile(atm.wave[np.newaxis,:],(len(T),1)))
        #B = rad.PlanckFct(T,atm.wave,'um',FluxUnit,'rad')
        Bsurf = rad.PlanckFct(TSurf,atm.wave[np.newaxis,:])

    ## Calculate thermal emission for each muObs 
    IntensityThermalEmission=np.zeros([len(muObs),atm.nWave], dtype=atm.numerical_precision)   # Create empty array

    pdb.set_trace()
    
    for i in range(len(muObs)):         # Loop through all muObs
        
        transmis=np.exp((-tau)/muObs[i])      # e^(-Tv/u) term
        pdb.set_trace() 
        #------------Calculate second term of integral equation: Integration over optical depth------------
        IntensityThermalEmissionAtm=-np.trapz(y=B,x=transmis,axis = 0)
            
        #------------Calculate surface contribution (first term integral equation)------------
        IntensityThermalEmissionGround=Bsurf*transmis[-1,:]
            
        #------------Sum atmospheric and surface contributions ------------
        IntensityThermalEmission[i,:]=IntensityThermalEmissionAtm+IntensityThermalEmissionGround
            
    # This is the thermal emission for each wavelength    
    thermal=np.squeeze(IntensityThermalEmission)
        
    # Check total thermal flux
    #TotalThermalFlux=np.trapz(IntensityThermalEmission[:,0]*np.pi,atm.wave)
    #AverageReradiationTemp=(TotalThermalFlux/sigmaSB)**(1/4)                 #only true if entire thermal emission is simulated
        
    # -----------------------------------------------------------------------------------------------------------------------------
    #if saveOpac:
        # Here, we compute the full tau and transmission grid associated to the full range of
        # the atmosphere. We do so in order to save the full grid in savespectrum/makestruc
    
        # And here, we save in the struc the full grid
    
        #dtau = extinctCoef * np.tile(atm.dz[:,np.newaxis],(1,atm.nWave))
        #dtau = extinctCoef * np.outer(atm.dz,np.ones(atm.nWave))
    #    dtaufull = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:]) * np.outer(atm.dz,np.ones(atm.nWave, dtype=atm.numerical_precision))
    #    # Build grid of cumulative sums of tau from TOA to surface, starting at 0
    #    taufull=np.vstack([np.zeros(len(atm.wave), dtype=atm.numerical_precision),np.cumsum(dtaufull,axis=0)])
        # Ensure taugrid > 0
    #    taufull=taufull*(1+(np.arange(1,(atm.nLay+1))[:,np.newaxis])*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-10)+(np.arange(1,(atm.nLay+1))[:,np.newaxis]*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-99)
    
     #   for i in range(len(muObs)):                                      # Loop through all muObs
      #      transmisfull=np.exp((-taufull)/muObs[i])                 # e^(-Tv/u)

      #  atm.opac['tau']        =taufull
    #    atm.opac['transmis']   =transmisfull
        #atm.opac['IntensityThermalEmissionAtm']   =IntensityThermalEmissionAtm
        #atm.opac['IntensityThermalEmissionGround']=IntensityThermalEmissionGround
    #----------------------------------------------------------------------------------------------------------------------------------    
    # Return spectrum with effective clouds
    pdb.set_trace() 
    return thermal
    
    
                                                            #    def calcEmissionSpectrum(atm, extinctCoef, saveOpac=False, muObs=np.array([0.5773502691896258]), FluxUnit='W/(m**2*um)'):
                                                            #        '''
                                                            #        Computes thermal emission
                                                            #        --------------------------
                                                            #        calcEmissionSpectrum(muObs,OutputUnit)
                                                            #        muObs set to nominal value unless specified
                                                            #        OutputUnit set to W/(m**2**um) unless specified
                                                            #        
                                                            #        Output is the intensity of the thermal emission in shape: (atm.nWave,) OR (atm.nWave,len(muObs)) if len(muObs)>1 
                                                            #        '''
                                                            #        # Defn req parameters
                                                            #        TSurf     = atm.T[-1]                       # Surface temp defn'ed at last entry of T -> T[-1]
                                                            #        atm.dz   = -np.diff(atm.z)        
                                                            #        #dtau = extinctCoef * np.tile(atm.dz[:,np.newaxis],(1,atm.nWave))
                                                            #
                                                            #        #dtau = extinctCoef * np.outer(atm.dz,np.ones(atm.nWave))
                                                            #        dtau = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:]) * np.outer(atm.dz,np.ones(atm.nWave))
                                                            #        
                                                            #        #if (np.array_equal(dtau,dtau2)):
                                                            #            #print "Success!"
                                                            #            #alternatively just type this into python console
                                                            #        
                                                            #        
                                                            #        # Build grid of cumulative sums of tau from TOA to surface, starting at 0
                                                            #        tau=np.vstack([np.zeros(len(atm.wave)),np.cumsum(dtau,axis=0)])
                                                            #        # Ensure taugrid > 0
                                                            #        tau=tau*(1+(np.arange(1,(atm.nLay+1))[:,np.newaxis])*np.ones([1,atm.nWave])*1e-10)+(np.arange(1,(atm.nLay+1))[:,np.newaxis]*np.ones([1,atm.nWave])*1e-99)
                                                            #        
                                                            #        ## Compute black body radiation for T and TSurf with radutils
                                                            #        B = rad.PlanckFct(np.tile(atm.T[:,np.newaxis],(1,atm.nWave)),np.tile(atm.wave[np.newaxis,:],(len(atm.T),1)),'um',FluxUnit,'rad')
                                                            #        #B = rad.PlanckFct(atm.T,atm.wave,'um',FluxUnit,'rad')
                                                            #        Bsurf = rad.PlanckFct(TSurf,atm.wave[np.newaxis,:],'um',FluxUnit,'flux')
                                                            #
                                                            #        ## Calculate thermal emission for each muObs 
                                                            #        IntensityThermalEmission=np.zeros([len(muObs),atm.nWave])       # Create empty array
                                                            #
                                                            #        for i in range(len(muObs)):                                      # Loop through all muObs
                                                            #        
                                                            #            transmis=np.exp((-tau)/muObs[i])                 # e^(-Tv/u)
                                                            #            #------------Calculate second term of integral equation: Integration over optical depth------------
                                                            #            IntensityThermalEmissionAtm=-np.trapz(y=B,x=transmis,axis = 0)
                                                            #            
                                                            #            #------------Calculate surface contribution (first term integral equation)------------
                                                            #            IntensityThermalEmissionGround=Bsurf*transmis[-1,:]
                                                            #            
                                                            #            #------------Sum atmospheric and surface contributions ------------
                                                            #            IntensityThermalEmission[i,:]=IntensityThermalEmissionAtm+IntensityThermalEmissionGround
                                                            #        
                                                            #        # Check total thermal flux
                                                            #        #TotalThermalFlux=np.trapz(IntensityThermalEmission[:,0]*np.pi,atm.wave)
                                                            #        #AverageReradiationTemp=(TotalThermalFlux/sigmaSB)**(1/4)                 #only true if entire thermal emission is simulated
                                                            #        
                                                            #        if saveOpac:
                                                            #            atm.opac['tau']        =tau
                                                            #            atm.opac['transmis']   =transmis
                                                            #            #atm.opac['IntensityThermalEmissionAtm']   =IntensityThermalEmissionAtm
                                                            #            #atm.opac['IntensityThermalEmissionGround']=IntensityThermalEmissionGround
                                                            #            
                                                            #        # Return spectrum
                                                            #        thermal=np.squeeze(IntensityThermalEmission)
                                                            #        return thermal
    

    #%% Scattering


thermal_standard=atm.calcEmissionSpectrum(extinctCoef)
print(calcEmissionNew(extinctCoef))
print(thermal_standard)
pdb.set_trace()
