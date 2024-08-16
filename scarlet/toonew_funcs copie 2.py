import numexpr as ne
import pdb
import scarlet
from scarlet import radutils as rad
from numba import jit, vectorize
import numpy as np
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10
import matplotlib.pyplot as plt
from auxbenneke.constants import unitfac, pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom

def get_fluxes_toon(atm,IrradStar,extinctCoef,scatCoef,asym = 0,u0 = 0.5773502691896258, intTopLay=0,method_for_therm ='Hemispheric Mean', method_for_ref='Quadrature', hard_surface=False, surf_reflect=0, only_mid=True, ftau_cld=0, b_top_for_ref=0, ref=True, therm=True):
    nlev=atm.nLay # number of levels (60 in Scarlet)
    nlay=atm.nLay-1 # Number of layers (Interfaces between Levels)
    w0=scatCoef/extinctCoef # Single scattering albedo on each level
    w0=0.5*(w0[:nlev-1,:]+w0[1:nlev,:]) #Single scattering albedo interpolated for on each layer
    nWave=atm.nWave # size of wavelength grid (unit of wavelengths: um)
    
    T = atm.T # Temperature grid (units of temperatures: K)
    z = atm.z # altitude grid (units: m)
    
    u1=0.5
    
    dz   = -np.diff(z) # vertical thickness of each layer (units: m)

    #build grid of tau on each layer (Same code as in calcEmissionSpectrum)
    dtau = 0.5*(extinctCoef[:nlev-1,:]+extinctCoef[1:nlev,:]) * np.outer(dz,np.ones(atm.nWave, dtype=atm.numerical_precision)) # 
    #if (np.array_equal(dtau,dtau2)):
        #print "Success!"
        #alternatively just type this into python console
        
    # Build grid of cumulative sums of tau from TOA to ground/cloud, starting at 0 (Same code as in calcEmissionSpectrum)
    tau=np.vstack([np.zeros(atm.nWave, dtype=atm.numerical_precision),np.cumsum(dtau,axis=0)])
    # Ensure taugrid > 0 (Same code as in calcEmissionSpectrum)
    tau*=(1+(np.arange(1,(nlev+1))[:,np.newaxis])*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-10)+(np.arange(1,(nlev+1))[:,np.newaxis]*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-99)
    
    if therm==True: # calculates upwards and downwards fluxes on each layer (adapted from Picasso)
        bb_all=np.zeros((nlev,atm.nWave))
        for i in range(nlev):
            bb_all[i]=rad.PlanckFct(atm.T[i],atm.wave,InputUnit='um',OutputUnit='W/(m**2*um)',RadianceOrFlux='rad') #Planck Function calculated on each level in units of W/(m**2*um)
    
        b0 = bb_all[0:-1,:] #Planck Function calculated at top of each layer
        b1 = (bb_all[1:,:] - b0) / dtau #Equation 26 Toon et Al.
    
        gam1=2.0-w0*(1.0+asym)  #Table 1 Toon et Al.
        gam2=w0*(1.0-asym) #Table 1 Toon et Al.
        gam3=0.5*(1.0-np.sqrt(3)*asym*u0) #Table 1 Toon et Al.
        gam4=1.0-gam3 #Table 1 Toon et Al.
    
        Lam=(gam1**2-gam2**2)**0.5 #Equation 21 Toon et Al.
        Gam=(gam1-Lam)/gam2 #Equation 22 Toon et Al.
    
        gamterm=1.0/(gam1+gam2) #part of Equation 27 Toon et Al.
    
        C_plus_up = 2*np.pi*u1*(b0 + b1* gamterm) #Equation 27 Toon et Al. #2 removed
        C_minus_up = 2*np.pi*u1*(b0 - b1* gamterm) #Equation 27 Toon et Al. #2 removed
    
        C_plus_down = 2*pi*u1*(b0 + b1 * dtau + b1 * gamterm) #Equation 27 Toon et Al. #2 removed
        C_minus_down = 2*pi*u1*(b0 + b1 * dtau - b1 * gamterm) #Equation 27 Toon et Al. #2 removed
    
        #calculate exponential terms needed for the tridiagonal rotated layered method
        exptrm = Lam*dtau
        #save from overflow 
        exptrm = slice_gt (exptrm, 35.0) 
    
        exptrm_positive = exp(exptrm) 
        exptrm_minus = 1.0/exptrm_positive

        #for flux heating calculations, the energy balance solver 
        #does not like a fixed zero at the TOA. 
        #to avoid a discontinuous kink at the last atmospher
        #layer we create this "fake" boundary condition
        #we imagine that the atmosphere continus up at an isothermal T and that 
        #there is optical depth from above the top to infinity 
        tau_top = dtau[0,:]*atm.p[0]/(atm.p[1]-atm.p[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
        #print(list(tau_top))
        #tau_top = 26.75*plevel[0]/(plevel[1]-plevel[0]) 
        b_top = (1.0 - exp(-tau_top / u1 )) * bb_all[0,:] * np.pi #  Btop=(1.-np.exp(-tautop/ubari))*B[0]
    
        if hard_surface:
            b_surface = bb_all[-1,:]*pi #for terrestrial, hard surface  
        else: 
            b_surface= (bb_all[-1,:] + b1[-1,:]*u1)*pi #(for non terrestrial)
        
        A, B, C, D = setup_tri_diag(nlay,atm.nWave,  C_plus_up, C_minus_up, 
                                C_plus_down, C_minus_down, b_top, b_surface, surf_reflect,
                                 Gam, dtau, 
                                exptrm_positive,  exptrm_minus) 
    
    
        positive = zeros((nlay, atm.nWave))
        negative = zeros((nlay, atm.nWave))

        #========================= Start loop over wavelength =========================
        L = 2*nlay
        for w in range(atm.nWave):
            #coefficient of posive and negative exponential terms 
            X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
            #unmix the coefficients
            positive[:,w] = X[::2] + X[1::2] 
            negative[:,w] = X[::2] - X[1::2]
    
    
    
        #if you stop here this is regular ole 2 stream
        f_up = (positive * exptrm_positive + Gam * negative * exptrm_minus + C_plus_up)

        #calculate everyting from Table 3 toon
        #from here forward is source function technique in toon
        G = (1/u1 - Lam)*positive     
        H = Gam*(Lam + 1/u1)*negative 
        J = Gam*(Lam + 1/u1)*positive 
        K = (1/u1 - Lam)*negative     
        alpha1 = 2*pi*(b0+b1*(gamterm - u1))
        alpha2 = 2*pi*b1
        sigma1 = 2*pi*(b0-b1*(gamterm - u1))
        sigma2 = 2*pi*b1

        flux_minus_all = zeros((nlev,atm.nWave))
        flux_plus_all = zeros((nlev,nWave))
        flux_minus_mdpt_all = zeros((nlev,nWave))
        flux_plus_mdpt_all = zeros((nlev,nWave))
    
        exptrm_positive_mdpt = exp(0.5*exptrm) 
        exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

        #================ START CRAZE LOOP OVER ANGLE #================
        flux_at_top_all = zeros((atm.nWave))
        flux_down_all = zeros((atm.nWave))

    
        ulist=np.array([0.09853,0.30453,0.56202,0.80198,0.96019])
        gaussweights=np.array([0.015747,0.073908,0.146386,0.167174,0.096781])
        
        for g in range(len(ulist)):
        
            flux_at_top = zeros((atm.nWave))
            flux_down = zeros((atm.nWave))
            
            flux_minus = zeros((nlev,atm.nWave))
            flux_plus = zeros((nlev,nWave))
            flux_minus_mdpt = zeros((nlev,nWave))
            flux_plus_mdpt = zeros((nlev,nWave))
            
            iubar = ulist[g]
            weight=gaussweights[g]

            if hard_surface:
                flux_plus[-1,:] = bb_all[-1,:] *2*pi # terrestrial flux /pi = intensity #2 removed
            else:
                flux_plus[-1,:] = ( bb_all[-1,:] + b1[-1,:] * iubar)*2*pi#no hard surface #2 removed     
                
            flux_minus[0,:] = (1 - exp(-tau_top / iubar)) * bb_all[0,:] *2*pi  #2 removed
            
            exptrm_angle = exp( - dtau / iubar)
            exptrm_angle_mdpt = exp( -0.5 * dtau / iubar) 

            for itop in range(nlay):

                #Equation 56 in Toon et Al.
                flux_minus[itop+1,:]=(flux_minus[itop,:]*exptrm_angle[itop,:]+
                                        (J[itop,:]/(Lam[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                        (K[itop,:]/(Lam[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                #Equation 56 in Toon et Al.
                flux_minus_mdpt[itop,:]=(flux_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(Lam[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-Lam[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-iubar))

                ibot=nlay-1-itop

                #Equation 55 in Toon et Al.
                flux_plus[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                    (G[ibot,:]/(Lam[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                    (H[ibot,:]/(Lam[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                    alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                    alpha2[ibot,:]*(iubar-(dtau[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                #Equation 55 in Toon et Al.
                flux_plus_mdpt[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                        (G[ibot,:]/(Lam[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                        (H[ibot,:]/(Lam[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                        alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                        alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )


                #Upwards flux at top of atmosphere
                flux_at_top[:] = flux_plus_mdpt[0,:] #nlevel by nwno 
                
            flux_minus_all+=flux_minus*weight            
            flux_minus_mdpt_all+=flux_minus_mdpt*weight
            flux_plus_all+=flux_plus*weight
            flux_plus_mdpt_all+=flux_plus_mdpt*weight
            flux_at_top_all+=flux_at_top*weight
            
        
   
        if only_mid==True:
            #thermal_component = flux_plus_mdpt_all,flux_minus_mdpt_all
            thermal_component = flux_plus_all,flux_minus_all
        else:
            thermal_component = (flux_plus_mdpt_all,flux_minus_mdpt_all), (flux_plus_all, flux_minus_all)
        
    if ref==True:
    
        if method_for_ref == 'Eddington': #Eddington Method
            gam1  = (7-w0*(4+3*ftau_cld*asym))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #Table 1 Toon et Al. 
            gam2  = -(1-w0*(4-3*ftau_cld*asym))/4 #(sq3*w0*0.5)*(1.-cosb)       #Table 1 Toon et Al.
            gam3  = (2-3*ftau_cld*asym*u0)/4 #Table 1 Toon et Al.
        elif method_for_ref == 'Quadrature':#quadrature
            gam1  = (sqrt(3)*0.5)*(2. - w0*(1.+ftau_cld*asym)) #Table 1 Toon et Al.
            gam2  = (sqrt(3)*w0*0.5)*(1.-ftau_cld*asym)        #Table 1 Toon et Al.
            gam3  = 0.5*(1.-sqrt(3)*ftau_cld*asym*u0) #Table 1 Toon et Al.
        
        gam4 = 1.0 - gam3 #Table 1 Toon et Al.
        
        Lam=(gam1**2-gam2**2)**0.5 #Equation 21 Toon et Al.
        Gam=(gam1-Lam)/gam2 #Equation 22 Toon et Al.
    
        denominator=Lam**2-1.0/u0**2
    
        C_plus_up= IrradStar*w0* (gam3*(gam1 - 1.0/u0) +gam2*gam4 ) / denominator * np.exp(-(tau[:-1])/u0) #Equation 23 Toon et Al.  
        C_minus_up=IrradStar*w0* (gam4*(gam1 + 1.0/u0) +gam2*gam3 ) / denominator * np.exp(-(tau[:-1])/u0) #Equation 24 Toon et Al.
   
        C_plus_down=IrradStar*w0* (gam3*(gam1 - 1.0/u0) +gam2*gam4 ) / denominator *np.exp(-(tau[1:])/u0) #Equation 23 Toon et Al.  
        C_minus_down=IrradStar*w0* (gam4*(gam1 + 1.0/u0) +gam2*gam3 ) / denominator *np.exp(-(tau[1:])/u0) #Equation 24 Toon et Al.
    
        #taus at the midpoint
        taumid=tau[:-1]+0.5*dtau
        x = exp(-taumid/u0)
    
        C_plus_mid=IrradStar*w0* (gam3*(gam1 - 1.0/u0) +gam2*gam4 ) / denominator * np.exp(-taumid/u0) #Equation 23 Toon et Al.  
        C_minus_mid=IrradStar*w0* (gam4*(gam1 + 1.0/u0) +gam2*gam3 ) / denominator * np.exp(-taumid/u0) #Equation 24 Toon et Al.
    
    
        #calculate exponential terms needed for the tridiagonal rotated layered method
        exptrm = Lam*dtau
        #save from overflow 
        exptrm = slice_gt (exptrm, 35.0) 

        exptrm_positive = exp(exptrm) #EP
        exptrm_minus = 1.0/exptrm_positive#EM

        #boundary conditions 
        b_top = b_top_for_ref                                      

        b_surface = 0. + surf_reflect*u0*IrradStar*exp(-tau[-1, :]/u0)  #Toon et Al. Equation 37 

        #Now we need the terms for the tridiagonal rotated layered method
        #if tridiagonal==0:
        A, B, C, D = setup_tri_diag(nlay,nWave,  C_plus_up, C_minus_up, 
                                C_plus_down, C_minus_down, b_top, b_surface, surf_reflect,
                                Gam, dtau, 
                                exptrm_positive,  exptrm_minus)    

        positive = zeros((nlay, atm.nWave))
        negative = zeros((nlay, atm.nWave))
        #========================= Start loop over wavelength =========================
        L = 2*nlay
        for w in range(nWave):
            #coefficient of posive and negative exponential terms 
            X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
            #unmix the coefficients
            positive[:,w] = X[::2] + X[1::2] #Equation 29 Toon et Al.
            negative[:,w] = X[::2] - X[1::2] #Equation 30 Toon et Al.


        #========================= Get fluxes if needed for climate =========================

        flux_minus=np.zeros(shape=(nlev,nWave))
        flux_plus=np.zeros(shape=(nlev,nWave))
                
        flux_minus_midpt = np.zeros(shape=(nlev,nWave))
        flux_plus_midpt = np.zeros(shape=(nlev,nWave))
        #use expression for bottom flux to get the flux_plus and flux_minus at last
        #bottom layer
        flux_minus[:-1, :]  = positive*Gam + negative + C_minus_up #Equation 32 Toon et Al.
        flux_plus[:-1, :]  = positive + Gam*negative + C_plus_up #Equation 31 Toon et Al.
                
        flux_zero_minus  = Gam[-1,:]*positive[-1,:]*exptrm_positive[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + C_minus_down[-1,:] #Equation 32 Toon et Al.
        flux_zero_plus  = positive[-1,:]*exptrm_positive[-1,:] + Gam[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + C_plus_down[-1,:] #Equation 31 Toon et Al.
                
        flux_minus[-1, :], flux_plus[-1, :] = flux_zero_minus, flux_zero_plus 
                
        #add in direct flux term to the downwelling radiation, liou 182
        flux_minus = flux_minus + u0*IrradStar*exp(-tau/u0)

        #now get midpoint values 
        exptrm_positive_midpt = exp(0.5*exptrm) #EP
        exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM
                
        #fluxes at the midpoints 
        flux_minus_midpt[:-1,:]= Gam*positive*exptrm_positive_midpt + negative*exptrm_minus_midpt + C_minus_mid #Equation 32 Toon et Al.
        flux_plus_midpt[:-1,:]= positive*exptrm_positive_midpt + Gam*negative*exptrm_minus_midpt + C_plus_mid #Equation 31 Toon et Al.
        #add in midpoint downwelling radiation
        flux_minus_midpt[:-1,:] = flux_minus_midpt[:-1,:] + u0*IrradStar*exp(-taumid/u0)
    
        if only_mid==True:
            #reflected_component = flux_plus_midpt*0.5,flux_minus_midpt*0.5
            reflected_component = flux_plus*0.5,flux_minus*0.5
        else:
            reflected_component = (flux_plus_midpt,flux_minus_midpt), (flux_plus, flux_minus)
    if (therm==True and ref==True):
        if only_mid==True:
            return thermal_component[0]+reflected_component[0],thermal_component[1]+reflected_component[1]
        else:
            return (thermal_component[0][0]+reflected_component[0][0],thermal_component[0][1]+reflected_component[0][1]),(thermal_component[1][0]+reflected_component[1][0],thermal_component[1][1]+reflected_component[1][1])
    elif therm==True:
        return thermal_component
    elif ref==True:
        return reflected_component
    else:
        print('aa')

@jit(nopython=True, cache=True)
def slice_gt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new>lim)] = lim
        array[i,:] = new     
    return array
    
@jit(nopython=True, cache=True)
def setup_tri_diag(nlayer,nwno ,c_plus_up, c_minus_up, 
    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    gama, dtau, exptrm_positive,  exptrm_minus):
    """
    Before we can solve the tridiagonal matrix (See Toon+1989) section
    "SOLUTION OF THE TwO-STREAM EQUATIONS FOR MULTIPLE LAYERS", we 
    need to set up the coefficients. 
    Parameters
    ----------
    nlayer : int 
        number of layers in the model 
    nwno : int 
        number of wavelength points
    c_plus_up : array 
        c-plus evaluated at the top of the atmosphere 
    c_minus_up : array 
        c_minus evaluated at the top of the atmosphere 
    c_plus_down : array 
        c_plus evaluated at the bottom of the atmosphere 
    c_minus_down : array 
        c_minus evaluated at the bottom of the atmosphere 
    b_top : array 
        The diffuse radiation into the model at the top of the atmosphere
    b_surface : array
        The diffuse radiation into the model at the bottom. Includes emission, reflection 
        of the unattenuated portion of the direct beam  
    surf_reflect : array 
        Surface reflectivity 
    g1 : array 
        table 1 toon et al 1989
    g2 : array 
        table 1 toon et al 1989
    g3 : array 
        table 1 toon et al 1989
    lamba : array 
        Eqn 21 toon et al 1989 
    gama : array 
        Eqn 22 toon et al 1989
    dtau : array 
        Opacity per layer
    exptrm_positive : array 
        Eqn 44, expoential terms needed for tridiagonal rotated layered, clipped at 35 
    exptrm_minus : array 
        Eqn 44, expoential terms needed for tridiagonal rotated layered, clipped at 35 
    Returns
    -------
    array 
        coefficient of the positive exponential term 
    
    """
    L = 2 * nlayer

    #EQN 44 

    e1 = exptrm_positive + gama*exptrm_minus
    e2 = exptrm_positive - gama*exptrm_minus
    e3 = gama*exptrm_positive + exptrm_minus
    e4 = gama*exptrm_positive - exptrm_minus


    #now build terms 
    A = zeros((L,nwno)) 
    B = zeros((L,nwno )) 
    C = zeros((L,nwno )) 
    D = zeros((L,nwno )) 

    A[0,:] = 0.0
    B[0,:] = gama[0,:] + 1.0
    C[0,:] = gama[0,:] - 1.0
    D[0,:] = b_top - c_minus_up[0,:]

    #even terms, not including the last !CMM1 = UP
    A[1::2,:][:-1] = (e1[:-1,:]+e3[:-1,:]) * (gama[1:,:]-1.0) #always good
    B[1::2,:][:-1] = (e2[:-1,:]+e4[:-1,:]) * (gama[1:,:]-1.0)
    C[1::2,:][:-1] = 2.0 * (1.0-gama[1:,:]**2)          #always good 
    D[1::2,:][:-1] =((gama[1:,:]-1.0)*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
                            (1.0-gama[1:,:])*(c_minus_down[:-1,:] - c_minus_up[1:,:]))
    #import pickle as pk
    #pk.dump({'GAMA_1':(gama[1:,:]-1.0), 'CPM1':c_plus_up[1:,:] , 'CP':c_plus_down[:-1,:], '1_GAMA':(1.0-gama[1:,:]), 
    #   'CM':c_minus_down[:-1,:],'CMM1':c_minus_up[1:,:],'Deven':D[1::2,:][:-1]}, open('../testing_notebooks/GFLUX_even_D_terms.pk','wb'))
    
    #odd terms, not including the first 
    A[::2,:][1:] = 2.0*(1.0-gama[:-1,:]**2)
    B[::2,:][1:] = (e1[:-1,:]-e3[:-1,:]) * (gama[1:,:]+1.0)
    C[::2,:][1:] = (e1[:-1,:]+e3[:-1,:]) * (gama[1:,:]-1.0)
    D[::2,:][1:] = (e3[:-1,:]*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
                            e1[:-1,:]*(c_minus_down[:-1,:] - c_minus_up[1:,:]))

    #last term [L-1]
    A[-1,:] = e1[-1,:]-surf_reflect*e3[-1,:]
    B[-1,:] = e2[-1,:]-surf_reflect*e4[-1,:]
    C[-1,:] = 0.0
    D[-1,:] = b_surface-c_plus_down[-1,:] + surf_reflect*c_minus_down[-1,:]

    return A, B, C, D
    
@jit(nopython=True, cache=True)
def tri_diag_solve(l, a, b, c, d):
    """
    Tridiagonal Matrix Algorithm solver, a b c d can be NumPy array type or Python list type.
    refer to this wiki_ and to this explanation_. 
    
    .. _wiki: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    .. _explanation: http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    
    A, B, C and D refer to: 
    .. math:: A(I)*X(I-1) + B(I)*X(I) + C(I)*X(I+1) = D(I)
    This solver returns X. 
    Parameters
    ----------
    A : array or list 
    B : array or list 
    C : array or list 
    C : array or list 
    Returns
    -------
    array 
        Solution, x 
    """
    AS, DS, CS, DS,XK = zeros(l), zeros(l), zeros(l), zeros(l), zeros(l) # copy arrays

    AS[-1] = a[-1]/b[-1]
    DS[-1] = d[-1]/b[-1]

    for i in range(l-2, -1, -1):
        x = 1.0 / (b[i] - c[i] * AS[i+1])
        AS[i] = a[i] * x
        DS[i] = (d[i]-c[i] * DS[i+1]) * x
    XK[0] = DS[0]
    for i in range(1,l):
        XK[i] = DS[i] - AS[i] * XK[i-1]
    return XK    