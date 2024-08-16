import numexpr as ne
import pdb
import scarlet
from scarlet import radutils as rad
from numba import jit, vectorize
import numpy as np
from numpy import exp, zeros, where, sqrt, cumsum , pi, outer, sinh, cosh, min, dot, array,log, log10
import matplotlib.pyplot as plt
from auxbenneke.constants import unitfac, pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
#print(atm.p)
#print(atm.wave)
#print(dir(atm))
from picaso import fluxes,climate,disco



atm.T=np.ones(60)*500
extinctCoef0=atm.opac['extinctCoef'].copy()
scatCoef0=atm.opac['scatCoef'].copy()

w=-1


def multiScatToon(IrradStar,extinctCoef0,scatCoef0,asym = 0,u0 = 0.5773502691896258,refl = False,thermal = True,calcJ = False,intTopLay=0,method ='Hemispheric Mean',w=-1):
    
    
    if w == -1:
        atm.omega = scatCoef0/extinctCoef0

    else:
        atm.omega = w*np.ones_like(extinctCoef0, dtype=atm.numerical_precision)

    extinctCoef = 0.5*(extinctCoef0[:atm.nLay-1,:]+extinctCoef0[1:atm.nLay,:])
    scatCoef = 0.5*(scatCoef0[:atm.nLay-1,:]+scatCoef0[1:atm.nLay,:]) 
    
    
    nlayer=atm.nLay-1
    
    dz   = -np.diff(atm.z)
    dtau = 0.5*(extinctCoef0[:nlayer,:]+extinctCoef0[1:atm.nLay,:]) * np.outer(dz,np.ones(atm.nWave, dtype=atm.numerical_precision))
    tau=np.vstack([np.zeros(atm.nWave, dtype=atm.numerical_precision),np.cumsum(dtau,axis=0)])
    # Ensure taugrid > 0
    tau*=(1+(np.arange(1,(atm.nLay+1))[:,np.newaxis])*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-10)+(np.arange(1,(atm.nLay+1))[:,np.newaxis]*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-99) 
   
    pdb.set_trace()
   
    surf_reflect=0
   
    '''
    Computes spectrum with nonhomogenous scattering
    --------------------------
    multiScatToon(asymmetry factor,u,IrradStar,tau,method)
    Intensity at top of layer for thermal spectrum default at 0
    Returns Fupw, Fdwn, and Fnet for every layer of atm
    '''
        
        
    plt.plot(atm.wave,atm.omega[0])
    #plt.xlim(0.9,1.1)
    plt.ylim(0,1)
    plt.xscale('log')
    plt.savefig('omega.png')
    plt.clf()
    
    #plt.plot(atm.wave,extinctCoef[30])
    #plt.xlim(0,10)
    #plt.savefig('extinctCoef.png')

    atm.dz   = -np.diff(atm.z)        
    taun = extinctCoef * np.outer(atm.dz,np.ones(atm.nWave, dtype=atm.numerical_precision)) #optical thickness on each layer

    #atm.omega = np.vstack([np.zeros(len(atm.wave), dtype=atm.numerical_precision),atm.omega])
    #atm.omega[0,:] = np.nan
     
    atm.omega = 0.5*(atm.omega[:atm.nLay-1,:]+atm.omega[1:atm.nLay,:])
    
    [l,d] = np.shape(atm.omega)
    
    atm.Fs = IrradStar
    atm.u0 = u0
        
    tauc=np.vstack([np.zeros(len(atm.wave), dtype=atm.numerical_precision),np.zeros(len(atm.wave), dtype=atm.numerical_precision),np.cumsum(taun,axis=0)])
    tauc = np.delete(tauc,(l+1),axis=0)
    tauc[0,:] = np.nan
        
    taun=np.vstack([np.zeros(len(atm.wave)),taun])
    taun[0,:] = np.nan
        

    
    
    
        
    if refl:
        GroundAlbedo = atm.GroundAlbedo
    else:
        GroundAlbedo = 0
        
    ## Force Hemispheric Mean method when thermal = true
    if thermal == True:
        method = 'Hemispheric Mean'
            
    # Build co-eff for various approaches
    if method == 'Eddington':
        atm.g1 = (7-atm.omega*(4-3*asym))/4
        atm.g2 = -(1-atm.omega*(4-3*asym))/4
        atm.g3 = (2-3*asym*atm.u0)/4
        u1 = 1/2
    elif method == 'Quadrature':
        atm.g1 = 3**(1/2)*(2-atm.omega*(1+asym))/2
        atm.g2 = atm.omega*3**(1/2)*(1-asym)/2
        atm.g3 = (1-3**(1/2)*asym*atm.u0)/2
        u1 = 1/3**(1/2)
    elif method == 'Hemispheric Mean':
        atm.g1 = 2-atm.omega*(1+asym)
        atm.g2 = atm.omega*(1-asym)
        atm.g3 = (1-3**(1/2)*asym*atm.u0)/2
        u1 = 1/2
        
    #new
    
    sq3 = np.sqrt(3.)
    atm.g1  = (sq3*0.5)*(2. - atm.omega*(1.+asym))
    atm.g2  = (sq3*atm.omega*0.5)*(1.-asym)         #table 1
    atm.g3  = 0.5*(1.-sq3*asym*atm.u0)
    atm.lmbda=np.sqrt(atm.g1**2 - atm.g2**2) 
    atm.gamma=(atm.g1-atm.lmbda)/atm.g2
      
    atm.g4 = 1.0 - atm.g3
    
    denominator=atm.lmbda**2 - 1.0/atm.u0**2.0
    
    a_minus=atm.Fs*atm.omega* (atm.g4*(atm.g1 + 1.0/atm.u0) +atm.g2*atm.g3 ) / denominator
    a_plus  = atm.Fs*atm.omega*(atm.g3*(atm.g1-1.0/atm.u0) +atm.g2*atm.g4) / denominator
    
    x = np.exp(-tau[:-1,:]/atm.u0)
    c_minus_up = a_minus*x #CMM1
    c_plus_up  = a_plus*x #CPM1
    x = np.exp(-tau[1:,:]/atm.u0)
    c_minus_down = a_minus*x #CM
    c_plus_down  = a_plus*x #CP
    
    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = atm.lmbda*dtau
    #save from overflow 
    exptrm = slice_gt (exptrm, 35.0)
    
    exptrm_positive = exp(exptrm) #EP
    exptrm_minus = 1.0/exptrm_positive#exp(-exptrm) #EM
    
    
     #boundary conditions 
    b_top = 0.0                                       
    b_surface = 0. + surf_reflect*atm.u0*atm.Fs*exp(-tau[-1, :]/atm.u0)
    
    A, B, C, D = fluxes.setup_tri_diag(nlayer,atm.nWave,  c_plus_up, c_minus_up, c_plus_down, c_minus_down, b_top, b_surface, surf_reflect, atm.gamma, dtau, exptrm_positive,  exptrm_minus)
    
    #..
    
    positive = zeros((nlayer, atm.nWave))
    negative = zeros((nlayer, atm.nWave))
    
    L = 2*nlayer
    for w in range(atm.nWave):
        #coefficient of posive and negative exponential terms 
        #pdb.set_trace()
        X = fluxes.tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
        #unmix the coefficients
        positive[:,w] = X[::2] + X[1::2] 
        negative[:,w] = X[::2] - X[1::2]
    
    #might have to add this in to avoid numerical problems later. 
    #if len(np.where(negative[:,w]/X[::2] < 1e-30)) >0 , print(negative[:,w],X[::2],negative[:,w]/X[::2])

    #evaluate the fluxes through the layers 
    #use the top optical depth expression to evaluate fp and fm 
    #at the top of each layer 
    flux_plus  = zeros((atm.nLay, atm.nWave))
    flux_minus = zeros((atm.nLay, atm.nWave))
    flux_plus[:-1,:]  = positive + atm.gamma*negative + c_plus_up #everything but the last row (botton of atmosphere)
    flux_minus[:-1,:] = positive*atm.gamma + negative + c_minus_up #everything but the last row (botton of atmosphere

    #use expression for bottom flux to get the flux_plus and flux_minus at last
    #bottom layer
    flux_plus[-1,:]  = positive[-1,:]*exptrm_positive[-1,:] + atm.gamma[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
    flux_minus[-1,:] = positive[-1,:]*exptrm_positive[-1,:]*atm.gamma[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + c_minus_down[-1,:]

    #we have solved for fluxes directly and no further integration is needed 
    #ubar is absorbed into the definition of g1-g4
    #ubar=0.5: hemispheric constant 
    #ubar=sqrt(3): gauss quadrature 
    #other cases in meador & weaver JAS, 37, 630-643, 1980

    #now add direct flux term to the downwelling radiation, Liou 1982
    flux_minus = flux_minus + atm.u0*atm.Fs*exp(-1.0*tau/atm.u0)

    #now calculate the fluxes at the midpoints of the layers 
    #exptrm_positive_mdpt = exp(0.5*exptrm) #EP_mdpt
    #exptrm_minus_mdpt = exp(-0.5*exptrm) #EM_mdpt

    #tau_mdpt = tau[:-1] + 0.5*dtau #start from bottom up to define midpoints 
    #c_plus_mdpt = a_plus*exp(-tau_mdpt/ubar0)
    #c_minus_mdpt = a_minus*exp(-tau_mdpt/ubar0)
    
    #flux_plus_mdpt = positive*exptrm_positive_mdpt + gama*negative*exptrm_minus_mdpt + c_plus_mdpt
    #flux_minus_mdpt = positive*exptrm_positive_mdpt*gama + negative*exptrm_minus_mdpt + c_minus_mdpt

    #add direct flux to downwelling term 
    #flux_minus_mdpt = flux_minus_mdpt + ubar0*F0PI*exp(-1.0*tau_mdpt/ubar0)

    pdb.set_trace()
    
    return flux_plus, flux_minus
    
def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, extinctCoef0, scatCoef0, w0, asym, plevel, ubar1,
    surf_reflect, hard_surface, dwno, calc_type):
    """
    This function uses the source function method, which is outlined here : 
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    
    The result of this routine is the top of the atmosphere thermal flux as 
    a function of gauss and chebychev points accross the disk. 

    Everything here is in CGS units:

    Fluxes - erg/s/cm^3
    Temperature - K 
    Wave grid - cm-1
    Pressure ; dyne/cm2

    Reminder: Flux = pi * Intensity, so if you are trying to compare the result of this with 
    a black body you will need to compare with pi * BB !

    Parameters
    ----------
    nlevel : int 
        Number of levels which occur at the grid points (not to be confused with layers which are
        mid points)
    wno : numpy.ndarray
        Wavenumber grid in inverse cm 
    nwno : int 
        Number of wavenumber points 
    numg : int 
        Number of gauss points (think longitude points)
    numt : int 
        Number of chebychev points (think latitude points)
    tlevel : numpy.ndarray
        Temperature as a function of level (not layer)
    dtau : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the per layer optical depth. 
    w0 : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    cosb : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the asymmetry of the 
        atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    plevel : numpy.ndarray
        Pressure for each level (not layer, which is midpoints). CGS units (dyne/cm2)
    ubar1 : numpy.ndarray
        This is a matrix of ng by nt. This describes the outgoing incident angles and is generally
        computed in `picaso.disco`
    surf_reflect : numpy.ndarray    
        Surface reflectivity as a function of wavenumber. 
    hard_surface : int
        0 for no hard surface (e.g. Jupiter/Neptune), 1 for hard surface (terrestrial)
    dwno : int 
        delta wno needed for climate
    calc_type : int 
        0 for spectrum model, 1 for climate solver
    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    """
    nlayer = nlevel - 1 #nlayers 
    
    if w == -1:
        atm.omega = scatCoef0/extinctCoef0

    else:
        atm.omega = w*np.ones_like(extinctCoef0, dtype=atm.numerical_precision)
    
    
    dz   = -np.diff(atm.z)
    dtau = 0.5*(extinctCoef0[:nlayer,:]+extinctCoef0[1:atm.nLay,:]) * np.outer(dz,np.ones(atm.nWave, dtype=atm.numerical_precision))

    mu1 = 0.5#0.88#0.5 #from Table 1 Toon  

    #get matrix of blackbodies 
    if calc_type == 0: 
        all_b = blackbody(tlevel, 1/wno) #returns nlevel by nwave   
    elif calc_type==1:
        all_b = blackbody_integrated(tlevel, wno, dwno)

    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89

    #hemispheric mean parameters from Tabe 1 toon 
    g1 = 2.0 - w0*(1+cosb); g2 = w0*(1-cosb)

    alpha = sqrt( (1.-w0) / (1.-w0*cosb) )
    lamda = sqrt(g1**2 - g2**2) #eqn 21 toon 
    gama = (g1-lamda)/g2 # #eqn 22 toon
    
    g1_plus_g2 = 1.0/(g1+g2) #second half of eqn.27

    #same as with reflected light, compute c_plus and c_minus 
    #these are eqns 27a & b in Toon89
    #_ups are evaluated at lower optical depth, TOA
    #_dows are evaluated at higher optical depth, bottom of atmosphere
    c_plus_up = 2*pi*mu1*(b0 + b1* g1_plus_g2) 
    c_minus_up = 2*pi*mu1*(b0 - b1* g1_plus_g2)
    #NOTE: to keep consistent with Toon, we keep these 2pis here. However, 
    #in 3d cases where we no long assume azimuthal symmetry, we divide out 
    #by 2pi when we multiply out the weights as seen in disco.compress_thermal 

    c_plus_down = 2*pi*mu1*(b0 + b1 * dtau + b1 * g1_plus_g2) 
    c_minus_down = 2*pi*mu1*(b0 + b1 * dtau - b1 * g1_plus_g2)



    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = lamda*dtau
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
    tau_top = dtau[0,:]*plevel[0]/(plevel[1]-plevel[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    #print(list(tau_top))
    #tau_top = 26.75*plevel[0]/(plevel[1]-plevel[0]) 
    b_top = (1.0 - exp(-tau_top / mu1 )) * all_b[0,:] * pi #  Btop=(1.-np.exp(-tautop/ubari))*B[0]
    
    if hard_surface:
        b_surface = all_b[-1,:]*pi #for terrestrial, hard surface  
    else: 
        b_surface= (all_b[-1,:] + b1[-1,:]*mu1)*pi #(for non terrestrial)

    #Now we need the terms for the tridiagonal rotated layered method
    #pentadiagonal solver is left here because it may be useful someday 
    #however, curret scipy implementation is too slow to use currently 
    #if tridiagonal==0:
    A, B, C, D = setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
                        c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                         gama, dtau, 
                        exptrm_positive,  exptrm_minus) 
    #else:
    #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
    #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    #                        gama, dtau, 
    #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 
    positive = zeros((nlayer, nwno))
    negative = zeros((nlayer, nwno))

    #========================= Start loop over wavelength =========================
    L = nlayer+nlayer
    for w in range(nwno):
        #coefficient of posive and negative exponential terms 
        X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
        #unmix the coefficients
        positive[:,w] = X[::2] + X[1::2] 
        negative[:,w] = X[::2] - X[1::2]
        #else:
        #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
        #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
        #   negative[:,w] = X[::2] - X[1::2]

    #if you stop here this is regular ole 2 stream
    f_up = (positive * exptrm_positive + gama * negative * exptrm_minus + c_plus_up)

    #calculate everyting from Table 3 toon
    #from here forward is source function technique in toon
    G = (1/mu1 - lamda)*positive     
    H = gama*(lamda + 1/mu1)*negative 
    J = gama*(lamda + 1/mu1)*positive 
    K = (1/mu1 - lamda)*negative     
    alpha1 = 2*pi*(b0+b1*(g1_plus_g2 - mu1)) 
    alpha2 = 2*pi*b1 
    sigma1 = 2*pi*(b0-b1*(g1_plus_g2 - mu1)) 
    sigma2 = 2*pi*b1 

    flux_minus = zeros((numg, numt,nlevel,nwno))
    flux_plus = zeros((numg, numt,nlevel,nwno))
    flux_minus_mdpt = zeros((numg, numt,nlevel,nwno))
    flux_plus_mdpt = zeros((numg, numt,nlevel,nwno))

    exptrm_positive_mdpt = exp(0.5*exptrm) 
    exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

    #================ START CRAZE LOOP OVER ANGLE #================
    flux_at_top = zeros((numg, numt, nwno))
    flux_down = zeros((numg, numt, nwno))

    #work through building eqn 55 in toon (tons of bookeeping exponentials)
    for ng in range(numg):
        for nt in range(numt): 

            iubar = ubar1[ng,nt]

            if hard_surface:
                flux_plus[ng,nt,-1,:] = all_b[-1,:] *2*pi  # terrestrial flux /pi = intensity
            else:
                flux_plus[ng,nt,-1,:] = ( all_b[-1,:] + b1[-1,:] * iubar)*2*pi #no hard surface   
                
            flux_minus[ng,nt,0,:] = (1 - exp(-tau_top / iubar)) * all_b[0,:] *2*pi
            
            exptrm_angle = exp( - dtau / iubar)
            exptrm_angle_mdpt = exp( -0.5 * dtau / iubar) 

            for itop in range(nlayer):

                #disbanning this for now because we dont need it in the thermal emission code
                flux_minus[ng,nt,itop+1,:]=(flux_minus[ng,nt,itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                flux_minus_mdpt[ng,nt,itop,:]=(flux_minus[ng,nt,itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau[itop,:]-iubar))

                ibot=nlayer-1-itop

                flux_plus[ng,nt,ibot,:]=(flux_plus[ng,nt,ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(iubar-(dtau[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                flux_plus_mdpt[ng,nt,ibot,:]=(flux_plus[ng,nt,ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )


            flux_at_top[ng,nt,:] = flux_plus_mdpt[ng,nt,0,:] #nlevel by nwno 

    return flux_at_top , (flux_minus, flux_plus, flux_minus_mdpt, flux_plus_mdpt)
    
    
  
  
  
  
  
  
  
  
  
  
  
  
        

def CP(tauc,tau,n):
    '''
    Computes Solar Radiation (Positive)
    --------------------------
    CP(tauc,tau,n)
    Computes solar radiation upward through a layer
    '''      
    C = atm.omega[n,:]*np.pi*atm.Fs*np.exp(-(tauc+tau)/atm.u0)*((atm.g1[n,:]-1/atm.u0)*atm.g3 + atm.g4*atm.g2[n,:]) / (atm.lmbda[n,:]**2 - 1/atm.u0**2)
    return C
    
    
def CM(tauc,tau,n):
    '''
    Computes Solar Radiation (Negative/Minus)
    --------------------------
    CM(tauc,tau,n)
    Computes solar radiation downwards through a layer
    '''      
    C = atm.omega[n,:]*np.pi*atm.Fs*np.exp(-(tauc+tau)/atm.u0)*((atm.g1[n,:]+1/atm.u0)*atm.g4 + atm.g2[n,:]*atm.g3) / (atm.lmbda[n,:]**2 - 1/atm.u0**2)
    return C
    
@jit(nopython=True, cache=True)
def slice_gt(array, lim):
    """Funciton to replace values with upper or lower limit
    """
    for i in range(array.shape[0]):
        new = array[i,:] 
        new[where(new>lim)] = lim
        array[i,:] = new     
    return array


toon_ref= multiScatToon(atm.IrradStar,extinctCoef0,scatCoef0,refl = True,thermal = True, calcJ = False, intTopLay=0, w=-1)

#toon_therm=get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1, surf_reflect, hard_surface, dwno, calc_type)
numg=5
numt=8

gangle,gweight,tangle,tweight=disco.get_angles_1d(num_g)

toon_therm=get_thermal_1d(atm.nLay, atm.wave,atm.nWave, numg,numt,atm.T, extinctCoef0, scatCoef0,asym=0, atm.p, ubar1, surf_reflect, hard_surface, dwno, calc_type)


#for qw in range(59):
#    toon[0][qw][:7900]=0
#    toon[1][qw][:7900]=0

plt.plot(atm.wave,toon[0][0])
plt.title('Upwards Flux Top Layer')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(-1000,1000)
#plt.plot(atm.wave[7900:],toon[0][0][7900:])
plt.savefig('toonfunc.png')
print(toon[0][0])
result_toon=np.trapz(x=atm.wave,y=toon[0][0])
print(result_toon)
