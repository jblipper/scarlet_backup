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


def get_reflected_1d(nlevel, wno,nwno, numg,numt, dtau, tau, w0, cosb, ftau_cld,
    surf_reflect,ubar0, ubar1, F0PI, 
    get_toa_intensity=1,get_lvl_flux=0,
    toon_coefficients=0,b_top=0):
    """
    Computes toon fluxes given tau and everything is 1 dimensional. This is the exact same function 
    as `get_flux_geom_3d` but is kept separately so we don't have to do unecessary indexing for fast
    retrievals. 
    Parameters
    ----------
    nlevel : int 
        Number of levels in the model 
    wno : array of float 
        Wave number grid in cm -1 
    nwno : int 
        Number of wave points
    numg : int 
        Number of Gauss angles 
    numt : int 
        Number of Chebyshev angles 
    DTAU : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    TAU : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT D-Eddington Correction
        Dimensions=# level by # wave        
    W0 : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    COSB : ndarray of float 
        This is the asymmetry factor 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    GCOS2 : ndarray of float 
        Parameter that allows us to directly include Rayleigh scattering 
        = 0.5*tau_rayleigh/(tau_rayleigh + tau_cloud)
    ftau_cld : ndarray of float 
        Fraction of cloud extinction to total 
        = tau_cloud/(tau_rayleigh + tau_cloud)
    ftau_ray : ndarray of float 
        Fraction of rayleigh extinction to total 
        = tau_rayleigh/(tau_rayleigh + tau_cloud)
    dtau_og : ndarray of float 
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# layer by # wave
    tau_og : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# level by # wave    
    w0_og : ndarray of float 
        Same as w0 but WITHOUT the delta eddington correction, if it was specified by user  
    cosb_og : ndarray of float 
        Same as cosbar buth WITHOUT the delta eddington correction, if it was specified by user
    surf_reflect : float 
        Surface reflectivity 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    ubar1 : ndarray of float 
        matrix of cosine of the observer angles
    cos_theta : float 
        Cosine of the phase angle of the planet 
    F0PI : array 
        Downward incident solar radiation
    single_phase : str 
        Single scattering phase function, default is the two-term henyey-greenstein phase function
    multi_phase : str 
        Multiple scattering phase function, defulat is N=2 Legendre polynomial 
    frac_a : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_b : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_c : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C), Default is : 1 - gcosb^2
    constant_back : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of back scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    constant_forward : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of forward scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    get_toa_intensity : int 
        (Optional) Default=1 is to only return the TOA intensity you would need for a 1D spectrum (1)
        otherwise it will return zeros for TOA intensity 
    get_lvl_flux : int 
        (Optional) Default=0 is to only compute TOA intensity and NOT return the lvl fluxes so this needs 
        to be flipped on for the climate calculations
    toon_coefficients : int     
        (Optional) 0 for quadrature (default) 1 for eddington

    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 
    """
    #these are only filled in if get_toa_intensity=1
    #outgoing intensity as a function of all the different angles
    xint_at_top = zeros(shape=(numg, numt, nwno))

    #these are only filled in if get_lvl_flux=1
    #fluxes at the boundaries 
    flux_minus_all = zeros(shape=(numg, numt,nlevel, nwno)) ## level downwelling fluxes
    flux_plus_all = zeros(shape=(numg, numt, nlevel, nwno)) ## level upwelling fluxes
    #fluxes at the midpoints
    flux_minus_midpt_all = zeros(shape=(numg, numt, nlevel, nwno)) ##  layer downwelling fluxes
    flux_plus_midpt_all = zeros(shape=(numg, numt, nlevel, nwno))  ## layer upwelling fluxes



    nlayer = nlevel - 1 

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    #terms not dependent on incident angle
    sq3 = sqrt(3.)
    if toon_coefficients == 1:#eddington
        g1  = (7-w0*(4+3*ftau_cld*cosb))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # 
        g2  = -(1-w0*(4-3*ftau_cld*cosb))/4 #(sq3*w0*0.5)*(1.-cosb)        #table 1 # 
    elif toon_coefficients == 0:#quadrature
        g1  = (sq3*0.5)*(2. - w0*(1.+ftau_cld*cosb)) #table 1 # 
        g2  = (sq3*w0*0.5)*(1.-ftau_cld*cosb)        #table 1 # 
    
    lamda = sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(numg):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u1 = 1/3**(1/2) #**
            u0 = ubar0[ng,nt]
            u0 = 0.5773502691896258 #**
            if toon_coefficients == 1 : #eddington
                g3  = (2-3*ftau_cld*cosb*u0)/4#0.5*(1.-sq3*cosb*ubar0[ng, nt]) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
            elif toon_coefficients == 0 :#quadrature
                g3  = 0.5*(1.-sq3*ftau_cld*cosb*u0) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
            
            # now calculate c_plus and c_minus (equation 23 and 24 toon)
            g4 = 1.0 - g3
            denominator = lamda**2 - 1.0/u0**2.0

            #everything but the exponential 
            a_minus = F0PI*w0* (g4*(g1 + 1.0/u0) +g2*g3 ) / denominator
            a_plus  = F0PI*w0*(g3*(g1-1.0/u0) +g2*g4) / denominator

            #add in exponential to get full eqn
            #_up is the terms evaluated at lower optical depths (higher altitudes)
            #_down is terms evaluated at higher optical depths (lower altitudes)
            x = exp(-tau[:-1,:]/u0)
            c_minus_up = a_minus*x #CMM1
            c_plus_up  = a_plus*x #CPM1
            x = exp(-tau[1:,:]/u0)
            c_minus_down = a_minus*x #CM
            c_plus_down  = a_plus*x #CP
            
            pdb.set_trace()

            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = lamda*dtau
            #save from overflow 
            exptrm = slice_gt (exptrm, 35.0) 

            exptrm_positive = exp(exptrm) #EP
            exptrm_minus = 1.0/exptrm_positive#EM


            #boundary conditions 
            #b_top = 0.0                                       

            b_surface = 0. + surf_reflect*u0*F0PI*exp(-tau[-1, :]/u0)

            #Now we need the terms for the tridiagonal rotated layered method
            #if tridiagonal==0:
            A, B, C, D = fluxes.setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
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
            L = 2*nlayer
            for w in range(nwno):
                #coefficient of posive and negative exponential terms 
                #if tridiagonal==0:
                X = fluxes.tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                #unmix the coefficients
                positive[:,w] = X[::2] + X[1::2] 
                negative[:,w] = X[::2] - X[1::2]

                #else: 
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                    #unmix the coefficients
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            #========================= End loop over wavelength =========================

            #========================= Get fluxes if needed for climate =========================
            if get_lvl_flux: 
                flux_minus=np.zeros(shape=(nlevel,nwno))
                flux_plus=np.zeros(shape=(nlevel,nwno))
                
                flux_minus_midpt = np.zeros(shape=(nlevel,nwno))
                flux_plus_midpt = np.zeros(shape=(nlevel,nwno))
                #use expression for bottom flux to get the flux_plus and flux_minus at last
                #bottom layer
                flux_minus[:-1, :]  = positive*gama + negative + c_minus_up
                flux_plus[:-1, :]  = positive + gama*negative + c_plus_up
                
                flux_zero_minus  = gama[-1,:]*positive[-1,:]*exptrm_positive[-1,:] + negative[-1,:]*exptrm_minus[-1,:] + c_minus_down[-1,:]
                flux_zero_plus  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
                
                flux_minus[-1, :], flux_plus[-1, :] = flux_zero_minus, flux_zero_plus 
                
                #add in direct flux term to the downwelling radiation, liou 182
                flux_minus = flux_minus + u0*F0PI*exp(-tau/u0)

                #now get midpoint values 
                exptrm_positive_midpt = exp(0.5*exptrm) #EP
                exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM
                
                #taus at the midpoint
                taumid=tau[:-1]+0.5*dtau
                x = exp(-taumid/ubar0[ng, nt])
                c_plus_mid= a_plus*x
                c_minus_mid=a_minus*x
                #fluxes at the midpoints 
                flux_minus_midpt[:-1,:]= gama*positive*exptrm_positive_midpt + negative*exptrm_minus_midpt + c_minus_mid
                flux_plus_midpt[:-1,:]= positive*exptrm_positive_midpt + gama*negative*exptrm_minus_midpt + c_plus_mid
                #add in midpoint downwelling radiation
                flux_minus_midpt[:-1,:] = flux_minus_midpt[:-1,:] + ubar0[ng, nt]*F0PI*exp(-taumid/ubar0[ng, nt])
                
                pdb.set_trace()

                #ARRAYS TO RETURN with all NG and NTs
                flux_minus_all[ng, nt, :, :]=flux_minus
                flux_plus_all[ng, nt, :, :]=flux_plus
                flux_minus_midpt_all[ng, nt, :, :]=flux_minus_midpt
                flux_plus_midpt_all[ng, nt, :, :]=flux_plus_midpt
            #========================= End get fluxes if needed for climate =========================

    return (flux_minus_all, flux_plus_all, flux_minus_midpt_all, flux_plus_midpt_all )
    





def get_thermal_1d(nlevel, wno,nwno, numg,numt,tlevel, dtau, w0,cosb,plevel, ubar1,
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

    mu1 =0.5 #0.88 #0.5773502691896258  #0.5#0.88#0.5 #from Table 1 Toon  

    #get matrix of blackbodies 
    if calc_type == 0: 
        all_b_0 = blackbody(tlevel, atm.wave/1e4)#*1e-7*1e-4 #returns nlevel by nwave  
        all_b=rad.convertIntensity(all_b_0,atm.wave/1e4,InputUnit='(ergs/s)/(cm**2*cm)',WavelengthUnit='cm',OutputUnit='W/(m**2*um)')
        all_b=all_b*np.pi
        plt.clf()
        plt.plot(atm.wave,all_b[0])
        plt.savefig('tessst.png')
        pdb.set_trace()
    elif calc_type==1:
        all_b_0 = blackbody_integrated(tlevel, wno, dwno)#*(10**5)
        all_b=rad.convertIntensity(all_b_0,wno,InputUnit='W/(m**2*cm**-1)',WavelengthUnit='cm**-1',OutputUnit='W/(m**2*um)')
        pdb.set_trace()
        
       # Tgrid = np.zeros([nlayer+2], dtype=atm.numerical_precision)
       # Tgrid[0] = np.nan
       # Tgrid[1] = atm.T[0]
      #  for i in range(2,nlayer+1):
      #      Tgrid[i] = (atm.T[i-2] + atm.T[i-1])/2
      #  Tgrid[-1] = atm.T[-1]
                            
      #  B0 = np.zeros([nlayer,nwno], dtype=atm.numerical_precision) ;  B0[0,:] = np.nan
      #  B1 = np.zeros([nlayer,nwno], dtype=atm.numerical_precision) ;  B1[0,:] = np.nan
       # for i in range (nlayer):
       #     B0[i,:] = rad.PlanckFct(Tgrid[i],atm.wave,'um','W/(m**2*um)','rad')
       #     B1[i,:] = (rad.PlanckFct(Tgrid[i+1],atm.wave,'um','W/(m**2*um)','rad') - B0[i,:])/dtau[i,:]

    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau # eqn 26 toon 89
    
    #pdb.set_trace()
    
    #b0=B0
    #b1=B1

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
    pdb.set_trace()


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
    A, B, C, D = fluxes.setup_tri_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
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
        X = fluxes.tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
        #unmix the coefficients
        positive[:,w] = X[::2] + X[1::2] 
        negative[:,w] = X[::2] - X[1::2]
        #else:
        #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
        #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
        #   negative[:,w] = X[::2] - X[1::2]

    pdb.set_trace()
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
            #iubar = 1/3**(1/2) #**

            if hard_surface:
                flux_plus[ng,nt,-1,:] = all_b[-1,:] *2*pi  # terrestrial flux /pi = intensity
            else:
                flux_plus[ng,nt,-1,:] = ( all_b[-1,:] + b1[-1,:] * iubar)*2*pi #no hard surface   
                
            flux_minus[ng,nt,0,:] = (1 - exp(-tau_top / iubar)) * all_b[0,:] *2*pi
            
            #pdb.set_trace()
            
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








def get_fluxes( pressure, temperature, dwni, DTAU, TAU, W0, COSB,ftau_cld, surf_reflect, ubar0,ubar1, F0PI,
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts,reflected=True, thermal=True):
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
    

    reflected : bool 
        Run reflected light
    thermal : bool 
        Run thermal emission

        
    Return
    ------
    array
        Visible and IR -- net (layer and level), upward (level) and downward (level)  fluxes
    """
    #print('enter climate')

    # for visible
    flux_net_v = np.zeros(shape=(ng,nt,nlevel)) #net level visible fluxes
    flux_net_v_layer=np.zeros(shape=(ng,nt,nlevel)) #net layer visible fluxes

    flux_plus_v= np.zeros(shape=(ng,nt,nlevel,nwno)) # level plus visible fluxes
    flux_minus_v= np.zeros(shape=(ng,nt,nlevel,nwno)) # level minus visible fluxes
    
    #"""<<<<<<< NEWCLIMA
    # for thermal
    flux_plus_midpt = np.zeros(shape=(ng,nt,nlevel,nwno))
    flux_minus_midpt = np.zeros(shape=(ng,nt,nlevel,nwno))

    flux_plus = np.zeros(shape=(ng,nt,nlevel,nwno))
    flux_minus = np.zeros(shape=(ng,nt,nlevel,nwno))
    #"""

    """<<<<<<< OG
    # for thermal
    flux_plus_midpt = np.zeros(shape=(nlevel,nwno))
    flux_minus_midpt = np.zeros(shape=(nlevel,nwno))

    flux_plus = np.zeros(shape=(nlevel,nwno))
    flux_minus = np.zeros(shape=(nlevel,nwno))
    """

    # outputs needed for climate
    flux_net_ir = np.zeros(shape=(nlevel)) #net level visible fluxes
    flux_net_ir_layer=np.zeros(shape=(nlevel)) #net layer visible fluxes

    flux_plus_ir= np.zeros(shape=(nlevel,nwno)) # level plus visible fluxes
    flux_minus_ir= np.zeros(shape=(nlevel,nwno)) # level minus visible fluxes

    
    #ugauss_angles= np.array([0.0985350858,0.3045357266,0.5620251898,0.8019865821,0.9601901429])    
    #ugauss_weights = np.array([0.0157479145,0.0739088701,0.1463869871,0.1671746381,0.0967815902])
    #ugauss_angles = np.array([0.66666])
    #ugauss_weights = np.array([0.5])

    if reflected:
        #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
        b_top = 0.0
        for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
            #"""
            #<<<<<<< NEWCLIMA
            #here only the fluxes are returned since we dont care about the outgoing intensity at the 
            #top, which is only used for albedo/ref light spectra
            ng_clima,nt_clima=1,1
            ubar0_clima = ubar0*0+0.5
            ubar1_clima = ubar1*0+0.5
            #_, out_ref_fluxes = get_reflected_1d(nlevel, wno,nwno,ng_clima,nt_clima,
            #                        DTAU[:,:,ig], TAU[:,:,ig], W0[:,:,ig], COSB[:,:,ig],
            #                        ftau_cld[:,:,ig],surf_reflect, ubar0_clima,ubar1_clima,
            #                        F0PI, get_toa_intensity=0, get_lvl_flux=1)
            out_ref_fluxes = get_reflected_1d(nlevel, wno,nwno,ng_clima,nt_clima,
                                    DTAU[:,:], TAU[:,:], W0[:,:], COSB,
                                    ftau_cld,surf_reflect, ubar0_clima,ubar1_clima,
                                    F0PI, get_toa_intensity=0, get_lvl_flux=1)

            flux_minus_all_v, flux_plus_all_v, flux_minus_midpt_all_v, flux_plus_midpt_all_v = out_ref_fluxes
            

            flux_net_v_layer += (np.sum(flux_plus_midpt_all_v,axis=3)-np.sum(flux_minus_midpt_all_v,axis=3))*gauss_wts[ig]
            flux_net_v += (np.sum(flux_plus_all_v,axis=3)-np.sum(flux_minus_all_v,axis=3))*gauss_wts[ig]

            #======="""
            #nlevel = atm.c.nlevel


            flux_plus_v += flux_plus_all_v*gauss_wts[ig]
            flux_minus_v += flux_minus_all_v*gauss_wts[ig]

        #if full output is requested add in xint at top for 3d plots


    if thermal:

        #use toon method (and tridiagonal matrix solver) to get net cumulative fluxes 
        
        for ig in range(ngauss): # correlated - loop (which is different from gauss-tchevychev angle)
            
            #remember all OG values (e.g. no delta eddington correction) go into thermal as well as 
            #the uncorrected raman single scattering 
            
            hard_surface = 0 
            #_,out_therm_fluxes = get_thermal_1d(nlevel, wno,nwno,ng,nt,temperature,
            #                                DTAU[:,:,ig], COSB[:,:,ig], pressure,ubar1,
            #                                surf_reflect, hard_surface, dwni, calc_type=1)
            _,out_therm_fluxes = get_thermal_1d(nlevel, wno,nwno,ng,nt,temperature,
                                            DTAU[:,:], W0, COSB, pressure,ubar1,
                                            surf_reflect, hard_surface, dwni, calc_type=1)

            flux_minus_all_i, flux_plus_all_i, flux_minus_midpt_all_i, flux_plus_midpt_all_i = out_therm_fluxes

            #gauss_wts[ig]=0.1 #**
            
            flux_plus += flux_plus_all_i*gauss_wts[ig]
            flux_minus += flux_minus_all_i*gauss_wts[ig]
            flux_plus_midpt += flux_plus_midpt_all_i*gauss_wts[ig]#*weights
            flux_minus_midpt += flux_minus_midpt_all_i*gauss_wts[ig]#*weights
            #pdb.set_trace()
            #print("aa")


        flux_plus = disco.compress_thermal(nwno, flux_plus, gweight, tweight)
        flux_minus= disco.compress_thermal(nwno, flux_minus, gweight, tweight)
        flux_plus_midpt= disco.compress_thermal(nwno, flux_plus_midpt, gweight, tweight)
        flux_minus_midpt= disco.compress_thermal(nwno, flux_minus_midpt, gweight, tweight)
        #"""

        for wvi in range(nwno):
            flux_net_ir_layer += (flux_plus_midpt[:,wvi]-flux_minus_midpt[:,wvi]) * dwni[wvi]
            flux_net_ir += (flux_plus[:,wvi]-flux_minus[:,wvi]) * dwni[wvi]

            #pdb.set_trace()
            flux_plus_ir[:,wvi] += flux_plus[:,wvi] * dwni[wvi]
            flux_minus_ir[:,wvi] += flux_minus[:,wvi] * dwni[wvi]
        """
        print('debug fluxes in get_fluxes', temperature)
        for wvi in range(nwno):
            for il in range(len(flux_plus_midpt[:,0])):
                print(wvi, dwni[wvi],flux_plus_midpt[il,wvi],flux_minus_midpt[il,wvi] )
        """

        #if full output is requested add in flux at top for 3d plots
    
    return flux_net_v_layer, flux_net_v, flux_plus_v, flux_minus_v , flux_net_ir_layer, flux_net_ir, flux_plus_ir, flux_minus_ir
    
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
    
@jit(nopython=True, cache=True)
def blackbody_integrated(T, wave, dwave):
    """
    This computes the total energey per wavenumber bin needed for the climate calculation 
    Note that this is different than the raw flux at an isolated wavenumber. Therefore this function is 
    different than the blackbody function in `picaso.fluxes` which computes blackbody in raw 
    cgs units. 
    
    Parameters 
    ----------
    T : float, array 
        temperature in Kelvin 
    wave : float, array 
        wavenumber in cm-1 
    dwave : float, array 
        Wavenumber bins in cm-1 
    
    Returns 
    -------
    array 
        num temperatures by num wavenumbers 
        units of ergs/cm*2/s/cm-1 for *integrated* bins ()
    """

    h = 6.62607004e-27 # erg s 
    c = 2.99792458e+10 # cm/s
    k = 1.38064852e-16 #erg / K
    
    #h = 6.62607004e-34 # J s 
    #c = 2.99792458e+8 # m/s
    #k = 1.38064852e-23 #J / K
    
    
    
    c1 = 2*h*c**2
    c2 = h*c/k
    
    #this number was tested for accuracy against the original number of bins (4)
    #nbb 1 create three wavenumber bins (one on either side of center)
    #It achieves <1% integration accuracy up to black bodies ~50 K for the 
    #legacy 196 and 661 (for 661 max error is only 1e-3%) wavenumber grids. 
    nbb = 1 

    num_wave = len(wave)
    num_T = len(T)

    planck_sum = zeros((num_T, num_wave))

    for i in range(num_wave):
        for j in range(num_T):
            for k in range(-nbb, nbb + 1, 1):
                wavenum = wave[i] + k * dwave[i] / (2.0 * nbb)
                #erg/s/cm2/(cm-1)
                planck_sum[j, i] += c1 * (wavenum**3) / (exp(c2 * wavenum / T[j])-1)
                
    planck_sum /= (2 * nbb + 1.0) 

    
    #pdb.set_trace()
    #print('BB')
    
    return planck_sum*1e-7*1e4


@jit(nopython=True, cache=True)
def blackbody(t,w):
    """
    Blackbody flux in cgs units in per unit wavelength (cm)

    Parameters
    ----------
    t : array,float
        Temperature (K)
    w : array, float
        Wavelength (cm)
    
    Returns
    -------
    ndarray with shape ntemp x numwave in units of erg/cm/s2/cm
    """
    h = 6.62607004e-27 # erg s 
    c = 2.99792458e+10 # cm/s
    k = 1.38064852e-16 #erg / K

    return ((2.0*h*c**2.0)/(w**5.0))*(1.0/(exp((h*c)/outer(t, w*k)) - 1.0)) #* (w*w)


atm.T=np.ones(60)*500
extinctCoef0=atm.opac['extinctCoef'].copy()
scatCoef0=atm.opac['scatCoef'].copy()


pressure=atm.p#*10
temperature=atm.T
#dwni=1e4/(atm.wave)
dwni=rad.convertWave(atm.wave,'um','cm**-1')

w=-1
if w == -1:
    atm.omega = scatCoef0/extinctCoef0

else:
    atm.omega = w*np.ones_like(extinctCoef0, dtype=atm.numerical_precision)

extinctCoef = 0.5*(extinctCoef0[:atm.nLay-1,:]+extinctCoef0[1:atm.nLay,:])
scatCoef = 0.5*(scatCoef0[:atm.nLay-1,:]+scatCoef0[1:atm.nLay,:]) 
    
    
nlayer=atm.nLay-1
    
dz   = -np.diff(atm.z)
DTAU = 0.5*(extinctCoef0[:nlayer,:]+extinctCoef0[1:atm.nLay,:]) * np.outer(dz,np.ones(atm.nWave, dtype=atm.numerical_precision))     
    
TAU=np.vstack([np.zeros(atm.nWave, dtype=atm.numerical_precision),np.cumsum(DTAU,axis=0)])
# Ensure taugrid > 0
TAU*=(1+(np.arange(1,(atm.nLay+1))[:,np.newaxis])*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-10)+(np.arange(1,(atm.nLay+1))[:,np.newaxis]*np.ones([1,atm.nWave], dtype=atm.numerical_precision)*1e-99)
#DTAU*=0

#TAU=np.vstack([np.zeros(atm.nWave, dtype=atm.numerical_precision),np.cumsum(DTAU,axis=0)])
    # Ensure taugrid > 0

W0 = 0.5*(atm.omega[:atm.nLay-1,:]+atm.omega[1:atm.nLay,:])
#W0*W0
#atm.omega=W0

COSB=0
ftau_cld=1
surf_reflect=0

ngauss=5

gangle, gweight, tangle,tweight=disco.get_angles_1d(ngauss)
ng=ngauss
nt=1
phase_angle=0 #90 #0.9553166699 #0.5773502691896258 #0
ubar0, ubar1, cos_theta ,latitude,longitude=disco.compute_disco(ng, nt, gangle, tangle, phase_angle)

F0PI=atm.IrradStar

#wno=1e4/(atm.wave) #2*pi?
wno=rad.convertWave(atm.wave,'um','cm**-1')
pdb.set_trace()
#wno=atm.wave
nwno=atm.nWave
nlevel=atm.nLay

#gauss_wts=[0.015747,0.073908,0.146386,0.167174,0.096781]
gauss_wts=gweight



flux_net_v_layer, flux_net_v, flux_plus_v, flux_minus_v , flux_net_ir_layer, flux_net_ir, flux_plus_ir, flux_minus_ir=get_fluxes(pressure, temperature, dwni, DTAU, TAU, W0, COSB, ftau_cld, surf_reflect, ubar0,ubar1, F0PI,
            wno,nwno,ng,nt,gweight,tweight, nlevel, ngauss, gauss_wts,reflected=True, thermal=False)  
            
            
flux_up=flux_plus_ir+flux_plus_v[0][0]
pdb.set_trace()

flux_down=flux_minus_ir+flux_minus_v[0][0]
            
plt.plot(atm.wave,flux_up[0])
plt.title('Upwards Flux Top Layer')
#plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
#plt.ylim(-1,5000000)
plt.savefig('toon_clim/clim_toon_up_top.png')
plt.clf()

plt.plot(atm.wave,flux_down[0])
plt.title('Downwards Flux Top Layer')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
#plt.ylim(-1,500000)
plt.savefig('toon_clim/clim_toon_down_top.png')
plt.clf()

plt.plot(atm.wave,flux_up[30])
plt.title('Upwards Flux Layer 30')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
#plt.ylim(-1,2500)
plt.savefig('toon_clim/clim_toon_up_mid.png')
plt.clf()

plt.plot(atm.wave,flux_down[30])
plt.title('Downwards Layer 30')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
#plt.ylim(-1,500000)
plt.savefig('toon_clim/clim_toon_down_mid.png')
plt.clf()

plt.plot(atm.wave,flux_up[0]-flux_up[30])
plt.title('flux_up[0]-flux_up[30]')
plt.xscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
#plt.ylim(-30,30)
plt.savefig('toon_clim/clim_toon_up_test.png')
plt.clf()

print(flux_up[0])
result_toon=np.trapz(x=atm.wave,y=flux_up[0])
result_toon=np.trapz(x=wno,y=flux_up[0])
print(result_toon)            

            
            
            
            
              