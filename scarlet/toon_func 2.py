import numexpr as ne
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
extinctCoef=atm.opac['extinctCoef'].copy()
scatCoef=atm.opac['scatCoef'].copy()
extinctCoef = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:])
scatCoef = 0.5*(scatCoef[:atm.nLay-1,:]+scatCoef[1:atm.nLay,:]) 

def multiScatToon(IrradStar,extinctCoef,scatCoef,asym = 0,u0 = 0.5773502691896258,refl = False,thermal = True,calcJ = False,intTopLay=0,method ='Hemispheric Mean',w = -1):
    '''
    Computes spectrum with nonhomogenous scattering
    --------------------------
    multiScatToon(asymmetry factor,u,IrradStar,tau,method)
    Intensity at top of layer for thermal spectrum default at 0
    Returns Fupw, Fdwn, and Fnet for every layer of atm
    '''
     
    if w == -1:
        atm.omega = scatCoef/extinctCoef

    else:
        atm.omega = w*np.ones_like(extinctCoef, dtype=atm.numerical_precision)
        
        
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

    atm.Fs = IrradStar
    atm.u0 = u0
    [l,d] = np.shape(atm.omega)
        
    tauc=np.vstack([np.zeros(len(atm.wave), dtype=atm.numerical_precision),np.zeros(len(atm.wave), dtype=atm.numerical_precision),np.cumsum(taun,axis=0)])
    tauc = np.delete(tauc,(l+1),axis=0)
    tauc[0,:] = np.nan
        
    taun=np.vstack([np.zeros(len(atm.wave)),taun])
    taun[0,:] = np.nan
        
    atm.omega = np.vstack([np.zeros(len(atm.wave), dtype=atm.numerical_precision),atm.omega])
    atm.omega[0,:] = np.nan
    
        
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
        
    atm.g4 = 1 - atm.g3
        
    atm.lmbda = (atm.g1**2 - atm.g2**2)**(1/2)
    atm.gamma = atm.g2/(atm.g1 + atm.lmbda)
        
    e1 = 1 + atm.gamma*np.exp(-atm.lmbda*taun)
    e2 = 1 - atm.gamma*np.exp(-atm.lmbda*taun)
    e3 = atm.gamma + np.exp(-atm.lmbda*taun)
    e4 = atm.gamma - np.exp(-atm.lmbda*taun)
    pdb.set_trace()
        
    # Initialize tridiagonal matrix (TDM)
    A = np.zeros([2*l+1,d], dtype=atm.numerical_precision)
    B = np.zeros([2*l+1,d], dtype=atm.numerical_precision)
    D = np.zeros([2*l+1,d], dtype=atm.numerical_precision)
    E = np.zeros([2*l+1,d], dtype=atm.numerical_precision)
        
    # Insert nan values for easier & clearer indexing (matches Toon et al. 1989 paper)
    A[0,:] = np.nan
    B[0,:] = np.nan
    D[0,:] = np.nan
    E[0,:] = np.nan
        
    A[1,:] = 0
    B[1,:] = e1[1,:]
    D[1,:] = -e2[1,:]
    E[1,:] = -CM(tauc[1,:],0,1)
        
    # Build TDM
    for j in range(1,l):
        k = 2*j+1
        A[k] = e2[j,:]*e3[j,:] - e4[j,:]*e1[j,:]
        B[k] = e1[j,:]*e1[j+1,:] - e3[j,:]*e3[j+1,:]
        D[k] = e3[j,:]*e4[j+1,:] - e1[j,:]*e2[j+1,:]
        E[k] = e3[j,:]*(CP(tauc[j+1,:],0,j+1) - CP(tauc[j,:],taun[j,:],j)) - e1[j,:]*(CM(tauc[j+1,:],0,j+1)-CM(tauc[j,:],taun[j,:],j)) 
        #print j,k
        # check indicies

    for j in range(1,l):
        k = 2*j
        A[k] = e2[j+1,:]*e1[j,:] - e3[j,:]*e4[j+1,:]
        B[k] = e2[j,:]*e2[j+1,:] - e4[j,:]*e4[j+1,:]
        D[k] = e1[j+1,:]*e4[j+1,:] - e2[j+1,:]*e3[j+1,:]
        E[k] = e2[j+1,:]*(CP(tauc[j+1,:],0,j+1) - CP(tauc[j,:],taun[j,:],j)) - e4[j+1,:]*(CM(tauc[j+1,:],0,j+1)-CM(tauc[j,:],taun[j,:],j))
        #print j,k
        # check indicies
    pdb.set_trace()
        
    # Last terms of TDM
    k = 2*l
    A[k] = e1[l,:] - GroundAlbedo*e3[l,:]
    B[k] = e2[l,:] - GroundAlbedo*e4[l,:]
    D[k] = 0
    E[k] = GroundAlbedo*atm.u0*np.exp(-tauc[l,:]/atm.u0)*np.pi*atm.Fs - CP(tauc[l,:],taun[l,:],l) + GroundAlbedo*CM(tauc[l,:],taun[l,:],l)
        
    # Delete nan row to correctly solve
    A = np.delete(A,0,axis=0)
    B = np.delete(B,0,axis=0)
    D = np.delete(D,0,axis=0)
    E = np.delete(E,0,axis=0)
        
    # Solve with tridiagonal matrix algorithm solver
    x=atm.TDMAsolver(A,B,D,E)
    pdb.set_trace()
    solution = np.vstack([np.zeros(len(atm.wave)),atm.TDMAsolver(A,B,D,E)], dtype=atm.numerical_precision)
    solution[0] = np.nan
    atm.solution = solution
            
    # Reflected solar light
    if refl:
        
        #tautest =  0
        # Upward Flux
        FupwSOL = np.zeros([l,d])
#        FP1 = solution[1,:]*(np.exp(-atm.lmbda[1,:]*(taun[1,:]-tautest)) + atm.gamma[1,:]*np.exp(-atm.lmbda[1,:]*tautest))
#        FP2 = solution[2,:]*(np.exp(-atm.lmbda[1,:]*(taun[1,:]-tautest)) - atm.gamma[1,:]*np.exp(-atm.lmbda[1,:]*tautest))
#        Fupw = FP1 + FP2 + CP(tauc[1,:],tautest,1)
        for i in range(1,l+1):
            k = 2*i-1
            FP1 = solution[k,:]*(np.exp(-atm.lmbda[i,:]*(0)) + atm.gamma[i,:]*np.exp(-atm.lmbda[i,:]*taun[i,:]))
            FP2 = solution[k+1,:]*(np.exp(-atm.lmbda[i,:]*(0)) - atm.gamma[i,:]*np.exp(-atm.lmbda[i,:]*taun[i,:]))
            FupwSOL[i-1,:] = FP1 + FP2 + CP(tauc[i,:],taun[i,:],i)
            
        # Downward Flux
        FdwnSOL = np.zeros([l,d], dtype=atm.numerical_precision)
        DirSOL = np.zeros([l,d], dtype=atm.numerical_precision)

        for i in range(1,l+1):
            k = 2*i-1
            FM1 = solution[k,:]*(atm.gamma[i,:]*np.exp(-atm.lmbda[i,:]*(0)) + np.exp(-atm.lmbda[i,:]*taun[i,:]))
            FM2 = solution[k+1,:]*(atm.gamma[i,:]*np.exp(-atm.lmbda[i,:]*(0)) - np.exp(-atm.lmbda[i,:]*taun[i,:]))
            DirSOL[i-1,:] = atm.u0*np.pi*atm.Fs*np.exp(-(tauc[i,:]+taun[i,:])/atm.u0)
            FdwnSOL[i-1,:] = FM1 + FM2 + CM(tauc[i,:],taun[i,:],i) + DirSOL[i-1,:]
                
        # Net Flux
        FnetSOL = np.zeros([l,d], dtype=atm.numerical_precision)
        for i in range(1,l+1):
            k = 2*i-1               
            FnetSOL[i-1,:] = solution[k,:]*(e1[i,:]-e3[i,:])+solution[k+1,:]*(e2[i,:]-e4[i,:])+ CP(tauc[i],taun[i],i) - CM(tauc[i],taun[i],i)        
        # FnetSOL = FupwSOL - FdwnSOL
        
        
    # Source Function Technique
    if thermal:    
            
        Tgrid = np.zeros([l+2], dtype=atm.numerical_precision)
        Tgrid[0] = np.nan
        Tgrid[1] = atm.T[0]
        for i in range(2,l+1):
            Tgrid[i] = (atm.T[i-2] + atm.T[i-1])/2
        Tgrid[-1] = atm.T[-1]
                            
        B0 = np.zeros([l+1,d], dtype=atm.numerical_precision) ;  B0[0,:] = np.nan
        B1 = np.zeros([l+1,d], dtype=atm.numerical_precision) ;  B1[0,:] = np.nan
        for i in range (1,l+1):
            B0[i,:] = rad.PlanckFct(Tgrid[i],atm.wave,'um','W/(m**2*um)','rad')
            B1[i,:] = (rad.PlanckFct(Tgrid[i+1],atm.wave,'um','W/(m**2*um)','rad') - B0[i,:])/taun[i,:]
                            
        # Define Hemispheric Mean 2 Stream Source Function parameters
        paramG = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   paramG[0,:] = np.nan
        paramK = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   paramK[0,:] = np.nan
        paramH = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   paramH[0,:] = np.nan
        paramJ = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   paramJ[0,:] = np.nan
        a1 = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   a1[0,:] = np.nan
        a2 = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   a2[0,:] = np.nan
        o1 = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   o1[0,:] = np.nan
        o2 = np.zeros([l+1,d], dtype=atm.numerical_precision)  ;   o2[0,:] = np.nan
            
        for i in range(1,l+1):
            j = 2*i-1
    
            paramG[i,:] = (solution[j,:] + solution[j+1,:]) * (1/u1 - atm.lmbda[i,:])
            paramK[i,:] = (solution[j,:] - solution[j+1,:]) * (1/u1 - atm.lmbda[i,:])
            paramH[i,:] = (solution[j,:] - solution[j+1,:]) * atm.gamma[i,:] * (atm.lmbda[i,:] + 1/u1)
            paramJ[i,:] = (solution[j,:] + solution[j+1,:]) * atm.gamma[i,:] * (atm.lmbda[i,:] + 1/u1)
            a1[i,:] = (B0[i,:] + B1[i,:]*(1/(atm.g1[i,:] + atm.g2[i,:])-u1)) 
            o1[i,:] = (B0[i,:] - B1[i,:]*(1/(atm.g1[i,:] + atm.g2[i,:])-u1)) 
            #pdb.set_trace()               
           
        a2 = B1
        o2 = B1
            
        # Intensity downwards (through bottom of layer)
        IdwnIR = np.zeros([l,d], dtype=atm.numerical_precision)
            
        # Intensity at the bottom of the top layer
        IM1 = intTopLay*np.exp(-taun[1,:]/u0)
        IM2 = paramJ[1,:]/(atm.lmbda[1,:]*u0+1) * (1-np.exp(-taun[1,:]*(atm.lmbda[1,:]+1/u0)))
        IM3 = paramK[1,:]/(atm.lmbda[1,:]*u0-1) * (np.exp(-taun[1,:]/u0)-np.exp(-taun[1,:]*atm.lmbda[1,:]))
        IM4 = o1[1,:]*(1-np.exp(-taun[1,:]/u0))
        IM5 = o2[1,:]*(u0*np.exp(-taun[1,:]/u0)+taun[1,:]-u0)
                    
        for j in range(1,l):
            i = j+1
            IM1 = IdwnIR[j-1,:]*np.exp(-taun[i,:]/u0)
            IM2 = paramJ[i,:]/(atm.lmbda[i,:]*u0+1) * (1-np.exp(-taun[i,:]*(atm.lmbda[i,:]+1/u0)))
            IM3 = paramK[i,:]/(atm.lmbda[i,:]*u0-1) * (np.exp(-taun[i,:]/u0)-np.exp(-taun[i,:]*atm.lmbda[i,:]))
            IM4 = o1[i,:]*(1-np.exp(-taun[i,:]/u0))
            IM5 = o2[i,:]*(u0*np.exp(-taun[i,:]/u0)+taun[i,:]-u0)
            IdwnIR[j,:] = IM1 + IM2 + IM3 + IM4 + IM5
            #pdb.set_trace()
                    
            
        # Surface Boundary
        k = l
#            IM1 = IdwnIR[l-1,:]*np.exp(-taun[k,:]/u0)
#            IM2 = paramJ[k,:]/(atm.lmbda[k,:]*u0+1) * (1-np.exp(-taun[k,:]*(atm.lmbda[k,:]+1/u0)))
#            IM3 = paramK[k,:]/(atm.lmbda[k,:]*u0-1) * (np.exp(-taun[k,:]/u0)-np.exp(-taun[k,:]*atm.lmbda[k,:]))
#            IM4 = o1[k,:]*(1-np.exp(-taun[k,:]/u0))
#            IM5 = o2[k,:]*(u0*np.exp(-taun[k,:]/u0)+taun[k,:]-u0)
#            ISurf = IM1 + IM2 + IM3 + IM4 + IM5
        ISurf = IdwnIR[-1,:]
            
        IupwIR = np.zeros([l,d], dtype=atm.numerical_precision)
        IP1 = ISurf*np.exp(-taun[k,:]/u0)
        IP2 = paramG[k,:]/(atm.lmbda[k,:]*u0-1) * (np.exp(-taun[k,:]/u0)-np.exp(-taun[k,:]*atm.lmbda[k,:]))
        IP3 = paramH[k,:]/(atm.lmbda[k,:]*u0+1) * (1-np.exp(-taun[k,:]*(atm.lmbda[k,:]+1/u0)))
        IP4 = a1[k,:]*(1-np.exp(-taun[k,:]/u0))
        IP5 = a2[k,:]*(u0-(taun[k,:]+u0)*np.exp(-taun[k,:]/u0))
        IupwIR[-1,:] = IP1 + IP2 + IP3 + IP4 + IP5
            
        
        # Flux upwards (through top of layer)
        for i in range(l-2,-1,-1):
            k = i+1
            IP1 = IupwIR[i+1,:]*np.exp(-taun[k,:]/u0)
            IP2 = paramG[k,:]/(atm.lmbda[k,:]*u0-1) * (np.exp(-taun[k,:]/u0)-np.exp(-taun[k,:]*atm.lmbda[k,:]))
            IP3 = paramH[k,:]/(atm.lmbda[k,:]*u0+1) * (1-np.exp(-taun[k,:]*(atm.lmbda[k,:]+1/u0)))
            IP4 = a1[k,:]*(1-np.exp(-taun[k,:]/u0))
            IP5 = a2[k,:]*(u0-(taun[k,:]+u0)*np.exp(-taun[k,:]/u0))                        
            
            IupwIR[i,:] = IP1 + IP2 + IP3 + IP4 + IP5 
        pdb.set_trace()    
            
            
            
            
            
           #  plt.plot(atm.wave,IP1)
#             plt.title('IP1 (Top Layer)')
#             plt.xlabel('Wavelength')
#             plt.ylabel('IP1')
#             #plt.xlim(0.9,1.1)
#             plt.ylim(-1,1)
#             plt.xscale('log')
#             plt.savefig(f'allplots/IP1_{i}.png')
#             plt.clf()
#         
#             plt.plot(atm.wave,IP2)
#             plt.title('IP2 (Top Layer)')
#             plt.xlabel('Wavelength')
#             plt.ylabel('IP2')
#             #plt.xlim(0.9,1.1)
#             plt.ylim(-1,1)
#             plt.xscale('log')
#             plt.savefig(f'allplots/IP2_{i}.png')
#             plt.clf()
#         
#             plt.plot(atm.wave,IP3)
#             plt.title('IP3 (Top Layer)')
#             plt.xlabel('Wavelength')
#             plt.ylabel('IP3')
#             #plt.xlim(0.9,1.1)
#             plt.ylim(-1,1)
#             plt.xscale('log')
#             plt.savefig(f'allplots/IP3_{i}.png')
#             plt.clf()
#             
#             plt.plot(atm.wave,IP4)
#             plt.title('IP4 (Top Layer)')
#             plt.xlabel('Wavelength')
#             plt.ylabel('IP4')
#             #plt.xlim(0.9,1.1)
#             plt.ylim(-1,1)
#             plt.xscale('log')
#             plt.savefig(f'allplots/IP4_{i}.png')
#             plt.clf()
#             
#             plt.plot(atm.wave,IP5)
#             plt.title('IP5 (Top Layer)')
#             plt.xlabel('Wavelength')
#             plt.ylabel('IP5')
#             #plt.xlim(0.9,1.1)
#             plt.ylim(-1,1)
#             plt.xscale('log')
#             plt.savefig(f'allplots/IP5_{i}.png')
#             plt.clf()            
            
            
            
            
            
            
            
            
#             plt.plot(atm.wave,IupwIR[58])
#             plt.title('IupwIR (Layer 58)')
#             plt.xlabel('Wavelength')
#             plt.ylabel('IupwIR')
#             #plt.xlim(0.9,1.1)
#             plt.ylim(-1,1)
#             plt.xscale('log')
#             plt.savefig(f'allplots/IupwIR_{i}.png')
#             plt.clf()
                
        # Convert
        FdwnIR = IdwnIR*np.pi
        FupwIR = IupwIR*np.pi
        FnetIR = FdwnIR + FupwIR
        
        plt.plot(atm.wave,IupwIR[50])
        plt.title('IupwIR (Layer 40)')
        plt.xlabel('Wavelength')
        plt.ylabel('IupwIR')
        #plt.xlim(0.9,1.1)
        plt.ylim(-500,500)
        plt.xscale('log')
        plt.savefig(f'IupwIR.png')
        plt.clf()
            
    if calcJ:
        JSOL = np.zeros([l,d], dtype=atm.numerical_precision)
        DirSOL = np.zeros([l,d], dtype=atm.numerical_precision)
                
        for i in range(1,l+1):
            k = 2*i-1      
            DirSOL[i-1,:] = atm.u0*np.pi*atm.Fs*np.exp(-(tauc[i,:]+taun[i,:])/atm.u0)
            JSOL[i-1,:] = (solution[k,:]*(e1[i,:]+e3[i,:])+solution[k+1,:]*(e2[i,:]+e4[i,:])+ CP(tauc[i],taun[i],i) - CM(tauc[i],taun[i],i) + DirSOL[i-1,:]/u0)/(4*np.pi*u1)   
            
        JIR = np.zeros([l,d], dtype=atm.numerical_precision)
        JIR[0,:] = IupwIR[1,:] - IupwIR[0,:] + intTopLay - IdwnIR[0,:]
            
        for i in range (1,l-1):
            JIR[i,:] = IupwIR[i+1,:] - IupwIR[i,:] + IdwnIR[i-1,:] - IdwnIR[i,:]
        JIR[-1,:] = sigmaSB*atm.params['Tint']**4/np.pi + IdwnIR[-2,:] - IupwIR[-1,:]
            
        J = JSOL + JIR
        return J
            
    elif refl and thermal:
        
        plt.plot(atm.wave,FupwSOL[0])
        plt.title('FupwSOL (Top Layer)')
        plt.xlabel('Wavelength')
        plt.ylabel('FupwSOL')
        #plt.xlim(0.9,1.1)
        plt.ylim(-100000,1400000)
        plt.xscale('log')
        plt.savefig('FupwSOL.png')
        plt.clf()
        
        plt.plot(atm.wave,FupwIR[0])
        plt.title('FupwIR (Top Layer)')
        plt.xlabel('Wavelength')
        plt.ylabel('FupwIR')
        #plt.xlim(0.9,1.1)
        plt.ylim(-1000,1000)
        plt.xscale('log')
        plt.savefig('FupwIR.png')
        plt.clf()
        
        plt.plot(atm.wave,FdwnSOL[0])
        plt.title('FdwnSOL (Top Layer)')
        plt.xlabel('Wavelength')
        plt.ylabel('FdwnSOL')
        #plt.xlim(0.9,1.1)
        plt.ylim(-1000,3000000)
        plt.xscale('log')
        plt.savefig('FdwnSOL.png')
        plt.clf()
        
        plt.plot(atm.wave,FdwnIR[30])
        plt.xlabel('Wavelength')
        plt.ylabel('FdwnIR')
        plt.title('FdwnIR (Layer 30)')
        #plt.xlim(0.9,1.1)
        plt.ylim(-1000,1000)
        plt.xscale('log')
        plt.savefig('FdwnIR.png')
        plt.clf()
        
        Fupw = FupwSOL + FupwIR
        Fdwn = FdwnSOL + FdwnIR
        atm.DirSOL = DirSOL #dbg
        atm.Fupw = Fupw
        atm.Fdwn = Fdwn
            
        Fnet = np.zeros([l,d], dtype=atm.numerical_precision)
        Fnet[0,:] = Fupw[1,:] - Fupw[0,:] + intTopLay*np.pi - Fdwn[0,:] + DirSOL[0,:]
        for i in range (1,l-1):
            Fnet[i,:] = Fupw[i+1,:] - Fupw[i,:] + Fdwn[i-1,:] - Fdwn[i,:] #+ atm.u0*np.pi*atm.Fs*np.exp(-(tauc[i+1,:]+taun[i+1,:])/atm.u0)
        Fnet[-1,:] = sigmaSB*atm.params['Tint']**4 + Fdwn[-2,:] - Fupw[-1,:] #+ atm.u0*np.pi*atm.Fs*np.exp(-(tauc[-1,:]+taun[-1,:])/atm.u0)

        return Fupw, Fdwn, Fnet
    elif refl:
        return FupwSOL, FdwnSOL, FnetSOL
    elif thermal:
        return FupwIR, FdwnIR, FnetIR
        

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


toon= multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = True,thermal = True, calcJ = False, intTopLay=0, w=-1)

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
