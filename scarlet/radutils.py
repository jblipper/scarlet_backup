# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:42:17 2016

@author: bbenneke
"""
from __future__ import print_function, division, absolute_import, unicode_literals
from auxbenneke.constants import c, h, au, Rsun, amu, Rconst
import numpy as np
#from scipy.signal import butter, lfilter
import auxbenneke.utilities as ut

from scipy.special import expn
import pdb


def convertWaveToSi(inp,unit1):

    # WavelengthUnit
    if unit1=='Hz':     #Frequency
        f=inp
        wave=c/f
        wavenumber=1/wave
    elif unit1=='m':    #Wavelength
        wave=inp
        f=c/wave
        wavenumber=1/wave
    elif unit1=='cm':    #Wavelength
        wave=inp*1e-2
        f=c/wave
        wavenumber=1/wave
    elif unit1=='um':
        wave=inp*1e-6
        f=c/wave
        wavenumber=1/wave
    elif unit1=='nm':
        wave=inp*1e-9
        f=c/wave
        wavenumber=1/wave
    elif unit1=='A':
        wave=inp*1e-10
        f=c/wave
        wavenumber=1/wave
    elif unit1=='m**-1': #Wavenumber
        wavenumber=inp
        f=wavenumber*c
        wave=1/wavenumber
    elif unit1=='cm**-1':
        wavenumber=inp*1e2
        f=wavenumber*c
        wave=1/wavenumber
    else:
        print('Input Unit Error!!!')

    ##Now wave is in meters
    ##Now f is in 1/s
    ##Now wavenumber is in meters**-1

    return wave, f, wavenumber


def convertWave(inp,unit1,unit2):

    wave, f, wavenumber = convertWaveToSi(inp,unit1)

    ##Now wave is in meters
    ##Now f is in 1/s
    ##Now wavenumber is in meters**-1

    # WavelengthUnit
    if unit2=='Hz':     #Frequency
        output = f
    elif unit2=='m':    #Wavelength
        output = wave
    elif unit2=='um':
        output=wave*1e6
    elif unit2=='nm':
        output=wave*1e9
    elif unit2=='A':
        output=wave*1e10
    elif unit2=='m**-1': #Wavenumber
        output=wavenumber
    elif unit2=='cm**-1':
        output=wavenumber*1e-2
    else:
        print('Output Unit Error!!!')

    return output



def convertIntensity(Iin,LambdaInput,InputUnit='W/(m**2*Hz)',WavelengthUnit='um',OutputUnit='W/(m**2*um)'):
    ##Unit converter for flux and intensity
    ##Ex: Iout = ConvertIntensityUnits(Iin,'W/(m**2*Hz)','W/(m**2*um)',wave,'m'
    ##    plot(wave*1e6,ConvertIntensityUnits(IncidentFlux,'W/(m**2*Hz)','W/(m**2*um)',wave,'m'
    ##    trapz(wave*1e6,ConvertIntensityUnits(IncidentFlux,'W/(m**2*Hz)','W/(m**2*um)',wave,'m'


    wave, f, wavenumber = convertWaveToSi(LambdaInput,WavelengthUnit)


    ##Now wave is in meters
    ##Now f is in 1/s
    ##Now wavenumber is in meters**-1


    #Convert Iin into 'W/(m**2*Hz)'
    if InputUnit=='W/(m**2*Hz)':                                  #Frequency Bin
        I=Iin
    elif InputUnit=='Jy':                                  #Frequency Bin
        I=Iin  /  (1e26)
    elif InputUnit=='W/(m**2*m)':                               #Wavelength Bin
        I=Iin  /  (c/(wave**2))
    elif InputUnit=='W/(m**2*um)':
        I=Iin  /  (c/(wave**2)  *1e-6)    #checked!!!
    elif InputUnit=='W/(m**2*nm)':
        I=Iin  /  (c/(wave**2)  *1e-9)    #checked!!!
    elif InputUnit=='W/(m**2*A)':
        I=Iin  /  (c/(wave**2)  *1e-10)    #checked!!!

    elif InputUnit=='(ergs/s)/(m**2*A)':
        I=Iin  /  (c/(wave**2)  *1e-10  * 1e7)

    elif InputUnit=='(ergs/s)/(cm**2*A)':
        I=Iin  /  (c/(wave**2) *1e-10  * 1e7  *1e-4)
        
    elif InputUnit=='(ergs/s)/(cm**2*cm)':                                  #added
        I=Iin  /  (c/(wave**2) *1e-10  * 1e7  *1e-4  *10*1e3*1e3*10  )       #added

    elif InputUnit=='W/(m**2*m**-1)':   #= W*m**-1 = W/m          #Wavenumber Bin
        I=Iin  /  (c)
    elif InputUnit=='W/(cm**2*cm**-1)':   #= W*cm**-1 = W/cm
        I=Iin  /  (c  *  1e-2)   #checked
    elif InputUnit=='W/(m**2*cm**-1)':
        I=Iin  /  (c  *  1e-2  *  1e4)
    elif InputUnit=='(photons/s)/(m**2*Hz)':     #-----Photon count------
        I=Iin  /  (1/(h*f))
    elif InputUnit=='(photons/s)/(m**2*um)':     #-----Photon count------
        I=Iin  /  (c/(wave**2)  *1e-6  /(h*f))
    elif InputUnit=='(photons/s)/(m**2*cm**-1)':     #-----Photon count------
        I=Iin  /  (c  *  1e-2  *  1e4    /(h*f))
    elif InputUnit=='(photons/s)/(cm**2*A)':     #-----Photon count------  #checked
        I=Iin  /  ((c/(wave**2)  *1e-6  /(h*f)) *1e-4 *1e-4)
    elif InputUnit=='(photons/s)/(m**2*cm**-1)':     #-----Photon count------ #checked
        I=Iin  /  (c  *  1e-2  *  1e4    /(h*f))
    else:
        print('Input Unit Error!!!')

    # Now:   I is in 'W/(m**2*Hz)'


    #Convert I into selected Output Unit
    if OutputUnit=='W/(m**2*Hz)':                                  #Frequency Bin
        Iout=I
    elif OutputUnit=='Jy':                                  #Frequency Bin
        Iout=I  *  (1e26)
    elif OutputUnit=='W/(m**2*m)':                               #Wavelength Bin
        Iout=I  *  (c/(wave**2))
    elif OutputUnit=='W/(m**2*um)':
        Iout=I  *  (c/(wave**2)  *1e-6)    #checked!!!
    elif OutputUnit=='W/(m**2*nm)':
        Iout=I  *  (c/(wave**2)  *1e-9)    #checked!!!
    elif OutputUnit=='W/(m**2*A)':
        Iout=I  *  (c/(wave**2)  *1e-10)    #checked!!!
    # elif OutputUnit=='W/(m**2*DeltaLogLambda800)'                               #Wavelength Bin
    #     R=800
    #     Iout=I  *  (c/(wave)/log(R+1))


    elif OutputUnit=='(ergs/s)/(m**2*A)':
        Iout=I  *  (c/(wave**2)  *1e-10  * 1e7)
    elif OutputUnit=='(ergs/s)/(cm**2*A)':         #checked!!!
        Iout=I  *  (c/(wave**2) *1e-10  * 1e7  *1e-4)

    elif OutputUnit=='W/(m**2*m**-1)':   #= W*m**-1 = W/m          #Wavenumber Bin
        Iout=I  *  (c)
    elif OutputUnit=='W/(cm**2*cm**-1)':   #= W*cm**-1 = W/cm
        Iout=I  *  (c  *  1e-2)   #checked
    elif OutputUnit=='W/(m**2*cm**-1)':
        Iout=I  *  (c  *  1e-2  *  1e4)
    elif OutputUnit=='(photons/s)/(m**2*Hz)':     #-----Photon count------
        Iout=I  *  (1/(h*f))
    elif OutputUnit=='(photons/s)/(m**2*um)':     #-----Photon count------  #checked
        Iout=I  *  (c/(wave**2)  *1e-6  /(h*f))
    elif OutputUnit=='(photons/s)/(m**2*A)':     #-----Photon count------  #checked
        Iout=I  *  (c/(wave**2)  *1e-6  /(h*f)) *1e-4
    elif OutputUnit=='(photons/s)/(cm**2*A)':     #-----Photon count------  #checked
        Iout=I  *  (c/(wave**2)  *1e-6  /(h*f)) *1e-4 *1e-4
    elif OutputUnit=='(photons/s)/(m**2*cm**-1)':     #-----Photon count------ #checked
        Iout=I  *  (c  *  1e-2  *  1e4    /(h*f))
    elif OutputUnit=='(photons/s)/(cm**2*cm**-1)':     #-----Photon count------ #checked
        Iout=I  *  (c  *  1e-2  *  1e4    /(h*f)) *1e-4
    else:
        print('Output Unit Error!!!')


    return Iout





#%%

def PlanckFct(T,Input,InputUnit='um',OutputUnit='W/(m**2*um)',RadianceOrFlux='rad'):
    '''
    %Black radiator
    %
    %PlanckFct(T,Input,InputUnit,OutputUnit,RadianceOrFlux)
    % e.g. Bsurf=PlanckFct(TSurf,nu,'cm^-1','W/(m^2*Hz)','rad');
    %
    %T=temperature in Kelvin
    %f=Frequency/Wavelength/Wavenumer
    %
    %Options for InputUnit:
    %frequency : 'Hz'
    %wavelength: 'm'    'um'     'nm'
    %wavenumber: 'm^-1' 'cm^-1'
    %
    %Options for OutputUnit:
    %frequency bin : 'W/(m^2*Hz)'
    %wavelength bin: 'W/(m^2*m)'       'W/(m^2*um)'
    %wavenumber bin: 'W/(m^2*m^-1)'    'W/(cm^2*cm^-1)'
    %
    %Options for RadianceOrFlux:
    %'rad' : The output is the surface radiance / surface intensity
    %'flux': The output is the surface flux
    '''

    h=6.62606896e-34 #J*s
    c=299792458     #in m/s
    k=1.3806504e-23 #in J/K


    lam, f, wavenumber = convertWaveToSi(Input,InputUnit)


    Bf = (2*h*f**3)/(c**2)  *   1/( np.exp((h*f)/(k*T))-1)
    #Bf = (2*h*f[:,np.newaxis]**3)/(c**2)  *   1/( np.exp( (h/k)*(np.outer(f,1/T)  )  -1  ))

    #Adjust Output to selected Output Unit
    #    B=ConvertIntensityUnits(Bf,'W/(m^2*Hz)',OutputUnit,lam,'m');

#    if OutputUnit=='W/(m**2*um)':
#        B=Bf * (c/(lam**2) *1e-6)     #checked!!!

    #Adjust Output to Radiance or Flux
    if RadianceOrFlux=='rad':
        Bf=Bf
    elif RadianceOrFlux=='flux':
        Bf=np.pi*Bf
    else:
        print('Error in selecting between radiance and flux!!!')

    Bout = convertIntensity(Bf,f,InputUnit='W/(m**2*Hz)',WavelengthUnit='Hz',OutputUnit=OutputUnit)

    return Bout
    


def calcTBright(flux,wavelength,fluxUnit='W/(m**2*um)', WavelengthUnit='um'):
    '''
    Computes the Brightness Temperature out of the thermal flux received 
    
    The input flux must be in units of flux 
    Usually we will use self.thermal for the flux which is in W/m²um
    
    We can also specify the Flux and Wavelength units
    
    The output Brightness Temperature will always be in Kelvins (K)
    
    '''
    h=6.62606896e-34 #J*s
    c=299792458     #in m/s
    k=1.3806504e-23 #in J/K
    
    # convert wave to SI units 
    lam, f, wavenumber = convertWaveToSi(wavelength,WavelengthUnit)   # now [lam] = m, [f] = Hz, [wavenumber] = m⁻¹
    
    # We will want to work everything out in SI units so that we avoid unit mistakes
    # We will put the flux in SI units, namely W/m**2*m    
    flux = convertIntensity(flux,wavelength,fluxUnit,WavelengthUnit,OutputUnit='W/(m**2*m)')
    
    # We switch to spectral radiance by dividing by Pi, this is important
    flux = flux/np.pi
    
    # The units of flux are W/m**2*m and every length is now in meters
    # We can compute Tb(wavelength(m))
    Tbright = (h*c) / (k * lam * np.log(1 + (2 * h * c**2 / (flux * lam**5)  ) ) )
    
    
    return Tbright
    
    
def PlanckDerivative(lam,T):
    #wave is in m
    
    h=6.62606896e-34 #J*s
    c=299792458     #in m/s
    k=1.3806504e-23 #in J/K
    a=h*c/(lam*k)
    dBdT = (2*h*c**2) / lam**5   *   (a*np.exp(a/T)) /  (lam**2 * ( np.exp(a/T)-1 )**2 ) 
    return dBdT
    
    

    
#%%

def uniformResPower(resolution, start, end):
    '''
    This function create a numpy array of wavelength with uniform resolution (lambda/$\delat$ lambda)
    '''
    wavelength = np.array([start])
    while True:
        right = wavelength[-1] + wavelength[-1] / resolution 
        if right > end:
            break
        wavelength = np.hstack((wavelength, right))
    return wavelength

    
    
def mag2flux(mag,band='V',OutputUnit='W/(m**2*um)'):#'(photons/s)/(m**2*um)'): 
    '''
    Function FluxFromMag(Bandpass,mag) takes two arguments.
    Bandpass is a string, and mag is a float. e.g. Bandpass='V', mag=5.95. 
    It returns an array (Flux, PhotonFlux,lambdaCen,lambdaRange). 
    Flux is in W/(m^2*um), photon flux is in photons/(s*m^2*um), lambdaCen and lambdaRange are in um.   
    '''

    MagTable = {'U': (0.360000000000000,0.150000000000000,1810,'Bessel (1979)'),
                'B': (0.440000000000000,0.220000000000000,4260,'Bessel (1979) '),
                'V': (0.550000000000000,0.160000000000000,3640,'Bessel (1979) '),
                'R': (0.640000000000000,0.230000000000000,3080,'Bessel (1979) '),
                'I': (0.790000000000000,0.190000000000000,2550,'Bessel (1979) '),
                'J': (1.26000000000000,0.160000000000000,1600,'Campins, Reike, & Lebovsky (1985) '),
                'H':(1.60000000000000,0.230000000000000,1080,'Campins, Reike, & Lebovsky (1985) '),
                'K':(2.22000000000000,0.230000000000000,670,'Campins, Reike, & Lebovsky (1985) '),
                'Ks':(2.22000000000000,0.230000000000000,670,'Campins, Reike, & Lebovsky (1985) '),  #just copied
                'g':(0.520000000000000,0.140000000000000,3730,'Schneider, Gunn, & Hoessel (1983) '),
                'r':(0.670000000000000,0.140000000000000,4490,'Schneider, Gunn, & Hoessel (1983) '),
                'i':(0.790000000000000,0.160000000000000,4760,'Schneider, Gunn, & Hoessel (1983) '),
                'z':(0.910000000000000,0.130000000000000,4810,'Schneider, Gunn, & Hoessel (1983)')} 
                
    lambdaCen = MagTable[band][0] 
    lambdaRange = (lambdaCen - MagTable[band][1]/2., lambdaCen + MagTable[band][1]/2) 
    Janskies = MagTable[band][2] * 10**(-0.4*mag)        

    flux = Janskies * 1e-26   #W/(m^2*Hz)
    flux = convertIntensity(flux, lambdaCen, InputUnit='W/(m**2*Hz)', WavelengthUnit='um',OutputUnit=OutputUnit) 

#    DeltaLambdaOverLambda = 1. / lambdaCen 
#    PhotonFlux = Janskies * 1.51e7 * DeltaLambdaOverLambda   #(photons/s)/(m^2*um)

    return flux,lambdaCen,lambdaRange 
    
    
    
#%%

def plotBB(ax,T,wave,InputUnit='um',OutputUnit='W/(m**2*um)',label=None,**kwargs):
    if label is None:
        label='T = {:g} K'.format(T)
    flux = PlanckFct(T,wave,InputUnit='um',OutputUnit='W/(m**2*um)',RadianceOrFlux='flux')
    ax.plot(wave,flux,label=label,**kwargs)
    
    
#%%

def TpGrey(tau,Teff=1000):
    T = (  3.0/4.0 * Teff**4 * (tau + 2.0/3.0)  )**(1.0/4.0)
    return T

def TpGuillot2010(tau,Teff=300,Tirr=5700,gammaV=0.25,muStar=0.5):
    return 0
 
    
#def TpTwoVis(p,
#             Tint=100,kappaIR=3e-3,gamma1=0.158,gamma2=0.158,alpha=0.5,beta=1.0,
#             Rstar=0.756*Rsun,Teffstar=5040,
#             ap=0.031*au,gp=10**3.341 / 100): #m/s**2
#
#    tau = kappaIR * p / gp
#
#    Tirr = beta * (Rstar/ap/2)**0.5 * Teffstar
#    
#    zeta1= 2.0/3.0   +   2.0/3.0/gamma1*( 1+ (gamma1*tau/2 - 1) * np.exp(-gamma1*tau) )   +  2.0*gamma1/3.0 * ( 1- tau**2/2 ) *  expn(2,gamma1*tau)
#    zeta2= 2.0/3.0   +   2.0/3.0/gamma2*( 1+ (gamma2*tau/2 - 1) * np.exp(-gamma2*tau) )   +  2.0*gamma2/3.0 * ( 1- tau**2/2 ) *  expn(2,gamma2*tau)
#          
#    Term0 = 3.0/4.0 * Tint**4 * (tau + 2.0/3.0)
#    Term1 = 3.0/4.0 * Tirr**4 * (1-alpha) * zeta1
#    Term2 = 3.0/4.0 * Tirr**4 * alpha     * zeta2
#    
#    T = (Term0+Term1+Term2)**(1.0/4.0)
#    return T

    
def TpTwoVis(p,
             Tint=100,kappaIR=3e-2,gamma1=0.158,gamma2=0.158,alpha=0.5,beta=1.0,
             Rstar=0.756*Rsun,Teffstar=5040,
             ap=0.031*au,gp=10**3.341 / 100): #m/s**2

    tau = kappaIR * p / gp

    Tirr = beta * (Rstar/ap/2)**0.5 * Teffstar
    
    zeta1= 2.0/3.0   +   2.0/3.0/gamma1*( 1+ (gamma1*tau/2 - 1) * np.exp(-gamma1*tau) )   +  2.0*gamma1/3.0 * ( 1- tau**2/2 ) *  expn(2,gamma1*tau)
    zeta2= 2.0/3.0   +   2.0/3.0/gamma2*( 1+ (gamma2*tau/2 - 1) * np.exp(-gamma2*tau) )   +  2.0*gamma2/3.0 * ( 1- tau**2/2 ) *  expn(2,gamma2*tau)
          
    Term0 = 3.0/4.0 * Tint**4 * (tau + 2.0/3.0)
    Term1 = 3.0/4.0 * Tirr**4 * (1-alpha) * zeta1
    Term2 = 3.0/4.0 * Tirr**4 * alpha     * zeta2
    
    T = (Term0+Term1+Term2)**(1.0/4.0)
    return T    
    
    
def rosselandMean(wave,y,T,waveUnit='um'):
    lam, f, wavenumber = convertWaveToSi(wave,waveUnit)
    dBdT = PlanckDerivative(lam,T)
    oneOveryR = np.trapz(x=lam,y=dBdT*1/y) / np.trapz(x=lam,y=dBdT)
    yR = 1 / oneOveryR
    return yR    
    
    
def adiabat_dTdz(gamma=1.4, mu=28, g=9.81):
    #gamma=1.4
    #mu=28 g/mol
    #g=9.81 N/kg
    dT_dz = - (gamma-1)/gamma * mu/1000 * g / Rconst  #K/m
    return dT_dz   

def adiabat_dlnTdlnp(gamma=1.4):    
    return 1-1/gamma 

def adiabat_T(T0,p0,p,gamma=1.4):    
    T = T0 * (p/p0)**(1-1/gamma)
    return T

# ---- Condensation curves ---- #
    
def condensT(p,water_abun=None,species='Cr',p_unit='Pa',FeH=0.,verbose=True):

    if p_unit == 'Pa':
        p_bar = p*1e-5 # convert p to bar
    elif p_unit == 'bar':
        p_bar = p
    else:
        raise ValueError('Invalid unit for pressure')
    
    # ---- from Morley+ 2012 ------------------------ #
    if species == 'Cr':
        res = 6.576-0.486*np.log10(p_bar) - 0.486*FeH
    
    elif species == 'MnS':
        res = 7.447-0.42*np.log10(p_bar) - 0.84*FeH
    
    elif species == 'Na2S':
        res = 10.045-0.72*np.log10(p_bar) - 1.08*FeH

    elif species == 'ZnS':
        res = 12.527-0.63*np.log10(p_bar) - 1.26*FeH    

    elif species == 'KCl':
        res = 12.479-0.879*np.log10(p_bar) - 0.879*FeH    
        
    # ------- from Visscher+ 2006 --------------------- #

    elif species == 'NH4H2PO4':
        res = 29.99-0.20*(11.*np.log10(p_bar) + 15.*FeH  )    # M/H in the paper
        
    # -------- from Magnus (1967) --------------------- #
    elif species == 'H2O':
        if water_abun is None:
            raise ValueError('Water abundance is None!')
        else:
            a = 6.1078 * 1e2 * 1e5
            b = 17.269388
            c = 273.16
            d = 35.86
            term1 = 1./b * np.log(p*water_abun/a)
            term2 = term1*d-c
            res = (term2/(term1-1.))**(-1.) * 1e-4
    else: 
        raise ValueError('Condensation curves are not tabulated here for this cloud species')
        
    return 1./res * 1e4 # returns condensation T in K


