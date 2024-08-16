# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:17:57 2016

@author: bbenneke
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import numpy.matlib

import matplotlib
import matplotlib.pyplot as plt
#plt.locator_params(axis = 'x', nbins = 4)

#import matplotlib.dates as mdates
#from matplotlib.ticker import FuncFormatter
from matplotlib import colors, ticker, cm #need for plotCarmaCloud()

#import scipy.io as spio7
from scipy.io.idl import readsav

from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel
from scipy.interpolate import CubicSpline

#import astropy.io.fits as pf
#from astropy.time import Time

import pdb
#import pickle
#from pprint import pprint

#from utilities import remOutliers, calcChi2, find_nearest, calclnlike

import os

import pandas as pd
from copy import deepcopy

try:
    from HRS_tools import Functions as Funcs
except:
    print('HRS_tools for high-res stuff could not be imported')

from bisect import bisect_left,bisect_right
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel,  Gaussian1DKernel, Trapezoid1DKernel
#import pyspectrum #, loadExoFit
from scipy.optimize import minimize
#import time

import auxbenneke.utilities as ut
from auxbenneke.constants import unitfac, pi, day, Rearth, Mearth, Mjup, Rjup, sigmaSB, cLight, hPlanck, parsec, Rsun, au, G, kBoltz, uAtom,mbar, uAtom

from scarlet import radutils as rad

import scipy.io as sio
from astropy.io import fits

from time import time, sleep
import h5py
from bisect import bisect

import pkg_resources
from astropy import table 
from scipy import interpolate

from sys import stdout

from copy import deepcopy
import copy

import warnings

try:
    import disort      # Scattering modules
except:
    print('DISORT not installed') 




try:
    import VULCAN
except:
    print('VULCAN not installed, cannot use chemical kinetics!')

    
#DB_FILE = pkg_resources.resource_filename('<package name>', 'data/sqlite.db')

from numba import jit, vectorize

jdref=2450000
big=1e10



class atmosphere(object):
    '''
    Atmosphere Code for Atmospheric Forward Models 
    '''
    
    
    #%% Inititalization Methods
    
    def __init__(self,name='',filename=None,basedirec='../scarlet_results/',subdirec=None,makesubdirec=True,
                 MoleculeListFile='MoleculesAll_HITEMP_Res250K_ExoMol_H2OCH4_20160524',
                 MetalListFile=None,
                 waveRange=np.array([1.0,2.0]),resolution=64,
                 MolNames    = np.array(['He','H2','H','CH4','C2H2','O2','OH','H2O','CO','CO2','NH3','HCN','H2S','PH3','Na','K','N2']),
                 AbsMolNames = np.array(['CH4','C2H2','O2','OH','H2O','CO','CO2','NH3','HCN','H2S','PH3','Na','K']),
                 doTransit=True, doThermal=True, fitRadiusLowRes=False, numerical_precision='float64',
                 nLay=60,pressureLevels=None, includeGrayCloudDeckThermal=False,
                 mieCondFiles=None,
                 plotTpChangesEveryIteration=True,
                 saveOpac=False,defaultxlim=None, verbose=False,rpAtBottomOfAtm=False,rpAt1Bar=False):

        #'gj1214b_kcl_zns_0nuc0_1xsol_kzz1e7_pregrid'

        if name!='empty':

            print('Initializing SCARLET Forward Model')

            
            self.scarletpath = os.path.dirname(pkg_resources.resource_filename('scarlet', ''))
            if basedirec.endswith('scarlet_results/'):
                basedirec = self.scarletpath+'/../scarlet_results/'
            
            self.datapath = self.scarletpath+'/data'
            print(self.datapath) 
            
            self.name=name
            self.iRun=0

            self.doTransit=doTransit
            self.doThermal=doThermal
            self.numerical_precision = numerical_precision
            self.includeGrayCloudDeckThermal = includeGrayCloudDeckThermal

            self.mieCondFiles=mieCondFiles
            self.saveOpac=saveOpac

            #----------------------------------------
            #set path to output files
            if subdirec is None:
                self.direc=basedirec
            else:
                self.direc=os.path.join(basedirec,subdirec)
            if not os.path.exists(basedirec) and basedirec!='':
                try:
                    os.makedirs(basedirec)
                except:
                    print('basedirec already exists.')
            
            if makesubdirec and (not os.path.exists(self.direc)):
                try:
                    os.makedirs(self.direc)
                except:
                    print('direc already exists.')

            #set filebase
            datetxt=ut.datestr()+'_'
            if filename is None:
                self.filename=datetxt+self.name
            else:
                self.filename=filename
            
            #add output file path and filebase name together
            self.filebase = os.path.join(self.direc,self.filename+'_')
            #----------------------------------------
            
            #Load Molecule Info
            print('MolNames: ',MolNames)
            print('AbsMolNames: ',AbsMolNames)
            self.loadMoleculeInfo(MolNames,AbsMolNames)
            
            #Load Chemical equilibrium look-up table
            self.readInAbunLUT()
            
            #Set pressure grid

            self.setPressureGrid(nLay,pressureLevels,rpAtBottomOfAtm,rpAt1Bar)
            self.fStarSurf=None
            
            #Load OpacitiesLUT And the Wavelength Grid
            self.MoleculeListFile=MoleculeListFile
            self.MetalListFile=MetalListFile
            self.loadOpacityLUTAndWaveGrid(MoleculeListFile,waveRange,resolution)

            if self.doTransit == False:
                fitRadiusLowRes = False

            self.fitRadiusLowRes = fitRadiusLowRes
            if self.fitRadiusLowRes:
                # self.loadOpacityLUTAndWaveGrid_lowres(MoleculeListFile,waveRange,128)
                # lowres = 128
                lowres = self.fitRadiusLowRes
                self.wave_lowres = self.wave[::lowres]
                self.nWave_lowres = len(self.wave_lowres)
                self.LookUpSigma_lowres = self.LookUpSigma[:,:,:,::lowres]
                self.sigmaAtm_lowres = self.sigmaAtm[:,:,::lowres]
                self.ciaH2LUT_lowres = self.ciaH2LUT[:,::lowres]
                self.ciaHeLUT_lowres = self.ciaHeLUT[:,::lowres]
                self.f_lowres = self.f[::lowres]
                self.indRM_lowres = self.indRM[::lowres]

            self.opacSources={'molLineAbsorb':True,
                                        'cia':True,
                               'rayleighScat':True,
                               'fineHazeScat':True,
                              'paramMieCloud':True,
                              'carmaMieCloud':True}
        
            #Other parameter
            self.chi2=float(np.nan)
            self.color=None
            self.offset=0.0
            self.plotTpChangesEveryIteration=plotTpChangesEveryIteration
            
            self.defaultxlim=defaultxlim
            
            self.verbose=verbose



    def reInitAtmosphereModel(self,oldLUT=False):
        '''
        Reinitializes the atmosphere model when the atmosphere object is reloaded from a saved pickle file
        If need, user can change nLay, pressureLevels, MolNames, AbsMolNames, MoleculeListFile, waveRange, resolution
        '''

        #Load Molecule Info
        self.loadMoleculeInfo(self.MolNames,self.AbsMolNames)

        #Load Chemical equilibrium look-up table
        self.readInAbunLUT()
        
        #Set pressure grid
        self.setPressureGrid(self.nLay,None)
        
        #Load OpacitiesLUT And the Wavelength Grid
        self.loadOpacityLUTAndWaveGrid(self.MoleculeListFile,self.waveRange,self.resolution,oldLUT=oldLUT)
        


    def setPressureGrid(self,nLay,pressureLevels,rpAtBottomOfAtm=False,rpAt1Bar=False):
    
        #Pressure Grid
        if pressureLevels is None:
            if type(nLay) is int:
                self.p=np.logspace(-5,9,nLay)
            elif len(nLay)==1:
                self.p=np.logspace(-5,9,nLay[0])
            else:
                self.p=np.r_[np.logspace(-5,0,nLay[0]),np.logspace(0,5,nLay[1])[1:-1],np.logspace(5,9,nLay[2])]
#                self.p=np.r_[np.logspace(-5,1.5,nLay[0]),np.logspace(1.5,6.5,nLay[1])[1:-1],np.logspace(6.5,11,nLay[2])]
#                self.p=np.r_[np.logspace(0.5,3.5,nLay[0]),np.logspace(3.5,6,nLay[1])[1:-1],np.logspace(6,7,nLay[2])] 
        else:     
            print('User defined pressure Levels!')
            self.p=pressureLevels
        self.p = self.p.astype(self.numerical_precision)

        self.nLay=len(self.p) #specify number of layers
        self.PSurf=self.p[-1]
        if rpAtBottomOfAtm:
            self.iLevRpRef= len(self.p)-1
        elif rpAt1Bar:
            self.iLevRpRef=bisect(self.p,1000*mbar)
        else:
            self.iLevRpRef=bisect(self.p,10*mbar)  #index for reference radius
        self.iLev1bar=bisect(self.p,1000*mbar)  
        self.iLev1mbar=bisect(self.p,1*mbar)  
       
        
    def loadMoleculeInfo(self,MolNames,AbsMolNames):
        #Molecules
        self.colors={'He':'yellow', 'H2':'orange','N2':'gray',
                     'CH4':'purple', 'C2H2':'lightgreen', 'O2':'gray', 'OH':'lightgray', 'H2O':'blue', 'CO':'red', 'CO2':'green',
                     'NH3':'brown', 'HCN':'peru', 'H2S':'C8', 'PH3':'C1', 'Na':'C9', 'K':'C6',
                     'TiO':'#1f77b4', 'SiO':'#ff7f0e', 'H-':'#2ca02c', 'VO':'#d62728', 'HDO':'#9467bd', 'FeH':'#8c564b', 'O3':'#8c564b', 'SO2':'gold', 'AlO':'gold', 'CrH':'gray', 'CrO':'r', 'CrO2':'g', 'CrO3':'k', 'VO2':'gray', 'TiO2':'gray', 'TiS':'gray', 'TiH':'gray',
                     'H':'C1', 'He+':'C2', 'Li':'C3', 'Li+':'C4', 'Be':'C5', 'Be+':'C6', 'Be++':'C7', 'B':'C5', 'B+':'C6', 'B++':'C7', 'C':'C8', 'C+':'C9', 'C++':'C10',
                     'N':'C11', 'N+':'C12', 'N++':'C13', 'O':'C14', 'O+':'C15', 'O++':'C16', 'F':'C17', 'F+':'C18', 'F++':'C19', 'Ne':'C20',
                     'Ne+':'C21', 'Ne++':'C22', 'Na+':'C23', 'Na++':'C24', 'Mg':'C25', 'Mg+':'C26', 'Mg++':'C27', 'Al':'C28', 'Al+':'C29', 'Al++':'C30',
                     'Si':'C31', 'Si+':'C32', 'Si++':'C33', 'P':'C31', 'P+':'C32', 'P++':'C33', 'S':'C31', 'S+':'C32', 'S++':'C33', 'Cl':'C34', 'Cl+':'C35', 'Cl++':'C36', 'Ar':'C37', 'Ar+':'C38', 'Ar++':'C39', 'K+':'C40',
                     'K++':'C41', 'Ca':'C42', 'Ca+':'C43', 'Ca++':'C44', 'Sc':'C45', 'Sc+':'C46', 'Sc++':'C47', 'Ti':'g', 'Ti+':'m', 'Ti++':'C50',
                     'V':'c', 'V+':'C52', 'V++':'C53', 'Cr':'C54', 'Cr+':'C55', 'Cr++':'C56', 'Mn':'C57', 'Mn+':'C58', 'Mn++':'C59', 'Fe':'r',
                     'Fe+':'orange', 'Fe++':'y', 'Co':'C63', 'Co+':'C64', 'Co++':'C65', 'Ni':'C66', 'Ni+':'C67', 'Ni++':'C68', 'Cu':'C69', 'Cu+':'C70', 'Cu++':'C70',
                     'Zn':'C71', 'Zn+':'C72', 'Zn++':'C73', 'Ga':'C74', 'Ga+':'C75', 'Ga++':'C76', 'Ge':'C77', 'Ge+':'C78', 'As':'C79', 'Se':'C80',
                     'Br':'C81', 'Kr':'C82', 'Rb':'C83', 'Sr':'C84', 'Sr+':'C84', 'Y':'C85', 'Y+':'C86', 'Y++':'C87', 'Zr':'C88', 'Zr+':'C89', 'Zr++':'C90',
                     'Nb':'C91', 'Nb+':'C92', 'Nb++':'C93', 'Mo':'C94', 'Mo+':'C95', 'Mo++':'C95', 'Tc':'C96', 'Tc+':'C96', 'Ru':'C97', 'Ru+':'C98', 'Ru++':'C98', 'Rh':'C99', 'Rh+':'C100', 'Rh++':'C100',
                     'Pd':'C101', 'Pd+':'C102', 'Ag':'C103', 'Ag+':'C104', 'Cd':'C105', 'Cd+':'C106', 'In':'C107', 'In+':'C108', 'Sn':'C109', 'Sn+':'C110',
                     'Sb':'C111', 'Te':'C112', 'I':'C113', 'Xe':'C114', 'Xe+':'C115', 'Cs':'C116', 'Ba':'C117', 'Ba+':'C118', 'La':'C119', 'La+':'C120',
                     'La++':'C121', 'Ce':'C122', 'Ce+':'C123', 'Ce++':'C124', 'Pr':'C125', 'Pr+':'C126', 'Pr++':'C127', 'Nd':'C128', 'Nd+':'C129', 'Nd++':'C130',
                     'Pm':'C131', 'Sm':'C132', 'Sm+':'C133', 'Sm++':'C134', 'Eu':'C135', 'Eu+':'C136', 'Eu++':'C137', 'Gd':'C138', 'Gd+':'C139', 'Gd++':'C140',
                     'Tb':'C141', 'Tb+':'C142', 'Tb++':'C143', 'Dy':'C144', 'Dy+':'C145', 'Dy++':'C146', 'Ho':'C147', 'Ho+':'C148', 'Ho++':'C149', 'Er':'C150',
                     'Er+':'C151', 'Er++':'C152', 'Tm':'C153', 'Tm+':'C154', 'Tm++':'C155', 'Yb':'C156', 'Yb+':'C157', 'Yb++':'C158', 'Lu':'C159', 'Lu+':'C160',
                     'Lu++':'C161', 'Hf':'C162', 'Hf+':'C163', 'Hf++':'C164', 'Ta':'C165', 'Ta+':'C166', 'W':'C167', 'W+':'C168', 'Re':'C169', 'Re+':'C170',
                     'Os':'C171', 'Os+':'C172', 'Ir':'C173', 'Ir+':'C174', 'Pt':'C175', 'Pt+':'C176', 'Pt++':'C177', 'Au':'C178', 'Au+':'C179', 'Au++':'C180',
                     'Hg':'C181', 'Hg+':'C182', 'Hg++':'C183', 'Tl':'C184', 'Pb':'C185', 'Pb+':'C186', 'Bi':'C187', 'Bi+':'C188', 'Po':'C189', 'At':'C190',
                     'Rn':'C191', 'Fr':'C192', 'Ra':'C193', 'Ac':'C194', 'Th':'C195', 'Th+':'C196', 'Th++':'C197', 'Pa':'C198', 'U':'C199', 'U+':'C200'}

        self.MolNames=MolNames   
        self.nMol=len(self.MolNames)

        self.AbsMolNames=AbsMolNames
        self.nAbsMol=len(self.AbsMolNames)
        self.AbsMolInd=np.nonzero(np.in1d(self.MolNames, self.AbsMolNames))[0]
        
        self.molInd=dict()
        for MolName in self.MolNames:
            self.molInd[MolName]=self.calcMolInd(MolName)

        print('MolNames:   ', MolNames)
        print('AbsMolNames:', AbsMolNames)
        
        # check that all AbsMolNames are in MolNames
        isin = np.isin(self.AbsMolNames,self.MolNames, invert=True)
        if np.sum(isin):
            raise Exception('Error: {} included in AbsMolNames but not in MolNames'.format(self.AbsMolNames[isin]))
        
        self.readMoleculesProperties()


    def loadOpacityLUTAndWaveGrid(self,MoleculeListFile,waveRange,resolution,oldLUT=False):
       
        #Load molecular absorption cross sections (also defines wavelength grid)
        if oldLUT is False:
            self.prepare_1D_CrossSecLUT(MoleculeListFile,waveRange,resolution)
        else:
            self.prepare_1D_CrossSecLUT_old(MoleculeListFile,waveRange,resolution)
        
        ind=3; self.resPower = self.wave[ind] / (self.wave[ind+1]-self.wave[ind])
        print('Wavelength range of atmosphere model {} - {}  (resolving power = {})'.format(str(np.round(self.wave[0],3)),str(np.round(self.wave[-1],3)),np.round(self.resPower,0)))
        self.nWave=len(self.wave)
        self.f=cLight/(self.wave*1e-6)   # Hz=1/s        
        self.indRM = np.arange(self.wave.size) #indices to calculate Rosseland mean opacity
        
        
        #Load collision induced absorption look-up tables
        self.read_CIA_LUT()
        if self.mieCondFiles is not None:
            self.read_Mie_LUT()

    
    def convertMolNameToFastchemNotation(self, molName):
        #Takes in a molecule name and returns its equivalent in the notation used in FastChem
        #FastChem Notation: molecules in alphabetic order and add '1' for elements that are present once in the molecule
        #except singular element name don't include the '1'
        #The FastChem notation is used in the ChemEquiLUT
        if molName == 'e-':
            return 'e-'
        
        arr = []
        tmp_str = ''
        for ch in molName:
            if ch.isupper():
                arr.append(tmp_str)
                tmp_str = ch
            elif ch.islower():
                tmp_str += ch
            elif ch.isnumeric():
                tmp_str += ch
            elif ch == '+' or ch == '-':
                arr.append(tmp_str)
                tmp_str = ch
        if tmp_str[0].isupper():
            arr.append(tmp_str)
            tmp_str = ''
                                
        arr = arr[1:]
        arr.sort()

        if len(arr) == 1 and not arr[0][-1].isnumeric() and tmp_str == '':
            return arr[0]
    
        result = ''
        for elem in arr:
            if not elem[-1].isnumeric():
                result += (elem + '1')
            else:
                result += elem
        result += tmp_str
        
        if result == 'C1H1N1':
            result += '_1'

        return result

    
    def readInAbunLUT(self):
        
        print('\n\nReading in the Chemical Equilibrium LUT...')
        filename = self.scarletpath+'/'+'../scarlet_LookUpQuickRead/FullChemEquiLUT_20220613.hdf5' # Updated table from FastChem 2.0 (available in bbenneke/shared)
        # filename = self.scarletpath+'/'+'../scarlet_LookUpQuickRead/ChemEquiLUT_20210709.hdf5' # Previous table from FastChem 1.0
        print('File Size = {0:10.2f} MB'.format(os.path.getsize(filename)/1024.**2))
        t0 = time()
        ChemEquiLUT = h5py.File(filename, 'r')
        
        allAbunLUT = ChemEquiLUT['AbunLUT'][:]
        self.CtoORatioList = ChemEquiLUT['CtoORatioList'][:]
        self.MetallicityList = ChemEquiLUT['MetallicityList'][:]
        self.PTable = ChemEquiLUT['Pres'][:]
        self.TTable = ChemEquiLUT['Temp'][:]
        allMolName = np.array([])
        for mol in ChemEquiLUT['MolName'][:]:
            allMolName = np.append(allMolName, mol.decode('UTF-8'))
        print('Finished reading Chemical Equilibrium LUT file')
        t1 = time()
        print('Loading time: {0:8.3f} seconds ({1:8.2f} MB/s)\n\n'.format(t1-t0,os.path.getsize(filename)/1024.**2 / (t1-t0)))

        self.AbunLUT = np.zeros(np.r_[allAbunLUT.shape[0:4],self.nMol], dtype=self.numerical_precision)
        FastChemMolNames = np.array([])
        for mol in self.MolNames:
            FastChemMolNames = np.append(FastChemMolNames, self.convertMolNameToFastchemNotation(mol))
        # find all indicies for active molecules
        for iMol,molNames in enumerate(FastChemMolNames):
            index = np.squeeze(np.where(molNames==allMolName))
            if molNames == allMolName[index]:
                self.AbunLUT[:,:,:,:,iMol] = allAbunLUT[:,:,:,:,index]
        
        self.AbunLUT=np.log10(self.AbunLUT)
        stdout.write('Finished preparing Chemical Equilibrium LUT file')
        
        return
        

    def readMoleculesProperties(self):
        molTable=table.Table.read(self.datapath+'/MoleculesAll_Iso0.csv',format='csv',delimiter=',')
        
        molTable['MolName']

        self.MolarMass=np.zeros([self.nMol], dtype=self.numerical_precision)
        self.RefracIndex=np.zeros(self.nMol, dtype=self.numerical_precision)
        self.KingsCorr=np.zeros(self.nMol, dtype=self.numerical_precision)
        for iMol,MolName in enumerate(self.MolNames):
            # if self.MolNames[iMol] == 'H-': 
            #     thisMol=molTable[molTable['MolName']=='O2'] # picked O2 but could be anything
            #     self.MolarMass[iMol] = 1.0
            #     self.RefracIndex[iMol] = 1.0
            #     self.KingsCorr[iMol] = 1.0
            # elif self.MolNames[iMol] == 'HDO':
            #     thisMol=molTable[molTable['MolName']=='H2O'] # make HDO like H2O
            #     self.MolarMass[iMol] = thisMol[0]['MolarMass'] + 1.0
            #     self.RefracIndex[iMol] = thisMol[0]['RefracIndex']
            #     self.KingsCorr[iMol] = thisMol[0]['KingsCorr']
            # elif self.MolNames[iMol] == 'FeH':
            #     self.MolarMass[iMol] = 57.0
            #     self.RefracIndex[iMol] = 1.0
            #     self.KingsCorr[iMol] = 1.0
            # elif self.MolNames[iMol] == 'O3':
            #     self.MolarMass[iMol] = 48.0
            #     self.RefracIndex[iMol] = 1.0
            #     self.KingsCorr[iMol] = 1.0
            # else:
            #     thisMol=molTable[molTable['MolName']==self.MolNames[iMol]]
            #     #print self.MolNames[iMol],thisMol[0][2]
            #     #generate array of mollar masses for active molecules
            #     try:
            #         self.MolarMass[iMol]=thisMol[0]['MolarMass']
            #         self.RefracIndex[iMol]=thisMol[0]['RefracIndex']
            #         self.KingsCorr[iMol]=thisMol[0]['KingsCorr']
            #     except:
            #         pdb.set_trace()
                    
            thisMol=molTable[molTable['MolName']==self.MolNames[iMol]]
            try:
                self.MolarMass[iMol]=thisMol[0]['MolarMass']
                self.RefracIndex[iMol]=thisMol[0]['RefracIndex']
                self.KingsCorr[iMol]=thisMol[0]['KingsCorr']
            except:
                print(MolName)
                print('Add species to scarlet/data/MoleculesAll_Iso0.csv file')
                pdb.set_trace()
       
                
    def read_CIA_LUT(self):
        """
        This function reads CIA files, transforms wavenumbers into wavelengths
        and interpolates the CIA coefficients over the spectral range given by self.wave.

        The attribute self.ciaList_Hitran is defined as a dictionary containing every important information from the HITRAN files.
        keys: 'MOL1', string of the first molecule's name
              'MOL2', string of the first molecule's name
              'TempList', array containing the temperatures for each given spectral range
              'wave', array containing the wavelengths for each given spectral range (before interpolation) [microns]
              'interpolatedCoeff', 2D array containing the interpolated coefficients depending on the temperature. The dimensions of this array is (temperature, self.wave)
        """

        # All the molecules files that are taken into account are listed in a dictionary: keys are molecules and values are paths
        cia_HITRAN_files = {'O2_O2': os.path.join(self.datapath, 'cia', 'O2-O2_2018b.cia'),
                            'N2_N2': os.path.join(self.datapath, 'cia', 'N2-N2_2021.cia'),
                            'CO2_CO2': os.path.join(self.datapath, 'cia', 'CO2-CO2_2018.cia')}

        # Initialize list that will contain the dictionaries for each spectral range
        CIAList_HITRAN = []

        # Loop over molecule files
        for molecules, file in cia_HITRAN_files.items():

            ##Initialize lots of lists
            temperatureList = []  # List that will contain every temperature of a given spectral range
            temperatureAllSpecRanges = []  # List that will contain the sublists of temperatures for all spectral ranges

            # Lists that will contain every wavenumber/coeff value for a given temperature
            waveList = []
            coeffList = []

            # List that contains every coeffList of a given spectral range
            coeffTotalList = []

            # List that contains all coeffTotalList (should contain a number of lists equivalent to the number of different spectral ranges)
            waveAllSpecRanges = []
            coeffAllSpecRanges = []

            # List that contains every minimum or maximum wavenumber for each subset of different temperatures
            specRangeMin = []
            specRangeMax = []

            # List that contains the reference number of a given temperature
            referenceList = []

            # Marker needed to consider a different step for the first header
            first_header = True

            # Read names for molecules 1 and 2 from the cia_HITRAn_files dictionary
            mol1 = molecules.split('_')[0]
            mol2 = molecules.split('_')[1]

            # Open file, read line by line and close it
            fileHandle = open(file, 'r')
            lineList = fileHandle.readlines()
            fileHandle.close()

            # Loop for every line of the file document
            for line in lineList:

                # if length of string is longer than the first two columns, then it is a header
                if len(line) > 40:

                    # Read header and split it, only keep the non-empty strings
                    readHeader = line.split(' ')
                    readHeader = list(filter(None, readHeader))

                    # Read temperature from header
                    Temp = float(readHeader[4])

                    # Read spectral range from header
                    specRangeMin.append(float(readHeader[1]))
                    specRangeMax.append(float(readHeader[2]))

                    # Keep in mind the index of the specRangeMin just added, in order to compare with the previous spectral range obtained.
                    # If spectral range is different than the previous spectral range read, then it is a different subset of data
                    index = len(specRangeMin) - 1

                    # Do this for every header, except for the first one (no coeffList is stored yet when reading the first header)
                    if not first_header:
                        coeffTotalList.append(np.array(coeffList))

                    # If these conditions are met, the data is part of a different spectral range, and a new sublist must be created
                    if ((specRangeMin[index] != specRangeMin[index - 1]) or (
                            specRangeMax[index] != specRangeMax[index - 1])) and not first_header:
                        # Add the waveList, the coeffTotalList and the temperature List of the previous spectral range to the lists containing all spectral ranges
                        waveAllSpecRanges.append(np.array(waveList))
                        coeffAllSpecRanges.append(np.array(coeffTotalList))
                        temperatureAllSpecRanges.append(np.array(temperatureList))

                        # Reset lists
                        coeffTotalList = []
                        temperatureList = []

                    # Append new temperature to list
                    temperatureList.append(Temp)

                    # Read reference and add to reference list
                    referenceList.append(int(readHeader[-1]))

                    # Turn marker to False because the next headers are not the first one
                    first_header = False

                    # Reset wave and coeff list
                    waveList = []
                    coeffList = []


                # If the line is not a header, do this instead
                else:
                    # Read line and split it, only keep the non-empty strings
                    readLine = line.split(' ')
                    readLine = list(filter(None, readLine))

                    # read wavenumber and its associated coefficient, and append them to their respective lists
                    waveList.append(float(readLine[0]))
                    coeffList.append(max(float(readLine[1]),0))

                # Do something a little different for the last line of the file.
                if line == lineList[-1]:
                    # Add the waveList, the coeffTotalList and the temperature List of the previous spectral range to the lists containing all spectral ranges
                    coeffTotalList.append(np.array(coeffList))
                    coeffAllSpecRanges.append(np.array(coeffTotalList))

                    waveAllSpecRanges.append(np.array(waveList))
                    temperatureAllSpecRanges.append(np.array(temperatureList))

            # Initialize an index to compare with the index of the reference
            indexToCompare = 0

            #If loop only applied to N2-N2. The first three spectral ranges are overlapping, so they have to be brought to a common spectral range adding zeros to complete the spectral range
            if file.endswith('N2-N2_2021.cia'):
                arr0 = np.append(coeffAllSpecRanges[0], np.zeros((14, 2000)), axis=1)
                arr1 = np.append(coeffAllSpecRanges[1], np.zeros((10, 1000)), axis=1)

                # Replace previous element by new element with the new interpolated array concatenated
                coeffAllSpecRanges[2] = np.concatenate((arr0, arr1, coeffAllSpecRanges[2]))
                temperatureAllSpecRanges[2] = np.concatenate(
                    (temperatureAllSpecRanges[0], temperatureAllSpecRanges[1], temperatureAllSpecRanges[2]))

                # Erase old elements from total lists
                del (coeffAllSpecRanges[1])
                del (waveAllSpecRanges[1])
                del (temperatureAllSpecRanges[1])
                del (coeffAllSpecRanges[0])
                del (waveAllSpecRanges[0])
                del (temperatureAllSpecRanges[0])

                # Change specRange min and max corresponding to the new spectral range
                specRangeMin[:34] = np.ones(34) * waveAllSpecRanges[0][0]
                specRangeMax[:34] = np.ones(34) * waveAllSpecRanges[0][-1]

                # Two references refer to the same N2-N2 CIA feature, we will cheat by changing the reference of one to match
                # the other so that the program recognize them as the same feature
                referenceList[39:44] = np.ones(5) * 2

            # Loop that has to be done over each reference from the reference list
            for index, reference in enumerate(referenceList):

                # If the reference is the same as the previous array of data, but the spectral range is different, an interpolation has to be done so that all spectral ranges
                # of the same reference are consistent and not overlapping
                if index != 0 and ((specRangeMin[index] != specRangeMin[index - 1]) or (
                        specRangeMax[index] != specRangeMax[index - 1])) and (
                        reference == referenceList[index - 1]) and (
                        specRangeMax[index] - specRangeMax[index - 1]) < 1000:
                    # Since the index of the list reference does not correspond to the index in the coeffAllSpecRanges list, we have to compare both indexes (index and indexToCompare) through a loop to get the same element
                    # IndexToCompare is incremented of 1 each time a different array corresponding to a different temperature is evaluated in the loop
                    for indexWave, specRange in enumerate(coeffAllSpecRanges):
                        for temperature in range(specRange.shape[0]):

                            # Comparison of the two indexes. If both indexes are the same, then we know that the data that respected the previous conditions is the list specRange with waveAllSpecRanges[indexWave]
                            if indexToCompare == index:

                                # Initialize interpolated table
                                newInterpolatedArray = np.zeros([specRange.shape[0], len(
                                    waveAllSpecRanges[indexWave - 1])])  # array of size (T, wave of previous specRange)

                                # Proceed to the interpolation on the new wavelength range, which corresponds to the wavelength range of the previous list
                                for i in range(len(waveAllSpecRanges[indexWave - 1])):
                                    newInterpolatedArray[:, i] = ut.interp1bb(waveAllSpecRanges[indexWave],
                                                                              specRange.transpose(),
                                                                              waveAllSpecRanges[indexWave - 1][i])

                                # Replace previous element with wrong wavelength range with the new interpolation
                                coeffAllSpecRanges[indexWave] = newInterpolatedArray
                                waveAllSpecRanges[indexWave] = waveAllSpecRanges[indexWave - 1]

                                # Replace previous element by new element with the new interpolated array concatenated
                                coeffAllSpecRanges[indexWave - 1] = np.vstack(
                                    (coeffAllSpecRanges[indexWave - 1], newInterpolatedArray))
                                temperatureAllSpecRanges[indexWave - 1] = np.hstack(
                                    (temperatureAllSpecRanges[indexWave - 1], temperatureAllSpecRanges[indexWave]))

                                # Erase old element from total list
                                del (coeffAllSpecRanges[indexWave])
                                del (waveAllSpecRanges[indexWave])
                                del (temperatureAllSpecRanges[indexWave])

                            # Incrementation of the index to compare
                            indexToCompare += 1

                # Reset the index to compare when trying a new index in the reference list
                indexToCompare = 0

            # Put the temperatures and their corresponding coefficients in increasing order
            if file.endswith('N2-N2_2021.cia'):
                temperatureAllSpecRanges[1] = np.roll(temperatureAllSpecRanges[1], 5)
                coeffAllSpecRanges[1] = np.roll(coeffAllSpecRanges[1], 5, axis=0)

            if file.endswith('CO2-CO2_2018.cia'):
                # Add CO2-CO2 CIAs between 2 and 2.5 um (Lee et al. 2016, Sensitivity of net thermal flux to the abundance of trace
                # gases in the lower atmosphere of Venus)
                coeffAllSpecRanges.append(np.full((1, 1000), 1.385277e-39 * 3e-8))
                coeffAllSpecRanges.append(np.full((1, 1050), 1.385277e-39 * 7.7e-9))
                waveAllSpecRanges.append(np.linspace(4000, 5000, 1000))
                waveAllSpecRanges.append(np.linspace(5000, 6050, 1050))
                temperatureAllSpecRanges.append(np.array([230]))
                temperatureAllSpecRanges.append(np.array([230]))

            # for each different spectral range, change from wavenumber to wavelength and flip tables (interp1bb requires monotonic increase?)
            for specRange in waveAllSpecRanges:
                specRange[:] = 10000 / specRange[::-1]

            # for each different spectral range flip tables (interp1bb requires monotonic increase?) and transpose array for dimension issues
            for index, specRange in enumerate(coeffAllSpecRanges):
                coeffAllSpecRanges[index] = specRange[:, ::-1].transpose()

            # Loop over every spectral range
            for index, specRange in enumerate(coeffAllSpecRanges):
                # Set first and last value of the array to 0, because the extrapolation outside the known values have to be 0
                specRange[0] = 0.0
                specRange[-1] = 0.0

                # initialize interpolated tables
                interp = np.zeros([len(temperatureAllSpecRanges[index]), self.nWave])

                for i in range(self.nWave):
                    # perform interpolation on each data table/file of given spectral range
                    interp[:, i] = ut.interp1bb(waveAllSpecRanges[index], specRange,
                                                self.wave[i])  # [j,i] where: i refers to wave, j refers to T

                # Create dictionary and add to CIAList_Hitran
                CIA_HITRAN = {'MOL1': mol1, 'MOL2': mol2, 'TempList': temperatureAllSpecRanges[index],
                              'wave': waveAllSpecRanges[index], 'interpolatedCoeff': interp}
                CIAList_HITRAN.append(CIA_HITRAN)

        self.ciaList_Hitran = CIAList_HITRAN

        #self.ciaLUT=0
        
        #open LUT data files
        f1=np.genfromtxt(os.path.join(self.datapath,'cia','final_CIA_LT.dat'),delimiter=' ',skip_header=3) #H2H2 low temp
        f2=np.genfromtxt(os.path.join(self.datapath,'cia','final_CIA_HT.dat'),delimiter=' ',skip_header=3) #H2H2 high temp
        f3=np.genfromtxt(os.path.join(self.datapath,'cia','cia_h2h2.dat'),delimiter=' ',skip_header=3) #H2H2 T>1000K (VHT = very high temp)
        f4=np.genfromtxt(os.path.join(self.datapath,'cia','ciah2he_dh_quantmech.dat'),delimiter=' ',skip_header=3) #HeH2 (same T range as VHT)
        

        #Split 1st row and read table to grab temperatures
        LT=f1[0,1:] #K
        HT=f2[0,1:]
        VHT=f3[0,1:] 
        #same for He

        #Split 1st col and convert from wavenumber to wavelength
        wave_LT=10000/f1[1:,0] #um 
        #same for HT
        wave_VHT=10000/f3[1:,0] 
        wave_He=10000/f4[1:,0]
        
        #L values from CIA data files (excluding wavenumber column and temperature row)
        L_LT=f1[1:,1:]
        L_HT=f2[1:,1:]
        L_VHT=f3[1:,1:]
        L_He=f4[1:,1:]
        
        
        #Flip tables (interp1bb requires monotonic increase?)
        wave_LT=wave_LT[::-1]
        wave_VHT=wave_VHT[::-1]
        wave_He=wave_He[::-1]
        
        
        L_LT=L_LT[::-1,:]
        L_HT=L_HT[::-1,:]
        L_VHT=L_VHT[::-1,:]
        L_He=L_He[::-1,:]
        
        
        #initialize wavelength interpolated tables
        #len(LT)=len(HT)=len(VHT)=7 for our files (probably by design?)
        interp_LT=np.zeros([self.nWave,len(LT)], dtype=self.numerical_precision)
        interp_HT=np.zeros([self.nWave,len(HT)], dtype=self.numerical_precision)
        interp_VHT=np.zeros([self.nWave,len(VHT)], dtype=self.numerical_precision)
        interp_He=np.zeros([self.nWave,len(VHT)], dtype=self.numerical_precision)
        
        
        for i in range(self.nWave):
                
            #perform interpolation on each data table/file (3 for H2H2 and 1 for H2He)
            interp_LT[i,:]=ut.interp1bb(wave_LT,L_LT,self.wave[i]) #[i,j] where: i refers to wave, j refers to T
            interp_HT[i,:]=ut.interp1bb(wave_LT,L_HT,self.wave[i])
            interp_VHT[i,:]=ut.interp1bb(wave_VHT,L_VHT,self.wave[i])
            interp_He[i,:]=ut.interp1bb(wave_He,L_He,self.wave[i])
                    
        #PROBLEM: HT and VHT don't seem to agree at 1000 K?
        #SOLUTION: have HT take precendence (this is what was done in MATLAB...)
        
        #Need to transpose in order to interpolate on T
        #Also concatenate the H2H2 tables
        self.ciaH2LUT=np.transpose(np.concatenate((interp_LT,interp_HT,interp_VHT),axis=1)) #check units???
        self.ciaHeLUT=np.transpose(interp_He)
        
        #define array of ALL temperatures:
        self.ciaT=np.concatenate((LT,HT,VHT)) #K
        
        
        
    def read_Mie_LUT(self):
        
        print(self.mieCondFiles)

        self.mieLUT=dict()

        self.conds=[]
        for mieCondFile in self.mieCondFiles:
            
            print('Loading: ', mieCondFile)
            
            f = sio.loadmat(os.path.join(self.datapath,'mie',mieCondFile))
            if 'MgSiO3' in self.mieCondFiles[0]:
                cond='MgSiO3'
            else:
                cond = str(f['Aerosol'][0][0][3])[3:-2]
            self.conds.append(cond)
            
            #don't need these in SI
            reff=f['rpart'][0] #[m]
            nu=f['nuMie']      #[1/cm]
            wave=10000/nu[0]   #[um]
            
            #Mie scattering
            Qsca_Gpart=f['Qsca_Gpart'] #[m^2]
            Qsca_Gpart=np.transpose(Qsca_Gpart) #need this for interp1bb() since it needs x as rows for f(x)?
            #Mie absorption
            Qabs_Gpart=f['Qabs_Gpart'] #[m^2]
            Qabs_Gpart=np.transpose(Qabs_Gpart)
                        
            #FLIP TABLE
            ''' NEED TO EXPLICITLY TAKE wave[0] (i.e. 1st row), since wave is 1 X 310 array!!! '''
            wave=wave[::-1] #need increasing order like self.wave
            Qsca_Gpart=np.flip(Qsca_Gpart,0) #also must flip these since they are f(wave) !!!
            Qabs_Gpart=np.flip(Qabs_Gpart,0)   
        
            #Initialize wavelength interpolated arrays
            Qsca_Gpart_interp=np.zeros([self.nWave,reff.size], dtype=self.numerical_precision)
            Qabs_Gpart_interp=np.zeros([self.nWave,reff.size], dtype=self.numerical_precision)
                        
            #do wavelength interpolation
            for i in range(self.nWave):
                Qsca_Gpart_interp[i,:]=ut.interp1bb(wave,Qsca_Gpart,self.wave[i]) 
                Qabs_Gpart_interp[i,:]=ut.interp1bb(wave,Qabs_Gpart,self.wave[i])
                                    
            print(cond)
            
            print('!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('SETTING Qabs_Gpart to 0.0 for Mie Scattering')
            Qabs_Gpart_interp[:]=0.0
            
            self.mieLUT[cond]=dict()
            self.mieLUT[cond]['reff']       = reff #effective particle radius array [m]
            self.mieLUT[cond]['Qabs_Gpart'] = Qabs_Gpart_interp #Mie scattering LUTs [m^2 per particle]
            self.mieLUT[cond]['Qsca_Gpart'] = Qsca_Gpart_interp #Mie absorption LUTs [m^2 per particle]
   

        
    def readCarmaFile(self,carmaFile):

        self.carmaFile=carmaFile
        print('\nReading in carmaFile: '+self.carmaFile)

        t = Table.read(self.carmaFile, format='ascii.basic', delimiter=' ')
        
        
        nLayInCarmaFile=t['z_index'][-1]
        if nLayInCarmaFile!=self.nLay:
            raise ValueError('Number of layers in Carma File does not match the number of pressure levels! ({:d}!={:d})'.format(nLayInCarmaFile,self.nLay))
        nSizeBins=t['bin_index'][-1]

        #loop over active condensates (create dict to hold npart for each condensate)
        self.npartConds=dict() 
        for iCond,cond in enumerate(self.conds):
            print(cond)
            if nSizeBins!=self.mieLUT[cond]['reff'].shape[0]:
                raise ValueError('Number of particle size bins in Carma File does not match the number of particle sizes in mieLUT! ({:d}!={:d})'.format(nSizeBins,self.mieLUT[cond]['reff'].shape[1]) )
            
            colname='n_'+cond.lower()+'[cm-3]'
            if colname in t.colnames:
                npart=t[colname].data.reshape(nSizeBins,self.nLay).transpose()
            else:
                warnings.warn('Column {:s} does not exist in {:s}\n Condensate cond is ignored!!!!!!!!!'.format(colname,self.carmaFile))
                npart=np.zeros([self.nLay,nSizeBins], dtype=self.numerical_precision)
                
            npart=np.flip(npart,0) #need to flip along layer axis to agree with pressure (since CARMA files are in order of DECREASING pressure)
            self.npartConds[cond] = npart*1e6   # [cm**-3] to [m**-3] [nLay,nSizeBins] Number density of particles in that layer and size bin

 
  
    def prepare_1D_CrossSecLUT(self,MoleculeListFile,waveRange,resolution):
        
        self.MoleculeListFile=MoleculeListFile
        self.waveRange=waveRange
        self.resolution=resolution
        
        print(waveRange[0],waveRange[1],resolution)
        print(os.getcwd())
        
        self.LUTFile = self.scarletpath+'/'+'../scarlet_LookUpQuickRead/'+os.path.basename(self.MoleculeListFile)+'_{}_{}_{}.mat'.format(waveRange[0],waveRange[1],resolution) 
        print(self.LUTFile)
        #print os.path.isfile(self.LUTFile)
        
        #Read in Look up table for opacity
        success=0
        while success==0:
            try:
                f = h5py.File(self.LUTFile,'r')
                self.wave=np.array(f['Wave_microns'])
                # self.wave.astype(self.numerical_precision)
                self.nWave=len(self.wave)
                success=1
            except:
                print('An error occured trying to read the Look Up Table file. Let\'s wait a few seconds and try again.')
                sleep(50)
#        try:
#            f = h5py.File(self.LUTFile)
#            self.wave=np.array(f['Wave_microns'])
#            self.nWave=len(self.wave)
#        except:
#            print('An error occured trying to read the Look Up Table file. Let\'s wait a few seconds and try again.')
#            sleep(50)
#            try:
#                f = h5py.File(self.LUTFile)
#                self.wave=np.array(f['Wave_microns'])
#                self.nWave=len(self.wave)
#            except:
#                print self.LUTFile
#                raise ValueError('ERROR: Look Up Table file not in scarlet_LookUpQuickRead or busted! Check file size!')  

        # If a MetalListFile is given, load it
        if self.MetalListFile == 'None' or self.MetalListFile == None:
            LookUpMolNames_metals = []
        else:
            if not self.MetalListFile != 'None' or self.MetalListFile != None:
                try:
                    self.LUTFile_metals = self.scarletpath + '/' + '../scarlet_LookUpQuickRead/' + os.path.basename(
                        self.MetalListFile) + '_{:g}_{:g}_{:g}.mat'.format(waveRange[0], waveRange[1], resolution)
                    print(self.LUTFile_metals)
                    f_metals = h5py.File(self.LUTFile_metals, 'r')
                    # wave_metals=np.array(f_metals['Wave_microns'])
                    # nWave_metals=len(wave_metals)
                    LookUpTGrid_metals = np.array(f_metals['LookUpTGrid'])
                    match = np.isin(LookUpTGrid_metals, np.array(f['LookUpTGrid']))
                    if match.sum() != len(match):
                        raise Exception(
                            'Error: the temperature grid in the metals file does not match the temperature grid in the molecules file!')
    
                    LookUpTGrid_metals.astype(self.numerical_precision)
    
                    sigma_metals = np.array(f_metals['sigma_mol'])[:]
                    # sigma_metals.astype(self.numerical_precision)
                    LookUpMolNames_metals = f_metals['LookUpMolNames'][:]
    
                    # rotker = Funcs.RotKerTransitCloudy(pl_rad=1*u.jupiterRad, pl_mass=1*u.jupiterMass, t_eq=2000*u.K, omega=2*np.pi/1.809/u.day, resolution=250000)
                    #
                    # broad_species = {'Fe':12, 'Ca+':16, 'V':10, 'Mn':10, 'Ni':10, 'Mg':20, 'Ca':10, 'Cr':10, 'Li':21, 'Sr+':10, 'Co':10, 'Si':10}
                    # for specie, fwhm in broad_species.items():
                    #
                    #     print('Broadening {} to {} km/s !!!!!!!!!!!!!!!!!'.format(specie, fwhm))
                    #
                    #     specie_ind = np.where(LookUpMolNames_metals == specie)[0][0]
                    #
                    #     kernel = rotker.return_fwhm_kernel(res_sampling=250000, fwhm_km=fwhm, n_os=100, pad=5)
                    #     for i in np.arange(len(LookUpTGrid_metals)):
                    #         broadened = convolve(sigma_metals[i,0,:,specie_ind], kernel, boundary='extend')
                    #         sigma_metals[i,0,:,specie_ind] = broadened
    
    
                except:
                    print('***WARNING: Metals Look Up Table file given but could not be loaded***')
                    LookUpMolNames_metals = []

        self.LookUpTGrid = np.array(f['LookUpTGrid'])
        # self.LookUpTGrid.astype(self.numerical_precision)

        LookUpPGrid = np.array(f['LookUpPGrid'])
        # LookUpPGrid.astype(self.numerical_precision)

        sigma = np.array(f['sigma_mol'])[:]
        # sigma.astype(self.numerical_precision)
        
        #Find out what molecules are saved in LookUpTable
        LookUpMolNames=f['LookUpMolNames'][:]

        # Build up LookUpSigma for the selected absorbing molecules self.AbsMolNames
        print('Selecting molecules in opacity look up table.')
        dim = np.array(sigma.shape)
        dim[3] = self.nAbsMol
        sigmaNew = np.zeros(dim, dtype=self.numerical_precision)
        for iMol, MolName in enumerate(self.AbsMolNames):
            # ind=np.where(MolName==LookUpMolNames)[0][0]
            # check standard opacity file first
            ind = np.where(LookUpMolNames == MolName.encode())[0]  
            # sometimes files have molecule names as strings ('') instead of bytes (b'')
            if len(ind) == 0:
                ind = np.where(LookUpMolNames == MolName)[0]
            # if still empty, check if species is in the metals opacity file (with no pressure broadening)
            if len(ind) == 0:
                if self.MetalListFile:
                    ind_metal = np.where(LookUpMolNames_metals == MolName)[0]
                    if len(ind_metal) == 0:
                        ind_metal = np.where(LookUpMolNames_metals == MolName.encode())[0]
                else:
                    ind_metal = []
            # check if absorbing molecule is in molecule opacity file
            if len(ind) != 0:
                ind = ind[0]
                sigmaNew[:, :, :, iMol] = sigma[:, :, :, ind]
            # otherwise, check if species is in the metals files
            elif len(ind_metal) != 0:
                ind = ind_metal[0]
                # bring metal opacities to same temperature grid as the molecules file
                # for i, T in enumerate(self.LookUpTGrid):
                for i, T in enumerate(LookUpTGrid_metals):
                    # lowest metal temperature (should be 2500K), set this to all lower self.LookUpTGrid values
                    if i == 0:
                        def find_nearest(array, value):
                            array = np.asarray(array)
                            idx = (np.abs(array - value)).argmin()
                            return idx  # array[idx]

                        T_lower_ind = find_nearest(self.LookUpTGrid, T)
                        # set all T values less than the minimum (2500K) to that value
                        sigmaNew[:T_lower_ind + 1, :, :, iMol] = sigma_metals[i, 0, :, ind]
                    else:
                        T_ind = find_nearest(self.LookUpTGrid, T)
                        sigmaNew[T_ind, :, :, iMol] = sigma_metals[i, 0, :, ind]

                # sigmaNew[:,:,:,iMol] = sigma_metals[:,:,ind]
            # if not in either table, raise error
            else:
                raise ValueError('LookUpQuickRead does not contain: ' + MolName)

        #     # otherwise raise error that molecules is not contained in opacity file
        #     if len(ind)==0:
        #         raise ValueError('LookUpQuickRead does not contain: '+MolName)
        #     else:
        #         ind=ind[0]
        #     sigmaNew[:,:,:,iMol] = sigma[:,:,:,ind]

        sigma = sigmaNew

        #Interpolate the entire table on the pressure levels used in the code (this allows for 1D interpolation (rather than 2D) when the code runs)
        print('Interpolating opacity look-up table onto pressure grid ...')
        dim=np.array(sigma.shape); dim[1]=self.nLay
        self.LookUpSigma = np.zeros(dim, dtype=self.numerical_precision)
        for iLay,pLay in enumerate(self.p-1):
            if pLay<LookUpPGrid[0]:
                self.LookUpSigma[:,iLay,:,:] = sigma[:,0,:,:]
            elif pLay>LookUpPGrid[-1]:
                self.LookUpSigma[:,iLay,:,:] = sigma[:,-1,:,:]
            else:
                ind = bisect(LookUpPGrid,pLay) - 1
                w = (pLay-LookUpPGrid[ind]) / (LookUpPGrid[ind+1]-LookUpPGrid[ind])
                self.LookUpSigma[:,iLay,:,:] = (1-w) * sigma[:,ind,:,:] + w * sigma[:,ind+1,:,:]

        self.LookUpSigma = np.moveaxis(self.LookUpSigma,[0,1,2,3],[0,1,3,2])

        self.sigmaAtm = np.zeros([self.nLay,self.nAbsMol,self.nWave], dtype=self.numerical_precision)

        if 0:
            #Plot LUT
            fig,ax=plt.subplots(dpi=30, figsize=(30,20))
            for iMol,MolName in enumerate(self.AbsMolNames):
                ax.plot(self.wave,self.LookUpSigma[3,20,iMol,:],label=MolName,color=self.colors[MolName],lw=0.5)
            ut.xspeclog(ax,level=1)
            ax.set_yscale('log')
            ax.legend(fontsize='xx-small')
            fig.savefig(self.filebase+'_OpacitiesLUT.pdf')
        f.close()


    def prepare_1D_CrossSecLUT_old(self,MoleculeListFile,waveRange,resolution):
        
        self.MoleculeListFile=MoleculeListFile
        self.waveRange=waveRange
        self.resolution=resolution
        
        print(waveRange[0],waveRange[1],resolution)
        print(os.getcwd())
        
        self.LUTFile = self.scarletpath+'/'+'../scarlet_LookUpQuickRead/'+os.path.basename(self.MoleculeListFile)+'_{:g}_{:g}_{:g}.mat'.format(waveRange[0],waveRange[1],resolution) 
        print(self.LUTFile)
        #print os.path.isfile(self.LUTFile)
        
        #Read in Look up table for opacity
        try:
            f = h5py.File(self.LUTFile)
            self.wave=10000/np.array(f['nu']).flatten()[::-1]
            self.nWave=len(self.wave)
        except:
            raise ValueError('ERROR: Look Up Table file not in scarlet_LookUpQuickRead or busted! Check file size!')  
        
        self.LookUpTGrid = np.array(f['LookUpTGrid']).flatten()
        LookUpPGrid = np.array(f['LookUpPGrid']).flatten()

        sigma = np.array(f['sigma_mol'])[:,:,::-1,:]

        #Find out what molecules are saved in LookUpTable
        LookUpMolNames = []
        for iMol in range(len(f['MolNames'])):
            LookUpMolNames.append(''.join(chr(i) for i in f[f['MolNames'][iMol][0]][:]))
        LookUpMolNames=np.array(LookUpMolNames)
        LookUpAbsMol=np.array(f['AbsMol']).flatten().astype(int)-1
        LookUpMolNames=LookUpMolNames[LookUpAbsMol]
        
        #Build up LookUpSigma for the selected absorbing molecules self.AbsMolNames
        print('Selecting molecules in opacity look up table.')
        dim=np.array(sigma.shape); dim[3]=self.nAbsMol
        sigmaNew = np.zeros(dim, dtype=self.numerical_precision)
        for iMol,MolName in enumerate(self.AbsMolNames):
            ind=np.where(MolName==LookUpMolNames)[0][0]
            sigmaNew[:,:,:,iMol] = sigma[:,:,:,ind]
        sigma=sigmaNew

        #Interpolate the entire table on the pressure levels used in the code (this allows for 1D interpolation (rather than 2D) when the code runs)
        print('Interpolating opacity look-up table onto pressure grid ...')
        dim=np.array(sigma.shape); dim[1]=self.nLay
        self.LookUpSigma = np.zeros(dim, dtype=self.numerical_precision)
        for iLay,pLay in enumerate(self.p):
            if pLay<LookUpPGrid[0]:
                self.LookUpSigma[:,iLay,:,:] = sigma[:,0,:,:]
            elif pLay>LookUpPGrid[-1]:
                self.LookUpSigma[:,iLay,:,:] = sigma[:,-1,:,:]
            else:
                ind = bisect(LookUpPGrid,pLay) - 1
                w = (pLay-LookUpPGrid[ind]) / (LookUpPGrid[ind+1]-LookUpPGrid[ind])
                self.LookUpSigma[:,iLay,:,:] = (1-w) * sigma[:,ind,:,:] + w * sigma[:,ind+1,:,:]

        self.LookUpSigma = np.moveaxis(self.LookUpSigma,[0,1,2,3],[0,1,3,2])

        self.sigmaAtm = np.zeros([self.nLay,self.nAbsMol,self.nWave], dtype=self.numerical_precision)








        
    #%% Labeling

    def createRunName(self,params,runName):
        #Update inputs of last run
        if isinstance(runName, list):
            self.runName=''            
            for key in runName:
                if key=='Mp':     
                    self.runName=self.runName+'_'+key+'{:05.1f}'.format(params[key]/Mearth)
                elif key=='pCloud':                
                    self.runName=self.runName+'_'+key+str(params[key]/100)+'mbar'
                elif key=='carmaFile':                
                    self.runName=self.runName+'_'+key+os.path.basename(params[key])
                elif key=='WellMixed':                
                    self.runName=self.runName+'_'+key
                    for MolName in sorted(params['qmol'], key=params['qmol'].get, reverse=True)[0:3]:
                        if params['qmol'][MolName]>0:
                            self.runName=self.runName+'_'+MolName+'_{:g}'.format(params['qmol'][MolName])    
                elif key=='WellMixed_dissociation':
                    self.runName=self.runName+'_'+key
                    for MolName in sorted(params['qmol'], key=params['qmol'].get, reverse=True)[0:6]:
                        if params['qmol'][MolName]>0:
                            self.runName=self.runName+'_'+MolName+'_{:g}'.format(params['qmol'][MolName])
                elif key=='SetProfiles':
                    self.runName=self.runName+'_'+key
                elif key=='option':  
                    if params[key]!='':
                        self.runName=self.runName+'_'+str(params[key])
                else:
                    self.runName=self.runName+'_'+key+str(params[key])
            self.runName=self.runName[1:]
        else:        
            self.runName = runName
        return self.runName 


    def autoLabel(self,include=[],standard=True,stdtype='legend'):
        txts=[]

#        txtspec=[]
#        for spec in self.specs:
#            if spec.meta['spectype']=='dppm':
#                txtspec.append('Transit')
#            if spec.meta['spectype']=='thermal':
#                txtspec.append('Thermal')
#        txts.append('+'.join(txtspec))
                
        if standard is True:
            if stdtype=='legend':
                if self.modelSetting['ComposType']=='WellMixed':
                    txts.append(str(self.params['qmol'])[1:-1])
                if self.modelSetting['ComposType'] == 'WellMixed_dissociation':
                    txts.append('Free Retrieval Dissociation')
                if self.modelSetting['ComposType']=='ChemEqui':
                    #txts.append('Chem. Consistent')
                    txts.append('{:.0f} x solar metallicity, C/O = {:.2f}'.format(self.params['Metallicity'],self.params['CtoO']))
                    #params['pQuench']
        
                if self.modelSetting['CloudTypes'][0]:
                    txts.append('Gray Clouds at '+'{:.1f} mbar'.format(self.params['pCloud']/100))
                if self.modelSetting['CloudTypes'][2]:
                    txts.append('Mie Clouds at '+'{:.1f} mbar'.format(self.params['miePAtTau1']/100))
            elif stdtype=='short':
                if self.modelSetting['ComposType']=='WellMixed':
                    txts.append(str(self.params['qmol'])[1:-1].replace("'","").replace(":","_").replace(" ","").replace(",","_"))
                if self.modelSetting['ComposType']=='WellMixed_dissociation':
                    txts.append('Free Retrieval Dissociation')
                if self.modelSetting['ComposType']=='ChemEqui':
                    txts.append('M{:.0f}C{:.2f}'.format(self.params['Metallicity'],self.params['CtoO']))
                if self.modelSetting['CloudTypes'][0]:
                    txts.append('{0:g}mbar'.format(self.params['pCloud']/100))
                if self.modelSetting['CloudTypes'][2]:
                    txts.append('Mie Clouds at '+'{:.1f} mbar'.format(self.params['miePAtTau1']/100))

            elif stdtype=='TeqMet':
                txts.append('T{0:04.0f}M{1:05.0f}C{2:04.2f}g{3:05.1f}'.format(self.Teq,self.params['Metallicity'],self.params['CtoO'],self.grav[self.iLevRpRef]))
                

        if ('Metallicity' in self.params) and ('Metallicity' in include):
            txts.append('M={}'.format(self.params['Metallicity']))

        if ('Metallicity' in self.params) and ('O/H' in include):
            txts.append('[O/H] ={0:4.0f} x solar'.format(self.params['Metallicity']))

        if ('Tp' in self.params) and ('Tp' in include):
            txts.append('{:s}'.format(self.params['Tp']))

        if ('BondAlbedo' in self.params) and ('BondAlbedo' in include):
            txts.append('A={:g}'.format(self.params['BondAlbedo']))

        if 'EquivBondAlbedo' in include:
            txts.append('A={:g}'.format(self.EquivBondAlbedo))

        #label=', '.join(txts)
        label='_'.join(txts)

        if 'BestFit' in self.filename:
            label=label+' (Best fit, '
            if self.modelSetting['ComposType']=='WellMixed':
                label=label+'Free Retrieval'
            if self.modelSetting['ComposType']=='WellMixed_dissociation':
                label=label+'Free Retrieval Dissociation'
            if self.modelSetting['ComposType']=='ChemEqui':
                label=label+'Chem. Consistent'
            label=label+', '+self.modelSetting['TempType']
            label=label+')'
        
        
        
        
        return label


    #%% Run Model Methods

    def runModel(self,modelSetting,params,
                 fitRadius=False,specsfitRadius=None,shiftDppm=False,fixedShiftDppm=None,dDppmMax=100,
                 runName='',
                 disp2terminal=False,returnVals=False,saveToFile=False,saveMatlabInputs=False, printFluxInfo=False):
        '''
        calls forward model
        all input parameters are the direct values (no log scale)
        
        params=dict()
        params['Rp']            = pla.Rp #3.870923*Rearth #
        params['Rstar']         = pla.Rstar
        params['Teffstar']      = pla.Teffstar
        params['Mp']            = pla.Mp
        params['ap']            = pla.ap
        params['Tint']          = 75   #pla.Tint
        params['HeatDistFactor']= 0.25 
        params['BondAlbedo']    = 0.1  
        params['GeometricAlbedo']= 0.1
        params['GroundAlbedo']  = 0
        params['pCloud']        = 100*bar
        params['cHaze']         = 1e-10
        params['Temp']          = 800

        params['mieRpart']         = 0.2
        params['miePAtTau1']       = 8*mbar 
        params['mieRelScaleHeight'] = 0.3
        '''

        self.createRunName(params,runName)
        self.modelSetting = modelSetting
        self.params = params
        self.RpOriginal = deepcopy(self.params['Rp'])
        self.iRun=self.iRun+1
        self.nonGrayIter = 0

        
        #---List of all input parameters for Atmosphere Code-------------------------
        self.Rstar=params['Rstar']
        self.Teffstar=params['Teffstar']   
        self.Rp=params['Rp'] 
        self.Mp=params['Mp']
        self.HeatDistFactor=params['HeatDistFactor']    # 0.25=incoming flux is distributed over 4*pi*Rp**2  # 0.5=incoming flux is distributed over 2piRp**2
        self.BondAlbedo=params['BondAlbedo'] 
        if 'GeometricAlbedo' in params:
            self.GeometricAlbedo=params['GeometricAlbedo']
        self.GroundAlbedo=params['GroundAlbedo']  
        self.pCloud=params['pCloud']
        self.cHaze=params['cHaze'] 

        if 'Teq' in params:
            params['ap'] = self.Rstar * self.Teffstar**2 / params['Teq']**2 * np.sqrt(  self.HeatDistFactor * (1 - self.BondAlbedo ) )
            print('Using user specified Teq to adjust semi-major axis ap')
        self.ap=params['ap']

        if np.isnan(self.Rp):
            raise ValueError('ERROR: Rp is nan" !')
        if np.isnan(self.Mp):
            raise ValueError('ERROR: Mp is nan" !')


        #---Stellar irradiation--------------------------------------------------------------------

        #Flux at stellar surface
        self.calcfStarSurf(params)
            # self.fStarSurf = self.fStarSurf * self.params['Lstar'] / L
        
        #saving full stellar model if provided (for VULCAN)
        if 'stellarDataFlux' in self.params:
            if 'Lstar' in self.params:
                #self.stellarDataFlux = self.params['stellarDataFlux']*self.params['Lstar']/L
                self.stellarDataFlux = self.params['stellarDataFlux']
            else:
                self.stellarDataFlux = self.params['stellarDataFlux']
            self.stellarDataWave = self.params['stellarDataWave']
            self.IrradStarFullRes = self.stellarDataFlux*(self.Rstar/self.ap)**2
        
        #irradiance at subsolar point
        self.IrradStar = self.fStarSurf*(self.Rstar/self.ap)**2   
        self.totalIncFluxAtSubSolarPoint = np.trapz(x=self.wave,y=self.IrradStar)
        if printFluxInfo:
            print('\nTotal Incident Flux at subsolar point:      ', self.totalIncFluxAtSubSolarPoint , 'W/(m**2)')
        
        #average irradiance for this 1D model
        self.IrradStarEff = self.IrradStar * self.HeatDistFactor * (1-self.BondAlbedo)                                  
        self.totalEffIncFlux = np.trapz(x=self.wave,y=self.IrradStarEff)
        if printFluxInfo:
            print ('Total Effective Incident Flux for 1D model: ', self.totalEffIncFlux , 'W/(m**2)')

        #Feautrier methods needs it in intensity (not flux) and in 'W/(m**2*Hz)'
        self.IrradStarEffIntensityPerHz=rad.convertIntensity(self.IrradStarEff,self.wave,'W/(m**2*um)','um','W/(m**2*Hz)') / np.pi     

        #Equilibrium temperature
        self.Teq = self.Teffstar*(self.Rstar/self.ap)**(0.5)*   (self.HeatDistFactor*(1-self.BondAlbedo))**(0.25)
        if printFluxInfo:
            print('Equilibrium Temperature:', self.Teq, 'K')

        self.TeqIncFlux = (self.totalEffIncFlux / sigmaSB)**0.25
        if printFluxInfo:
            print('Equilibrium Temperature:', self.TeqIncFlux, 'K (only inc flux in wave range)\n')

        self.Teq = self.TeqIncFlux
        
        if 'Teq' in params:
            self.Teq=params['Teq']
        
        #sys.exit()
        
        #---Read in clouds inputs-----------------------------------------------------------------------        
        if modelSetting['CloudTypes'][3]==1:
            self.readCarmaFile(self.params['carmaFile'])



        #---If fitRadius, repeatedly run until Rp is a good match to the data in specsfitRadius----------
        self.RpFittedRadius=np.nan
        if fitRadius:
            
            self.prepInstrResp(specsfitRadius, low_res_mode=self.fitRadiusLowRes)

            try:
                #Do fast optimization of Rp (can fail in very rare exceptional circumstances)
                dDppm=1e10
                counter=0
                while np.abs(dDppm)>dDppmMax:
                    counter=counter+1
    
                    #Run Atmosphere Model
                    modelSettingFitRadius=deepcopy(modelSetting)
                    modelSettingFitRadius['thermalReflCombined']=[]         # 'Disort'; 'Feautrier'; 'Toon'
                    modelSettingFitRadius['thermalOnly']        =[] # 'NoScat'; 'MuObs'; 'Toon'; 'Feautrier'; 'thermalNoScatMol'
                    modelSettingFitRadius['albedoOnly']         =[] # 'Disort'; 'Toon'; 
                    modelSettingFitRadius['transit']            =[]         # 'dppmMol'
                    
                    self.calcAtmosphere(modelSettingFitRadius,params,updateOnlyDppm=True,low_res_mode=self.fitRadiusLowRes)
                    astromodels=self.instrResp(specsfitRadius,low_res_mode=self.fitRadiusLowRes)
                    
                    #self.plotSpectrum(ax=self.ax,save=True,spectype='dppm',specs=specsfitRadius,resPower=500,xscale='log',presAxis=False,presLevels=False,label=str(counter))
                                    #self.fig.savefig(self.filebase+self.runName+'_Spectrum_Test.pdf')            
    
                    #Determine new Rp
                    RpSuggest,dDppm,self.chi2 = calcNewRadiusToMatchDppms(specsfitRadius,astromodels,params['Rp'],params['Rstar'])
                    
                    if counter<7:
                        if RpSuggest<0.8*params['Rp']:
                            newRp=params['Rp']*0.8
                            print('Max correction: Rp --> 0.8*Rp')
                        elif RpSuggest>1.2*params['Rp']:
                            newRp=params['Rp']*1.2
                            print('Max correction: Rp --> 1.2*Rp')
                        else:
                            newRp=params['Rp'] + (RpSuggest-params['Rp'])*1.0
                    elif counter<15:
                        if RpSuggest<0.8*params['Rp']:
                            newRp=params['Rp']*0.8
                            print('Max correction: Rp --> 0.8*Rp')
                        elif RpSuggest>1.2*params['Rp']:
                            newRp=params['Rp']*1.2
                            print('Max correction: Rp --> 1.2*Rp')
                        else:
                            print('Damped Rp correction (0.2)')
                            newRp=params['Rp'] + (RpSuggest-params['Rp'])*0.2
                    else:
                        newRp = minimize(chi2AsFunctionOfRp,self.RpOriginal,args=(self,specsfitRadius),method='Nelder-Mead',options={'maxfev':30,'xatol': 0.1})#,options={'disp': True}) #, ,options={'gtol': 1e-6, 'disp': True})
                        newRp =newRp['x'][0]
                        fitRadius=False
                        shiftDppm=False
                        dDppm=0.0
    
                    # print('Change Rp: {:f} --> {:f} Rearth | shiftDppm={:f}  (chi2={:f})'.format(params['Rp']/Rearth, newRp/Rearth, dDppm, self.chi2))

                    params['Rp']       =newRp
                    self.Rp            =newRp
                    self.RpFittedRadius=newRp

            except:
                #This part may be not necessary anymore:
                #Do advanced optimzation of Rp (slower), but sometimes necessary when the fast Rp optimization leads to an error
                print('!!!! WARNING: Doing advanced optimzation of Rp (slower) because fast optimization led to an error !!!!')
                print('Last Rp/Rearth attempted:'+str(params['Rp']/Rearth))
                params['Rp']=deepcopy(self.RpOriginal)
                print('New guess for Rp/Rearth:'+str(params['Rp']/Rearth))

                t0 = time()
                self.calcAtmosphere(modelSetting,params,updateOnlyDppm=True,low_res_mode=self.fitRadiusLowRes)
                astromodels=self.instrResp(specsfitRadius,low_res_mode=self.fitRadiusLowRes)
                print('\ncalcAtmosphere took {} seconds'.format(str( np.round((time()-t0),3))))

                newRp = minimize(chi2AsFunctionOfRp,self.RpOriginal,args=(self,specsfitRadius),method='Nelder-Mead',options={'maxfev':30,'xatol': 0.1})#,options={'disp': True}) # ,options={'gtol': 1e-6, 'disp': True})
                newRp =newRp['x'][0]

                params['Rp']       =newRp
                self.Rp            =newRp
                self.RpFittedRadius=newRp

                fitRadius=False
                shiftDppm=False
                dDppm=0.0
            
            if modelSetting['maxNonGrayIter'] is not None:
                self.nonGrayIter = modelSetting['maxNonGrayIter']-1 #Allow 1 more iteration of 
        
        if self.fitRadiusLowRes:
            self.prepInstrResp(specsfitRadius)
            
        #---Calculate full atmospheric scenario and spectra---------------------------------------------------------
        self.calcAtmosphere(modelSetting,params)   # params['Rp'] containts the best fitting radius now, if it was fitted
        
        if shiftDppm:
            if fixedShiftDppm is None:
                astromodels=self.instrResp(specsfitRadius)
                RpNew,self.appliedShiftDppm,self.chi2 = calcNewRadiusToMatchDppms(specsfitRadius,astromodels,params['Rp'],params['Rstar'])
                print('Rp={:f}, Rp_BestFit={:f} Rearth | appliedShiftDppm={:f}  (chi2 = {:f})'.format(params['Rp']/Rearth, RpNew/Rearth, self.appliedShiftDppm, self.chi2))
            else:
                self.appliedShiftDppm = fixedShiftDppm
            self.dppm=self.dppm+self.appliedShiftDppm
            print('          shift applied to dppm')
            if 'dppmMol' in modelSetting['transit']:
                self.dppmMol=self.dppmMol+self.appliedShiftDppm
                print('          shift applied to dppmMol')
                
                

#        #Broadening
#        if params['resPower'] is not None:
#            kernel=Gaussian1DKernel(self.resPower / params['resPower'] / 2.35)   # 2.35 because FWHM = 2.35 standard deviation
#            self.dppm   =convolve(self.dppm   ,kernel)
#            self.thermal=convolve(self.thermal,kernel)

        #---Compute addititonal quantities---------------------------------------------------------
        if self.doThermal:
            self.calcSecEclppm()
        else:
            self.secEclppm = np.ones(self.nWave, dtype=self.numerical_precision)

        if saveToFile:
            self.save()
        if returnVals:
            return self.wave,self.dppm,self.secEclppm,self.thermal,self.T,self.qmol_lay,self.modelRpRs


    def calcAtmosphere(self,modelSetting,params,updateOnlyDppm=False,updateTp=True,disp2terminal=False,returnOpac=False,
                       thermalReflCombined=None,thermalOnly=None,albedoOnly=None, low_res_mode=False):

        if self.verbose:
            print('\nStarting calcAtmosphere() with detailed print statements')
            print('TempType = ', modelSetting['TempType'])
            print('ComposType = ', modelSetting['ComposType'])
        

        t0 = time()
        
        #---Calculate Atmospheric Structure (iterate)-----------------------------------
        if updateTp is True:
            self.T   = self.Teq * np.ones_like(self.p, dtype=self.numerical_precision)
        
        # Loop twice iff we use ChemKinetic + NonGray ( 1st calcTp() in ChemEqui regime )
        if modelSetting['ComposType']=='ChemKinetic' and (modelSetting['TempType']=='NonGray' or modelSetting['TempType']=='NonGrayConv' or modelSetting['TempType']=='NonGrayTrad'):
            nbIterations = 2
        else:
            nbIterations = 1
            
        if self.verbose: print('\nNumber of iterations is set to ', nbIterations, '\n')
        
        # Self-consistent loop
        for iLoop in range(nbIterations):
            
            if self.verbose: print('calcAtmosphere loop ', iLoop, '\n')
            
            if iLoop==0 and modelSetting['ComposType']=='ChemKinetic':
                if self.verbose: print('For ChemKinetic case, do first calcTpProfile() in ChemEqui')
                if self.verbose: print('Set ComposType to ChemEqui\n')
                
                #make sure 1st calcTp() is in ChemEqui regime
                finalComposType = modelSetting['ComposType']
                modelSetting['ComposType'] = 'ChemEqui'
                
                if self.verbose: print('Going to calcTp()\n')
                self.T                                                                                      = self.calcTpProfile(modelSetting,params,firstIter=(iLoop==0) )
                
                if self.verbose: print('Set ComposType back to ChemKinetic\n')
                #set desired chemical regime back
                modelSetting['ComposType'] = finalComposType
            else:
                if self.verbose: print('Going to calcTp()\n')
                if updateTp is True:
                    self.T                                                                                      = self.calcTpProfile(modelSetting,params,firstIter=(iLoop==0) )
                        #            print '!!! MAKING AD-HOC CONVECTIVE ADJUSTMENT !!!!!!!!'
                        #            self.qmol_lay                                                                               = self.calcComposition(modelSetting,params,self.T)
                        #            self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r = self.calcHydroEqui(modelSetting,params,self.T)
                        #            self.plotTz(makeConvAdjust=True)
                        #            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            if self.verbose: print('Going to calcComposition() in calcAtmosphere loop\n')
            self.qmol_lay                                                                               = self.calcComposition(modelSetting,params,self.T, firstIter=(iLoop==0))
            if self.verbose: print('Going to executeOption(), calcHydroEqui and calcOpacities in calcAtmosphere()\n')
            self.executeOption()
            self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r = self.calcHydroEqui(modelSetting,params,self.T)
            self.extinctCoef,self.absorbCoef, self.scatCoef                                             = self.calcOpacities(modelSetting,params,self.T,saveOpac=self.saveOpac, low_res_mode=low_res_mode)

        
        if self.verbose: print('Iterative modeling of Composition, HydroEqui, and Opacities finished!!!\n')
        
        #---Calculate spectra for here on----------------------------------------
        if self.doTransit:
            # t0 = time()
            self.dppm                = self.calcTransitSpectrum(self.extinctCoef,saveOpac=self.saveOpac)
            # print('calcTransitSpectrum took {} seconds'.format(str( np.round((time()-t0),3))))
        else:
            self.dppm = np.ones(self.nWave, dtype=self.numerical_precision)

        if updateOnlyDppm is False:        
            #Convert to correct units (wave and dppm were converted above)
            self.modelRpRs=params['Rp']/params['Rstar']
            
            #---Thermal + Reflected Flux combined--------------------------            
            if 'Disort' in modelSetting['thermalReflCombined']:
                extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
                scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
                self.totalFluxDisort = self.calcDisort(self.IrradStar,extinctCoef,scatCoef,thermal = True)*np.pi #division by irradstar done in calcDisort method
            if 'Feautrier' in modelSetting['thermalReflCombined']:
                B,J,K,H = self.solveRTE(self.T,modelSetting,params,self.IrradStarEffIntensityPerHz)
                IdownPerHz = self.IrradStarEffIntensityPerHz
                IupPerHz   = 4.0 * H[0,:] - IdownPerHz
                self.totalFluxFeautrier = rad.convertIntensity(IupPerHz,self.wave,InputUnit='W/(m**2*Hz)',WavelengthUnit='um',OutputUnit='W/(m**2*um)')*np.pi
                
            if 'Toon' in modelSetting['thermalReflCombined']:
                #extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
                #scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
                #Fupw, Fdwn, Fnet = self.multiScatToon(self.IrradStar,extinctCoef,scatCoef,refl = True,thermal = True)
                Fupw, Fdwn= self.multiScatToon(self.IrradStar,self.extinctCoef,self.scatCoef,self.T,ref = True,therm = True)
                
                self.totalFluxToon = Fupw[0,:]
                
            #---Thermal Only------------------------------------------------            
            self.muObs = np.array([1/np.sqrt(3)])

            if 'NoScat' in modelSetting['thermalOnly']:
                if self.doThermal:
                    self.thermalNoScat = np.pi * self.calcEmissionSpectrum(self.extinctCoef,muObs=self.muObs,saveOpac=self.saveOpac)
                else:
                    self.thermalNoScat = np.ones(self.nWave, dtype=self.numerical_precision)
            if 'MuObs' in modelSetting['thermalOnly']:
                self.muObs=np.r_[1/np.sqrt(3),np.arange(1,0,-0.1),0.05,0.01]
                self.thermalMuObs = self.calcEmissionSpectrum(self.extinctCoef,muObs=self.muObs)
            if 'Toon' in modelSetting['thermalOnly']:
                #extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
                #scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
                Fupw, Fdwn= self.multiScatToon(self.IrradStar,self.extinctCoef,self.scatCoef,self.T,ref = False,therm = True)
                self.thermalToon = Fupw[0,:] 
            if 'Feautrier' in modelSetting['thermalOnly']:
                B,J,K,H = self.solveRTE(self.T,modelSetting,params,0.0*self.IrradStarEffIntensityPerHz)
                self.thermalFeautrier = rad.convertIntensity(H[0,:],self.wave,InputUnit='W/(m**2*Hz)',WavelengthUnit='um',OutputUnit='W/(m**2*um)')*4.0*np.pi
            if 'thermalNoScatMol' in modelSetting['thermalOnly']:
                self.thermalNoScatMol=np.zeros([self.nAbsMol,self.nWave], dtype=self.numerical_precision)
                for iMol,MolName in enumerate(self.AbsMolNames):
                    extinctCoef,absorbCoef,scatCoef  = self.calcOpacities(modelSetting,params,self.T,iMol=iMol)
                    self.thermalNoScatMol[iMol,:]    = self.calcEmissionSpectrum(extinctCoef,muObs=self.muObs)*np.pi       
                
            #---Albedo only-------------------------------------------------            
            if 'Disort' in modelSetting['albedoOnly']:
                extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
                scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
                self.albedoDisort = self.calcDisort(self.IrradStar,extinctCoef,scatCoef,thermal = False) #division by irradstar done in calcDisort method

            if 'Feautrier' in modelSetting['albedoOnly']:
                B,J,K,H = self.solveRTE(0.0*self.T,modelSetting,params,self.IrradStarEffIntensityPerHz)
                IdownPerHz = self.IrradStarEffIntensityPerHz
                IupPerHz   = 4.0 * H[0,:] - IdownPerHz
                self.albedoFeautrier = (IupPerHz*np.pi)/(IdownPerHz*np.pi)
                
            if 'Toon' in modelSetting['albedoOnly']:
                #extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
                #scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
                # ?IrradStar = np.ones_like(self.wave, dtype=self.numerical_precision)
                Fupw, Fdwn=self.multiScatToon(self.IrradStar,self.extinctCoef,self.scatCoef,self.T,ref = True,therm = False)
                #Fupw, Fdwn, Fnet = self.multiScatToon(IrradStar,extinctCoef,scatCoef,refl = True, thermal = False, method = 'Eddington')
                self.albedoToon = Fupw[0,:] / self.IrradStar
            
            #---Transit Depth by molecular---------------------------------            
            if 'dppmMol' in modelSetting['transit']:
                self.dppmMol=np.zeros([self.nAbsMol,self.nWave], dtype=self.numerical_precision)
                for iMol,MolName in enumerate(self.AbsMolNames):
                    extinctCoef,absorbCoef,scatCoef  = self.calcOpacities(modelSetting,params,self.T,iMol=iMol)
                    self.dppmMol[iMol,:]             = self.calcTransitSpectrum(extinctCoef)


            # multiplicative factor to the thermal flux to parametrize a hot spot (should be slightly less than 1)
            if 'diffFactor' in modelSetting.keys():
                if modelSetting['diffFactor']:
                    print('diffFactor = {}'.format(self.params['diffFactor']))
                    self.thermalNoScat *= self.params['diffFactor']

            #---Choose what goes into self.thermal and self.albedoSpec-------
            if thermalOnly is None:
                if self.modelSetting['thermalOnly']==[]:
                    thermalOnly=None
                else:
                    thermalOnly=self.modelSetting['thermalOnly'][0]
                    if thermalOnly=='NoScat':
                        self.thermal=self.thermalNoScat
                    elif thermalOnly=='MuObs':
                        self.thermal=self.thermalMuObs
                    elif thermalOnly=='Toon':
                        self.thermal=self.thermalToon
                    elif thermalOnly=='Feautrier':
                        self.thermal=self.thermalFeautrier
                        
                        
            if albedoOnly is None:
                if self.modelSetting['albedoOnly']==[]:
                    albedoOnly=None
                else:
                    albedoOnly=self.modelSetting['albedoOnly'][0]
                    if albedoOnly=='Disort':
                        self.albedoSpec=self.albedoDisort
                    elif albedoOnly=='Feautrier':
                        self.albedoSpec=self.albedoFeautrier
                    elif albedoOnly=='Toon':
                        self.albedoSpec=self.albedoToon
                    elif albedoOnly=='SetGeometricAlbedo':
                        self.albedoSpec=self.GeometricAlbedo*np.ones_like(self.wave, dtype=self.numerical_precision)
                        
            
            #Calculate calc_dmodeldT
            if modelSetting['calcTmodModels'][1]:
                print('Calculating calc_dmodeldT')
                self.thermalTmod = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
                for iLay in range(self.nLay):
                    print('{:d}/{:d}'.format(iLay,self.nLay))
                    T=deepcopy(self.T)
                    T[iLay]=T[iLay]+1
                    extinctCoef,absorbCoef,scatCoef = self.calcOpacities(modelSetting,params,T)
                    thermal = self.calcEmissionSpectrum(extinctCoef)
                    self.thermalTmod[iLay,:]=thermal
                    
        #For later reference
        self.params=params

        t1 = time()
        # stdout.write('          ({:f} sec) '.format(t1-t0))
        stdout.write('          ({} sec) '.format(str(np.round(t1-t0,3))))




    #%% Physical Modeling (highest level)
    def calcHydroEqui(self,modelSetting,params,T):

        #Note: All values without "grid" in variable name are at cell centers
        
        #array of molar masses (self.MolarMass) from readMoleculesProperties() above
        MuAve=uAtom*np.sum(self.MolarMass[np.newaxis,:]*self.qmol_lay,axis=1)  #[kg]
        
        #don't need to define p since already have self.p
        ntot=self.p/(kBoltz*T) # [m^-3]
        
        nmol=((ntot[:,np.newaxis]).dot(np.ones([1,self.nMol], dtype=self.numerical_precision)))*self.qmol_lay #(iLev) in [m^-3]
        
        r               =np.zeros(self.nLay, dtype=self.numerical_precision)   # [m]
        grav            =np.zeros(self.nLay, dtype=self.numerical_precision)   # [m]
        scaleHeight     =np.zeros(self.nLay, dtype=self.numerical_precision)   # [m]
        dz              =np.zeros(self.nLay, dtype=self.numerical_precision)   # [m]
        
        r[self.iLevRpRef]=deepcopy(params['Rp'])
        
        #Atmosphere below r[self.iLevRpRef]=self.Rp to higher pressures:
        for iLay in range(self.iLevRpRef,self.nLay-1):
                for repeat in range(1,5):
                    grav[iLay]        = G*self.Mp / ( r[iLay] - 0.5*dz[iLay] )**2                                     # [m/s^2]
                    scaleHeight[iLay] = (kBoltz*T[iLay])/(MuAve[iLay]*grav[iLay])                 # [m]
                    dz[iLay]          = -scaleHeight[iLay]*np.log(self.p[iLay]/self.p[iLay+1]) # [m]
                r[iLay+1]     = r[iLay] - dz[iLay]
        #Atmosphere above r[self.iLevRpRef]=self.Rp to lower pressures:
        for iLay in range(self.iLevRpRef,0,-1):
                for repeat in range(1,5):
                    grav[iLay-1]        = G*self.Mp / ( r[iLay] + 0.5*dz[iLay-1] )**2                                     # [m/s^2]
                    scaleHeight[iLay-1] = (kBoltz*T[iLay-1])/(MuAve[iLay-1]*grav[iLay-1])         # [m]
                    dz[iLay-1]          = -scaleHeight[iLay-1]*np.log(self.p[iLay-1]/self.p[iLay]) # [m]
                r[iLay-1]     = r[iLay] + dz[iLay-1]
                
                   
  

        ## --> at this point it has determined r completely (everything else can now be derived from that)
        
        RpBase=r[-1]
        
        z=r-r[-1]         # [m]
        dz=np.zeros(self.nLay-1, dtype=self.numerical_precision)   # [m]
        dz=-np.diff(z)            # [m]   
        
        grav=G*self.Mp /  r**2        # [m/s^2]
        scaleHeight=(kBoltz*T)/(MuAve*grav) # [m]
        
        #consistency check
#        scaleHeightLay = 0.5*(scaleHeight[:self.nLay-1]+scaleHeight[1:self.nLay])
#        dz2=-scaleHeightLay*np.log(self.p[:-1]/self.p[1:]) # [m]
#        print (dz2-dz)/dz*100

        return z,dz,grav,ntot,nmol,MuAve,scaleHeight,RpBase,r


    def calcTpProfile(self,modelSetting,params,firstIter=False,LucyUnsold=False,runConvection=False):
        '''
        Computes temperature-pressure profile
        --------------------------
        calcTpProfile(self,modelSetting,params,LucyUnsold = False,convection = False,stop = 100)
        LucyUnsold -> Calculates TP for brown dwarf
        Convection -> Turns on convection
        stop -> Maximum number of iterations allowed for NonGray TP profile calculation
        Returns T for every layer
        '''
        
        if modelSetting['TempType']=='parameters' or modelSetting['TempType']=='TintHeatDist':
            T=np.interp(np.log(self.p),np.log(params['Tprof'][0]),params['Tprof'][1])
        
        elif modelSetting['TempType']=='TeqUniform':
            T=self.Teq*np.ones_like(self.p, dtype=self.numerical_precision)

        elif modelSetting['TempType']=='TeqUniformFactor':
            T=self.Teq*np.ones_like(self.p, dtype=self.numerical_precision)*0.8

        elif modelSetting['TempType']=='FreeUniform':
            T=params['Temp']*np.ones_like(self.p, dtype=self.numerical_precision)
            
        elif modelSetting['TempType']=='FreeTPcustomLay':
            log_pressure_layers = np.log10(modelSetting['TPcustomLay'])
            T_on_layers = []
            for i in range(log_pressure_layers.size):
                if len(str(i))==1 : istr = '0'+str(i)
                else : istr = str(i)
                T_on_layers.append(params['T'+istr])
            T=np.interp(np.log10(self.p), log_pressure_layers, np.array(T_on_layers))
        
        elif modelSetting['TempType'][0:10]=='TpFreeProf':
            nTFreeLayers = int(modelSetting['TempType'][-2:])
            # log_pressure_layers = np.linspace(np.log10(1e-8*1e5),np.log10(1e2*1e5),nTFreeLayers) #nTFreeLayers from 1e4 to 1e-7 bar equally spaced in log
            # try to use given pressure bounds for fit TP
            try:
                log_pressure_layers = np.linspace(modelSetting['FreeProf_logp_toa'],modelSetting['FreeProf_logp_boa'],nTFreeLayers, dtype=self.numerical_precision)
            # otherwise use atmosphere bounds
            except:
                log_pressure_layers = np.linspace(np.log10(self.p[0]),np.log10(self.p[-1]),nTFreeLayers, dtype=self.numerical_precision)
            self.TpFreeProf_pressure_grid = log_pressure_layers
            T_on_layers = []
            for i in range(nTFreeLayers):
                if len(str(i))==1 : istr = '0'+str(i)
                else : istr = str(i)
                T_on_layers.append(params['T'+istr])
            
            if modelSetting['Tspline']==False:
                T=np.interp(np.log10(self.p), log_pressure_layers, np.array(T_on_layers, dtype=self.numerical_precision))
            else: #if T is spline interpolated
                Tinterp=CubicSpline(log_pressure_layers,np.array(T_on_layers, dtype=self.numerical_precision))
                TPinterp = Tinterp(np.log10(self.p[np.where(np.log10(self.p)>=log_pressure_layers[0])]))
                if np.any(TPinterp<0) or np.all(np.isfinite(TPinterp))==False:
                    T=np.interp(np.log10(self.p),log_pressure_layers,np.array(T_on_layers, dtype=self.numerical_precision))
                else:
                    T = np.ones(np.shape(self.p), dtype=self.numerical_precision)
                    T[np.where(np.log10(self.p)<log_pressure_layers[0])]=T_on_layers[0]
                    T[np.where(np.log10(self.p)>=log_pressure_layers[0])]=TPinterp

        elif modelSetting['TempType'][0:12]=='TtauFreeProf':
            
            nTFreeLayers = int(modelSetting['TempType'][-2:])
            log_tau_layers = np.linspace(np.log10(1e-8),np.log10(1e5),nTFreeLayers) #nTFreeLayers from tau=XX to XX equally spaced in log
            T_on_layers = []
            for i in range(nTFreeLayers):
                if len(str(i))==1 : istr = '0'+str(i)
                else : istr = str(i)
                T_on_layers.append(params['T'+istr])
            T_on_layers = np.array(T_on_layers)
            
            #first: T = uniform at Teq for first determination of p(tau)
            T=self.Teq*np.ones(np.shape(self.p), dtype=self.numerical_precision)

            for i in range(2):
                self.qmol_lay                                                                                = self.calcComposition(modelSetting,params,T)
                self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r  = self.calcHydroEqui(modelSetting,params,T)
                self.rho                                                                                     = self.ntot * self.MuAve
                self.extinctCoef,self.absorbCoef, self.scatCoef                                              = self.calcOpacities(modelSetting,params,T)
                
                dtau = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:]) * np.outer(self.dz,np.ones(self.nWave, dtype=self.numerical_precision))
        
                # Build grid of cumulative sums of tau from TOA to surface, starting at 0
                tau=np.vstack([np.zeros(len(self.wave)),np.cumsum(dtau,axis=0)], dtype=self.numerical_precision)
                # Ensure taugrid > 0
                tau=tau*(1+(np.arange(1,(self.nLay+1))[:,np.newaxis])*np.ones([1,self.nWave])*1e-10)+(np.arange(1,(self.nLay+1))[:,np.newaxis]*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-99)

                tauR = rad.rosselandMean(self.wave[self.indRM],tau[:,self.indRM],self.Teq,waveUnit='um')  # tauR(p) (Rosseland mean) ; indRM is to perform the mean over the wavelength range of the instruments
                ptau = np.interp(log_tau_layers,np.log10(tauR),self.p) #calculate p(tau) for the nTFreeLayers values of tau
                same_press = np.sum(np.diff(ptau)==0)
                
                if modelSetting['Tspline']==False:
                    if same_press:
                        T=np.interp(np.log10(self.p),np.log10(ptau[:-same_press]),T_on_layers[:-same_press])
                    else:
                        T=np.interp(np.log10(self.p),np.log10(ptau),T_on_layers) #T interpolated on scarlet pressure grid, ptau better each iteration
                
                else: #if spline interpolation  
                    if same_press:
                        Tinterp=CubicSpline(np.log10(ptau[:-same_press]),T_on_layers[:-same_press])
                    else:
                        Tinterp=CubicSpline(np.log10(ptau),T_on_layers)
                    
                    TPinterp = Tinterp(np.log10(self.p[np.where(self.p>=ptau[0])]))
                    if np.any(TPinterp<0) or np.all(np.isfinite(TPinterp))==False:
                        T=np.interp(np.log10(self.p),np.log10(ptau),T_on_layers)
                    else:
                        T[np.where(self.p<ptau[0])]=T_on_layers[0]
                        T[np.where(self.p>=ptau[0])]=TPinterp

        elif modelSetting['TempType']=='TpTwoVis':
            T = rad.TpTwoVis(self.p,Tint=params['Tint'],
                               kappaIR=params['kappaIR'],
                               gamma1=params['gamma1'],
                               gamma2=params['gamma2'],
                               alpha=params['alpha'],
                               beta=params['beta'],
                               Rstar=params['Rstar'],
                               Teffstar=params['Teffstar'],
                               ap=params['ap'],
                               gp=G*params['Mp']/((params['Rp'])**2))
            
            
        elif modelSetting['TempType']=='NonGray' or modelSetting['TempType']=='NonGrayConv':
        
            T = self.calcNonGrayTpProf(modelSetting,params,firstIter,LucyUnsold,runConvection)

        elif modelSetting['TempType']=='NonGrayTrad':

            T = self.calcNonGrayTpTradProf(modelSetting,params,firstIter,LucyUnsold,runConvection)
        
        else:
            raise ValueError('ERROR: UNKNOWN modelSetting["TempType"] !') 
        
        T = T.astype(self.numerical_precision)
        
        return T
        
        
    def calcComposition(self,modelSetting,params,T, firstIter=False):   

        if modelSetting['ComposType']=='ChemEqui':
            params['Metallicity']
            params['CtoO']
            params['pQuench']
            qmol_lay=self.InterpFromChemEquiGridLUT(self.p,T,params)
            
            # add quenching
            ind_pQuenched   = np.where(self.p<params['pQuench']) # pressures that are well mixed
            qmolQuenched    = qmol_lay[np.argmin(np.abs(self.p-params['pQuench'])),:] # composition of well mixed pressures
            qmol_lay[ind_pQuenched,:] = qmolQuenched

            if np.isnan(np.sum(qmol_lay)):
                print('***********WARNING************')
                print('Equilibrium chemistry returned nan values: replacing these with 0.0')
                print('Problematic species are:')
                bad_inds = np.where(np.isnan(np.sum(qmol_lay,axis=0)))[0]
                print(self.MolNames[bad_inds])
                qmol_lay = np.nan_to_num(qmol_lay)

                print('Overwriting these abundances with 1e-12')
                qmol_lay[:,bad_inds] = 1e-12
                print('******************************')

            totals = np.sum(qmol_lay,1) 
            if np.any(totals<0.95) or np.any(totals>1.01):
                print('!!! WARNING: Sum of Mixing Ratios <0.95 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Sum in in each layer:')
                print(totals)
                print('Metallicity={:g}, C/O={:g}, pQuench={:g}'.format(params['Metallicity'],params['CtoO'],params['pQuench']))
            
            qmol_lay = qmol_lay / totals[:,np.newaxis]

            
            
        elif modelSetting['ComposType']=='ChemKinetic':
            # Starts just as ChemEqui
            params['Metallicity']
            params['CtoO']
            params['pQuench']
            
            #if it's the first iteration, need to use ChemEqui to initialize qmol_lay
            try:
                mix_ratio = self.qmol_lay
            except:    
                qmol_lay=self.InterpFromChemEquiGridLUT(self.p,T,params)
                mix_ratio = qmol_lay
            
            
            # Then run VULCAN
            pressure = self.p
            planet = self.pla
            nLay = self.nLay
            molecules = self.MolNames
            
            # load the full stellar model to VULCAN if using a stellar model spectrum
            if 'stellarDataFlux' in params:
                wave_array = self.stellarDataWave
                IrradStar = self.IrradStarFullRes
            else: #load the scarlet wave and IrradStar
                wave_array = self.wave
                IrradStar = self.IrradStar
            
            try:
                nmol = self.nmol
            except:
                print('\nself.nmol not yet computed, set nmol to None for now')
                nmol = None
            
            vulcan_tol = modelSetting['vulcanTol']
            useDensity = vulcan_tol['useDensity']
            
            print('\nRunning VULCAN!')
            if firstIter==True:
                qmol_lay, self.vulcanComp = VULCAN.calcChemKinec(T, pressure, planet, nLay, molecules, mix_ratio, nmol, wave_array, IrradStar, vulcan_tol, write_cfg=True)
            else:
                qmol_lay, self.vulcanComp = VULCAN.calcChemKinec(T, pressure, planet, nLay, molecules, mix_ratio, nmol, wave_array, IrradStar, vulcan_tol, useDensity, write_cfg=False, makeChemFuns=False, lastConvergedy=self.vulcanComp)
            
            # add quenching
            ind_pQuenched   = np.where(self.p<params['pQuench']) # pressures that are well mixed
            qmolQuenched    = qmol_lay[np.argmin(np.abs(self.p-params['pQuench'])),:] # composition of well mixed pressures
            qmol_lay[ind_pQuenched,:] = qmolQuenched

            # Code will run into nans if there are any of H-, HDO or FeH (as they are artificially added), need to set them to some value here to avoid nans
            # if np.any(self.MolNames=='H-'):
            #     iMol=np.where(self.MolNames=='H-')[0];   qmol_lay[:,iMol]= 0.0
            if np.any(self.MolNames=='HDO'):
                iMol=np.where(self.MolNames=='HDO')[0];   qmol_lay[:,iMol]= 0.0
            if np.any(self.MolNames=='FeH'): 
                iMol=np.where(self.MolNames=='FeH')[0];   qmol_lay[:,iMol]= 0.0
            if np.any(self.MolNames=='O3'): 
                iMol=np.where(self.MolNames=='O3')[0];   qmol_lay[:,iMol]= 0.0
            
            totals = np.sum(qmol_lay,1)
            if np.any(totals < 0.99) or np.any(totals > 1.01):
                print('\n !!! WARNING : Sum of mixing ratio < 0.99 !!!!!!!!!!!!')
                print('Sum in each layer :')
                print(totals)
                print('\nSome significantly abundant VULCAN molecule is not considered by Scarlet')
            qmol_lay = qmol_lay/totals[:, np.newaxis]
            
            
        elif modelSetting['ComposType']=='WellMixed':
            #copy params['qmol'] values into each layer of qmol_lay           
            qmol_lay=np.zeros([self.nLay,self.nMol], dtype=self.numerical_precision)
            for i,MolName in enumerate(self.MolNames):
                if MolName in params['qmol']:
                    qmol_lay[:,i]=params['qmol'][MolName]
            #check sum of mixing ratios to be 1
            total = np.sum(qmol_lay[3,:])            
            if total<0.999 or total>1.001:
                print('Sum of Mixing Ratios Error:  sum='+str(total))

        # Following the power law parametrization of Parmentier et al. 2018
        elif modelSetting['ComposType'] == 'WellMixed_dissociation':
            # copy params['qmol'] values into each layer of qmol_lay
            qmol_lay = np.zeros([self.nLay, self.nMol], dtype=self.numerical_precision)
            H2_index = np.where(self.MolNames == 'H2')[0][0]

            # thermal dissociation functions
            # Eq 2, Parmentier et al. 2018
            def Ad(alpha, beta, gamma, P, T):
                logAd = (alpha * np.log10(P)) + (beta / T) - gamma
                return 10 ** logAd

            # Eq 1, Parmentier et al. 2018
            def A(A0, Ad):
                A = ((1 / np.sqrt(A0)) + (1 / np.sqrt(Ad))) ** (-2.)
                return A

            def diss_profile(P, T, A_0, alpha, beta, gamma, A_0_ref):
                '''
                A_0: VMR without dissociation
                alpha, beta, gamma, A_0_ref free parameters
                '''
                # Dissociated Abundance
                log_A_shift = np.log10(A_0 / A_0_ref)
                A_d = 10 ** (log_A_shift - gamma) * P ** alpha * 10 ** (beta / T)

                # Combine dissociated abundance with original abundance
                return ((1 / A_0) ** 0.5 + (1 / A_d) ** 0.5) ** (-2)

            # Equation to set the dissociated abundance profile to qmol_lay and add the difference as H2
            def update_dissociation_abundance_profile(ind, A0, alpha, beta, gamma, A_0_ref):
                # dissociation abundance profile
                # profile = A(A0,Ad(alpha,beta,gamma,self.p*1e-5,T))
                profile = diss_profile(P=self.p * 1e-5, T=T, A_0=A0, alpha=alpha, beta=beta, gamma=gamma,
                                       A_0_ref=A_0_ref)
                
                pdb.set_trace()
                
                # set abundance profile
                qmol_lay[:, ind] = profile
                # add what is removed as H2
                qmol_lay[:, H2_index] += (A0 - profile)

            for i, MolName in enumerate(self.MolNames):
                if MolName in params['qmol']:
                    # if H2O, VO, TiO, H-, Na, K, use dissociation profiles
                    if MolName == 'H2O':
                        # values from Table 1 of Parmentier et al. 2018
                        alpha_H2O, beta_H2O, gamma_H2O, A_0_ref_H2O = 2.0, 4.83 * 1e4, 15.9, 10 ** -3.3
                        # # abundance in deep atmosphere unaffected by dissociation
                        # A0_H2O = params['qmol'][MolName]
                        # # dissociation abundance profile
                        # H2O = A(A0_H2O,Ad(alpha_H2O,beta_H2O,gamma_H2O,self.p*1e-5,T))
                        # # set abundance profile
                        # qmol_lay[:,i]=H2O
                        # # add what is removed as H2
                        # qmol_lay[:,H2_index]+=H2O
                        # qmol_lay[:,H2_index]-=A0_H2O
                        update_dissociation_abundance_profile(ind=i, A0=params['qmol'][MolName], alpha=alpha_H2O,
                                                              beta=beta_H2O, gamma=gamma_H2O, A_0_ref=A_0_ref_H2O)
                    elif MolName == 'TiO':
                        # values from Table 1 of Parmentier et al. 2018
                        alpha_TiO, beta_TiO, gamma_TiO, A_0_ref_TiO = 1.6, 5.94 * 1e4, 23.0, 10 ** -7.1
                        update_dissociation_abundance_profile(ind=i, A0=params['qmol'][MolName], alpha=alpha_TiO,
                                                              beta=beta_TiO, gamma=gamma_TiO, A_0_ref=A_0_ref_TiO)
                    elif MolName == 'VO':
                        # values from Table 1 of Parmentier et al. 2018
                        alpha_VO, beta_VO, gamma_VO, A_0_ref_VO = 1.5, 5.40 * 1e4, 23.8, 10 ** -9.2
                        update_dissociation_abundance_profile(ind=i, A0=params['qmol'][MolName], alpha=alpha_VO,
                                                              beta=beta_VO, gamma=gamma_VO, A_0_ref=A_0_ref_VO)
                    # elif MolName == 'H-':
                    #     alpha_Hminus, beta_Hminus, gamma_Hminus, A_0_ref_Hminus = 0.6, -0.14*1e4, 7.7, 10**-8.3
                    #     update_dissociation_abundance_profile(ind=i, A0=params['qmol'][MolName], alpha=alpha_Hminus, beta=beta_Hminus, gamma=gamma_Hminus, A_0_ref=A_0_ref_Hminus)
                    elif MolName == 'Na':
                        alpha_Na, beta_Na, gamma_Na, A_0_ref_Na = 0.6, 1.89 * 1e4, 12.2, 10 ** -5.5
                        update_dissociation_abundance_profile(ind=i, A0=params['qmol'][MolName], alpha=alpha_Na,
                                                              beta=beta_Na, gamma=gamma_Na, A_0_ref=A_0_ref_Na)
                    elif MolName == 'K':
                        alpha_K, beta_K, gamma_K, A_0_ref_K = 0.6, 1.28 * 1e4, 12.7, 10 ** -7.1
                        update_dissociation_abundance_profile(ind=i, A0=params['qmol'][MolName], alpha=alpha_K,
                                                              beta=beta_K, gamma=gamma_K, A_0_ref=A_0_ref_K)
                    else:
                        qmol_lay[:, i] = params['qmol'][MolName]

            # check sum of mixing ratios to be 1
            total = np.sum(qmol_lay[3, :])
            if total < 0.999 or total > 1.001:
                print('Sum of Mixing Ratios Error:  sum=' + str(total))

        elif modelSetting['ComposType']=='SetProfiles':
            qmol_lay=np.zeros([self.nLay,self.nMol], dtype=self.numerical_precision)
            for i,MolName in enumerate(self.MolNames):
                if MolName in params['qmol']:
                    qmol_lay[:,i]=np.exp(ut.interp1dEx(np.log(params['qmol']['p']),np.log(params['qmol'][MolName]),np.log(self.p)))
           
        else:
            print('Unknown modelSetting[ComposType]: '+modelSetting['ComposType']        )

        #        #hardwired change abundances
        #        print('WARNING: Hardwired change of molecular abundances !!!!!!!!!!!!')
        #        iMol=np.where(self.MolNames=='Na')[0];  qmol_lay[:,iMol]=qmol_lay[:,iMol]/1e6
        #        iMol=np.where(self.MolNames=='K')[0];   qmol_lay[:,iMol]=qmol_lay[:,iMol]/1e3


        #Modifying Abundundances
        try: # try statement to ensure compatibility with files that don't have 'hardwAbun' in params
            for MolName in self.params['hardwAbun'].keys():
                iMol=np.where(self.MolNames==MolName)[0]
                if self.params['hardwAbun'][MolName][1]=='fix': # fix abundance to a constant volume mixing ratio at all layers
                    print('Setting {} abundance to {} at all layers'.format(MolName, self.params['hardwAbun'][MolName][0]) )
                    qmol_lay[:,iMol]= self.params['hardwAbun'][MolName][0]
                elif self.params['hardwAbun'][MolName][1]=='factor': # multiply the calculated abundance by a factor
                    print('Multiplying {} abundance by a factor of {}'.format(MolName, self.params['hardwAbun'][MolName][0]) )
                    qmol_lay[:,iMol]=qmol_lay[:,iMol] * self.params['hardwAbun'][MolName][0]
                elif self.params['hardwAbun'][MolName][1]=='relative': # set abundance as a factor relative to another molecule (i.e. set HDO abundance as 1% of water)
                    try:
                        self.params['hardwAbun'][MolName][2]
                    except:
                        raise Exception("Error: user must specify which molecule abundance {} is set relative to (e.g. '{}':[1e-2,'relative','H2O'])".format(MolName,MolName) )
                    print('Setting {} abundance as {} of {}'.format(MolName, self.params['hardwAbun'][MolName][0], self.params['hardwAbun'][MolName][2]) )
                    relative_Mol_ind = np.where(self.MolNames==self.params['hardwAbun'][MolName][2])[0]
                    # set the abundance of the secondary molecule (e.g. HDO = 1% of H2O)
                    qmol_lay[:,iMol]=qmol_lay[:,relative_Mol_ind] * self.params['hardwAbun'][MolName][0]
                    # also re-scale the abundance of the primary molecule (e.g. lower abundance of H2O by 1% to account for the 1% gone into HDO)
                    qmol_lay[:,relative_Mol_ind] -=qmol_lay[:,relative_Mol_ind] * self.params['hardwAbun'][MolName][0]
        except:
            pass
        
        # adjusting for hardwired changes
        totals = np.sum(qmol_lay,1) 
        if np.any(totals<0.95) or np.any(totals>1.01):
            print('!!! WARNING: Sum of Mixing Ratios <0.95 after abundance changes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Sum in in each layer:')
            print(totals)
            try:
                print('Metallicity={:g}, C/O={:g}, pQuench=q{:g}'.format(params['Metallicity'],params['CtoO'],params['pQuench']))
            except:
                pass
                    
        qmol_lay = qmol_lay / totals[:,np.newaxis]
            
        return qmol_lay
    

    def InterpFromChemEquiGridLUT(self,p,T,params):
        '''
        Computes mixing ratios from LUT
        --------------------------
        InterpFromChemEquiGridLUT(self,p,T,params)        
        Output is qmol_lay in shape: (self.nLay,self.nMol)
        '''
        qmol_lay = 1e-20*np.ones((len(p),self.nMol), dtype=self.numerical_precision)
        
        for i in range(len(p)):
            qmol_lay[i,:]=10**(np.squeeze((ut.interp4bb(self.MetallicityList,self.CtoORatioList,self.PTable,self.TTable,self.AbunLUT,params['Metallicity'],params['CtoO'],np.log10(p[i]),T[i]))))        
                
        return qmol_lay



    def executeOption(self):
        '''Allows user to force some adhoc options. A warning will be given'''

        if 'option' in self.params.keys():
            if self.params['option']=='CH4toCO':
                print('Adhoc option: Applying CH4toCO !!!!')
                indCH4=np.where(self.MolNames=='CH4')[0][0]
                indCO =np.where(self.MolNames=='CO' )[0][0]
                self.qmol_lay[:,indCO]  = self.qmol_lay[:,indCO] + self.qmol_lay[:,indCH4] - 1e-15
                self.qmol_lay[:,indCH4] = 1e-15
                #print(np.sum(self.qmol_lay,1))


    #%% Opacities

    def calcOpacities(self,modelSetting,params,T,iMol=None,saveOpac=False,low_res_mode=False):

        #extinctCoef = np.zeros(self.nLay,self.nWave

        '''
        return extinctCoef, absorbCoef, scatCoef in 1/m    [dimensions: nLay,nWave]
        '''
        #self.T = np.interp(np.log(self.p),np.log(params['Tprof'][0]),params['Tprof'][1])
        #--Gas opacities------------------------------------------------------------------------
        #Molecular line absorption
        if self.opacSources['molLineAbsorb']:
            if iMol is None:
                if low_res_mode:
                    sigma = self.interpCrossSec_lowres(T)
                else:
                    sigma = self.interpCrossSec(T)
                nabsmol = self.nmol[:,self.AbsMolInd]
                molLineAbsorb = np.sum(sigma*nabsmol[:,:,np.newaxis],axis=1)
            else:
                if low_res_mode:
                    sigma = self.interpCrossSec_lowres(T,iMol=iMol)
                else:
                    sigma = self.interpCrossSec(T,iMol=iMol)
                nabsmol = self.nmol[:,self.AbsMolInd]
                molLineAbsorb = sigma[:,iMol]*nabsmol[:,iMol,np.newaxis]
        else:
            molLineAbsorb = 0.0

        #Collision Induced Absorption (CIA)
        if self.opacSources['cia']:
            if low_res_mode:
                cia = self.calcCIA_lowres(T)
            else:
                cia = self.calcCIA(T)
        else:
            cia = 0.0
        
        #Rayleigh scattering by gas molecules
        if self.opacSources['rayleighScat']:
            if low_res_mode:
                rayleighScat = self.calcRayleighScat_lowres()
            else:
                rayleighScat = self.calcRayleighScat()
        else:
            rayleighScat=0.0


        #--Cloud opacities---------------------------------------------------------------------

        #Ad-hoc small-particle hazes (increased Rayleigh scattering due to small particles)
        if (self.opacSources['fineHazeScat']) and (self.modelSetting['CloudTypes'][1]==1):
            fineHazeScat = rayleighScat * params['cHaze']
        else:
            fineHazeScat=0.0

        #Parameterized Mie Clouds
        if (self.opacSources['paramMieCloud']) and (modelSetting['CloudTypes'][2]==1):
            paramMieCloudAbsorp, paramMieCloudScat = self.calcParamCloudOpac(modelSetting,params)
        else:
            paramMieCloudAbsorp=0.0
            paramMieCloudScat=0.0

        #Mie scattering from CARMA particle densities
        if (self.opacSources['carmaMieCloud']) and (modelSetting['CloudTypes'][3]==1):
            #Add clouds opacitity from carma
            carmaMieCloudAbsorp, carmaMieCloudScat = self.calcMieOpacFromNPart(self.npartConds)
        else:
            carmaMieCloudAbsorp=0.0
            carmaMieCloudScat=0.0
        
        #--Summing up contributions-----------------------------------------------------------
        absorbCoef  = molLineAbsorb + cia + paramMieCloudAbsorp + carmaMieCloudAbsorp
        scatCoef    = rayleighScat + fineHazeScat + paramMieCloudScat + carmaMieCloudScat
        extinctCoef =  absorbCoef + scatCoef

        #--Saving-----------------------------------------------------------------------------
        if saveOpac:
            
            self.opac=dict()

            self.opac['extinctCoef']=extinctCoef

            self.opac['absorbCoef']=absorbCoef
            self.opac['molLineAbsorb']=molLineAbsorb
            self.opac['cia']=cia
            self.opac['paramMieCloudAbsorp']=paramMieCloudAbsorp
            self.opac['carmaMieCloudAbsorp']=carmaMieCloudAbsorp
            
            self.opac['scatCoef']=scatCoef
            self.opac['rayleighScat']=rayleighScat
            self.opac['fineHazeScat']=fineHazeScat
            self.opac['paramMieCloudScat']=paramMieCloudScat
            self.opac['carmaMieCloudScat']=carmaMieCloudScat
            
        return extinctCoef, absorbCoef, scatCoef
    


    def interpCrossSec(self,T,iMol=None):
        'T is vector of temperature'
        'pressure is set by self.p'
        'return sigma with shape [nLay,nWave,nAbsMol]'
        

        self.sigmaAtm[:]=0

        #default (do all molecules at once so they can be summed)
        if iMol is None: 
            for iLay,TLay in enumerate(T):
                if TLay<=self.LookUpTGrid[0]:
                    self.sigmaAtm[iLay,:,:] = self.LookUpSigma[0,iLay,:,:]
        
                elif TLay>=self.LookUpTGrid[-1]:
                    self.sigmaAtm[iLay,:,:] = self.LookUpSigma[-1,iLay,:,:]
                else:
                    ind = bisect(self.LookUpTGrid,TLay) - 1
                    w = (TLay-self.LookUpTGrid[ind]) / (self.LookUpTGrid[ind+1]-self.LookUpTGrid[ind])
                    self.sigmaAtm[iLay,:,:] = (1-w) * self.LookUpSigma[ind,iLay,:,:] + w * self.LookUpSigma[ind+1,iLay,:,:]
        
        #only calculate for specific molecule if requested (saves computations; ~13x faster for 13 absorbing molecules?)
        else:  
            for iLay,TLay in enumerate(T):
                if TLay<=self.LookUpTGrid[0]:
                    self.sigmaAtm[iLay,iMol,:] = self.LookUpSigma[0,iLay,iMol,:]
        
                elif TLay>=self.LookUpTGrid[-1]:
                    self.sigmaAtm[iLay,iMol,:] = self.LookUpSigma[-1,iLay,iMol,:]
                else:
                    ind = bisect(self.LookUpTGrid,TLay) - 1
                    w = (TLay-self.LookUpTGrid[ind]) / (self.LookUpTGrid[ind+1]-self.LookUpTGrid[ind])
                    self.sigmaAtm[iLay,iMol,:] = (1-w) * self.LookUpSigma[ind,iLay,iMol,:] + w * self.LookUpSigma[ind+1,iLay,iMol,:]
        
        return self.sigmaAtm 



    def interpCrossSec_lowres(self,T,iMol=None):
        'T is vector of temperature'
        'pressure is set by self.p'
        'return sigma with shape [nLay,nWave,nAbsMol]'

        self.sigmaAtm_lowres[:]=0

        #default (do all molecules at once so they can be summed)
        if iMol is None:
            for iLay,TLay in enumerate(T):
                if TLay<self.LookUpTGrid[0]:
                    self.sigmaAtm_lowres[iLay,:,:] = self.LookUpSigma_lowres[0,iLay,:,:]

                elif TLay>self.LookUpTGrid[-1]:
                    self.sigmaAtm_lowres[iLay,:,:] = self.LookUpSigma_lowres[-1,iLay,:,:]
                else:
                    ind = bisect(self.LookUpTGrid,TLay) - 1
                    # t0 = time()
                    w = (TLay-self.LookUpTGrid[ind]) / (self.LookUpTGrid[ind+1]-self.LookUpTGrid[ind])
                    self.sigmaAtm_lowres[iLay,:,:] = (1-w) * self.LookUpSigma_lowres[ind,iLay,:,:] + w * self.LookUpSigma_lowres[ind+1,iLay,:,:]
                    # print('normal took {} seconds'.format(str( np.round((time()-t0),3))))
                    # numba not actually faster in this case
                    # t0 = time()
                    # self.sigmaAtm[iLay,:,:] = calcsigmaAtm_numba(self.LookUpTGrid, TLay, self.LookUpSigma, iLay, ind)
                    # print('numba took {} seconds'.format(str( np.round((time()-t0),3))))

        #only calculate for specific molecule if requested (saves computations; ~13x faster for 13 absorbing molecules?)
        else:
            for iLay,TLay in enumerate(T):
                if TLay<self.LookUpTGrid[0]:
                    self.sigmaAtm_lowres[iLay,iMol,:] = self.LookUpSigma_lowres[0,iLay,iMol,:]

                elif TLay>self.LookUpTGrid[-1]:
                    self.sigmaAtm_lowres[iLay,iMol,:] = self.LookUpSigma_lowres[-1,iLay,iMol,:]
                else:
                    ind = bisect(self.LookUpTGrid,TLay) - 1
                    w = (TLay-self.LookUpTGrid[ind]) / (self.LookUpTGrid[ind+1]-self.LookUpTGrid[ind])
                    self.sigmaAtm_lowres[iLay,iMol,:] = (1-w) * self.LookUpSigma_lowres[ind,iLay,iMol,:] + w * self.LookUpSigma_lowres[ind+1,iLay,iMol,:]

        return self.sigmaAtm_lowres

    
    def calcCIA(self,T):

        nmol=self.nmol * 10**-6  #Convert to cgs
        amg=2.68676*10**19 #cm^(-3) ; amagat: number of ideal gas molecules per unit volume at STP
        
        #want to sum over axis 2 (i.e. 1 in Python) to get total number density per layer
        #for now, assume the following:
        
        kappa=np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        #kappa_NH2=np.zeros([self.nLay,self.nWave]) #is this for if we want to exclude He?
    
        for i in range(self.nLay):
            
            #CO2-CO2, N2-N2, O2-O2
            for cia_dict in self.ciaList_Hitran:
                if cia_dict['MOL1'] in self.MolNames or cia_dict['MOL2'] in self.MolNames:
                    #Units of nmol are already in cm-3 when reading data from self.ciaList_Hitran
                    nMol1 = np.sum(nmol, 1) * self.getMixRatio(cia_dict['MOL1'])
                    nMol2 = np.sum(nmol, 1) * self.getMixRatio(cia_dict['MOL2'])
    
                    #Interp1bb does not work when there is only one temperature. No interpolation needed when there's only 1 temperature
                    if len(cia_dict['TempList']) == 1:
                        L0Sq = (cia_dict['interpolatedCoeff'])
    
                    #Interpolation over temperatures
                    else:
                        L0Sq = ut.interp1bb(cia_dict['TempList'], (cia_dict['interpolatedCoeff']), T[i])
    
                    kappa[i, :] = kappa[i, :] + L0Sq * nMol1[i] * nMol2[i]

            #H2H2 and H2He
            if (T[i]<=1000):
                if 'H2' in self.MolNames:                   
                    nH2_Amagat=np.sum(nmol,1)*self.getMixRatio('H2')/amg
    
                    #H2H2
                    #self.ciaT contains temperatures from CIA LUT
                    #self.ciaH2LUT contains the L0Sq values at each of these temperatures (already interpolated over self.nWave)
                    L0Sq=ut.interp1bb(self.ciaT,self.ciaH2LUT,T[i])
                    kappa[i,:]=kappa[i,:]+L0Sq * nH2_Amagat[i]**2
                
                #this implies that should use HT values for T=1000 K? (i.e. remove 1st col of VHT LUT?)  
            elif (T[i]>1000):   
                if 'H2' in self.MolNames:                
                    nH2_Amagat=np.sum(nmol,1)*self.getMixRatio('H2')/amg
                    L0Sq=ut.interp1bb(self.ciaT,self.ciaH2LUT,T[i])
                    kappa[i,:]=kappa[i,:]+L0Sq * nH2_Amagat[i]**2
            
                if 'H2' in self.MolNames and 'He' in self.MolNames:                
                    #H2He (only relevant for T>1000K so only take last 7 values of self.ciaT !!!)
                    nHe_Amagat=np.sum(nmol,1)*self.getMixRatio('He')/amg
                    L0Sq=ut.interp1bb(self.ciaT[14:],self.ciaHeLUT,T[i])
                    kappa[i,:]=kappa[i,:]+L0Sq*nH2_Amagat[i]*nHe_Amagat[i]
        
        kappa=kappa*100 #1/cm to 1/m
                
        return kappa
    

    def calcCIA_lowres(self,T):

        nmol=self.nmol * 10**-6  #Convert to cgs
        amg=2.68676*10**19 #cm^(-3) ; amagat: number of ideal gas molecules per unit volume at STP

        #want to sum over axis 2 (i.e. 1 in Python) to get total number density per layer
        #for now, assume the following:
        nH2_Amagat=np.sum(nmol,1)*0.85/amg
        nHe_Amagat=np.sum(nmol,1)*0.15/amg

        kappa=np.zeros([self.nLay,self.nWave_lowres], dtype=self.numerical_precision)
        #kappa_NH2=np.zeros([self.nLay,self.nWave]) #is this for if we want to exclude He?

        for i in range(self.nLay):
            if (T[i]<=1000):

                #H2H2
                #self.ciaT contains temperatures from CIA LUT
                #self.ciaH2LUT contains the L0Sq values at each of these temperatures (already interpolated over self.nWave)

                L0Sq=ut.interp1bb(self.ciaT,self.ciaH2LUT_lowres,T[i])
                kappa[i,:]=L0Sq * nH2_Amagat[i]**2

            #this implies that should use HT values for T=1000 K? (i.e. remove 1st col of VHT LUT?)
            elif (T[i]>1000):

                L0Sq=ut.interp1bb(self.ciaT,self.ciaH2LUT_lowres,T[i])
                kappa[i,:]=L0Sq * nH2_Amagat[i]**2

                #H2He (only relevant for T>1000K so only take last 7 values of self.ciaT !!!)
                L0Sq=ut.interp1bb(self.ciaT[14:],self.ciaHeLUT_lowres,T[i])
                # kappa[i,:]=kappa[i,:]+L0Sq*nH2_Amagat[i]*nHe_Amagat[i]
                kappa[i,:]+=L0Sq*nH2_Amagat[i]*nHe_Amagat[i]

        # kappa=kappa*100 #1/cm to 1/m
        kappa*=100 #1/cm to 1/m

        return kappa


    
    def calcRayleighScat(self):
        
        #maximum index in wave vector for which Rayleigh scattering is considered
        #self.wave is in microns (?)
        #note: "inverted" relative to RayleighScat.m which uses nu
        imaxRayleigh=self.nWave
        
        #Calculate cross sections at pref, Tref (i.e. close to STP)
        pref=101325 #Pa
        Tref=288 #K
        nref=pref/(kBoltz*Tref)  #m
        nref=nref*10**-6  #cm^(-3) (cgs)
        
        nu=10000/self.wave #1/cm
        nmol=self.nmol*10**-6 #cm^(-3) NEED cgs!
        
        #initialize scattering coefficient in each layer and at each wavelength
        ScatCoef=np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        
        for iMol,MolName in enumerate(self.MolNames):
            
            if (self.RefracIndex[iMol]>1):
                
                #Rayleigh scattering cross section
                sigma_Ray_Mol=(24*pi**3*nu[:imaxRayleigh]**4)/(nref**2)*((self.RefracIndex[iMol]**2-1)/(self.RefracIndex[iMol]**2+2))**2*self.KingsCorr[iMol] # cm^2 (cross section as function of wavelength)
                
                #Scattering coefficient = (scattering coefficient array so far) + (number density for each layer and wavelength) * (scattering coefficient per layer and wavelength)
                #Note: for some reason, needed to transpose number density matrix??? (to get same dimensions for elementwise multiplication)

                #ScatCoef[:,:imaxRayleigh] = ScatCoef[:,:imaxRayleigh] + np.tile(nmol[:,iMol][:,np.newaxis],[1,imaxRayleigh]) * np.tile(sigma_Ray_Mol,[self.nLay,1]) # cm^(-3) * cm^2 = 1/cm
                ScatCoef[:,:imaxRayleigh] = ScatCoef[:,:imaxRayleigh] + np.outer(nmol[:,iMol],sigma_Ray_Mol)
        
        ScatCoef=ScatCoef*100  #1/cm to 1/m
        return ScatCoef




    def calcRayleighScat_lowres(self):

        #maximum index in wave vector for which Rayleigh scattering is considered
        #self.wave is in microns (?)
        #note: "inverted" relative to RayleighScat.m which uses nu
        imaxRayleigh=self.nWave_lowres

        #Calculate cross sections at pref, Tref (i.e. close to STP)
        pref=101325 #Pa
        Tref=288 #K
        nref=pref/(kBoltz*Tref)  #m
        # nref=nref*10**-6  #cm^(-3) (cgs)
        nref*=10**-6  #cm^(-3) (cgs)

        nu=10000/self.wave_lowres #1/cm
        nmol=self.nmol*10**-6 #cm^(-3) NEED cgs!

        #initialize scattering coefficient in each layer and at each wavelength
        ScatCoef=np.zeros([self.nLay,self.nWave_lowres], dtype=self.numerical_precision)

        for iMol,MolName in enumerate(self.MolNames):

            if (self.RefracIndex[iMol]>1):
                # t0 = time()
                #Rayleigh scattering cross section
                sigma_Ray_Mol=(24*pi**3*nu[:imaxRayleigh]**4)/(nref**2)*((self.RefracIndex[iMol]**2-1)/(self.RefracIndex[iMol]**2+2))**2*self.KingsCorr[iMol] # cm^2 (cross section as function of wavelength)

                #Scattering coefficient = (scattering coefficient array so far) + (number density for each layer and wavelength) * (scattering coefficient per layer and wavelength)
                #Note: for some reason, needed to transpose number density matrix??? (to get same dimensions for elementwise multiplication)

                #ScatCoef[:,:imaxRayleigh] = ScatCoef[:,:imaxRayleigh] + np.tile(nmol[:,iMol][:,np.newaxis],[1,imaxRayleigh]) * np.tile(sigma_Ray_Mol,[self.nLay,1]) # cm^(-3) * cm^2 = 1/cm
                # ScatCoef[:,:imaxRayleigh] = ScatCoef[:,:imaxRayleigh] + np.outer(nmol[:,iMol],sigma_Ray_Mol)
                ScatCoef[:,:imaxRayleigh] += np.outer(nmol[:,iMol],sigma_Ray_Mol)


                # print('normal took {} seconds'.format(str( np.round((time()-t0),3))))
                # # numba not actually faster in this case
                # t0 = time()
                # ScatCoef[:,:imaxRayleigh] +=ScatCoef_numba(imaxRayleigh, self.RefracIndex, self.KingsCorr, iMol, nref, nu, nmol)
                # print('numba took {} seconds'.format(str( np.round((time()-t0),3))))
        ScatCoef=ScatCoef*100  #1/cm to 1/m
        return ScatCoef



    def calcParamCloudOpac(self,modelSetting,params,waveRef=1.5):
        
#        mieRpart   = params['mieRpart']*1e-6
#        p1 = params['miePAtTau1']
#        p0 = params['miePAtTau1'] / (1+params['mieDiffFacPRange'])
#        cond=self.conds[0]
#
#        #--Determine number density of particles as a function of z (npart in 1/m^3) to match the desired cloud opacity at wave=waveRef
#        iwave=bisect(self.wave,waveRef)
#        Qabs_Gpart = np.interp(mieRpart,self.mieLUT[cond]['reff'],self.mieLUT[cond]['Qabs_Gpart'][iwave,:])    #m^2 per particle
#        Qsca_Gpart = np.interp(mieRpart,self.mieLUT[cond]['reff'],self.mieLUT[cond]['Qsca_Gpart'][iwave,:])    #m^2 per particle
#        Qext_Gpart = Qabs_Gpart + Qsca_Gpart
#
#        columnDensToTau1 = 1.0 / Qext_Gpart  #[particles/m^2]
#        z0 = np.interp(p0,self.p,self.z) #[m]
#        z1 = np.interp(p1,self.p,self.z) #[m]    
#        
#        deltaZ = z0-z1 #[m]
#        #slope = 2*columnDensToTau1/deltaZ**2   #Slope of particle density rise [1/m^3 per meter = 1/m^4]
#
#        RpBase = self.Rp - self.zgrid[self.iLevRpRef]
#        r1 = RpBase+z1
#        dl = np.sqrt(2*r1*deltaZ)
#        slope = 2*columnDensToTau1/dl**2   #Slope of particle density rise [1/m^3 per meter = 1/m^4]
#        #particle number density in [1/m^3] is  npart = slope * (z0-z) --> we assume that the particle density rising linear below z0 
#        A = grazingColumnDensToTau1 / (H*np.exp(-z1/H))
#        npart = np.zeros([self.nLay])
#        npart[self.z<z0] = slope * (z0 - self.z[self.z<z0])

        
        mieRpart   = params['mieRpart']*1e-6
        p1         = params['miePAtTau1']
        indLay = bisect(self.p,p1)
        H = params['mieRelScaleHeight']*self.scaleHeight[indLay]   # [m]
        cond=self.conds[0]

        #--Determine number density of particles as a function of z (npart in 1/m^3) to match the desired cloud opacity at wave=waveRef
        iwave=bisect(self.wave,waveRef)
        Qabs_Gpart = np.interp(mieRpart,self.mieLUT[cond]['reff'],self.mieLUT[cond]['Qabs_Gpart'][iwave,:])    #m^2 per particle
        Qsca_Gpart = np.interp(mieRpart,self.mieLUT[cond]['reff'],self.mieLUT[cond]['Qsca_Gpart'][iwave,:])    #m^2 per particle
        Qext_Gpart = Qabs_Gpart + Qsca_Gpart

        vertColumnDensToTau1 = 1.0/Qext_Gpart  #[particles/m^2]                

        z1 = np.interp(p1,self.p,self.z)                # [m]    
        r1 = self.RpBase+z1           # [m]
        dl = np.sqrt(2*r1*H)     # [m]
        grazingColumnDensToTau1 = vertColumnDensToTau1 * H/dl
        
        #particle number density in [1/m^3] is  npart = slope * (z0-z) --> we assume that the particle density rising linear below z0 
        npart = grazingColumnDensToTau1 / H * np.exp(-(self.z-z1)/H)
        npart[npart>1e20]=1e20

        #totalPart = np.trapz(npart[:indLay],x=-self.z[:indLay])
        #--Now calculate opacities at all wavelengths based on npart
        #        mieScat = np.zeros([self.nLay,self.nWave])
        #        mieAbsorp = np.zeros([self.nLay,self.nWave])

        Qabs_Gpart_ForRpart = ut.interp1bb(self.mieLUT[cond]['reff'],self.mieLUT[cond]['Qabs_Gpart'].transpose(),mieRpart)
        Qsca_Gpart_ForRpart = ut.interp1bb(self.mieLUT[cond]['reff'],self.mieLUT[cond]['Qsca_Gpart'].transpose(),mieRpart)
        
        #mieScat   = np.outer(npart,Qabs_Gpart_ForRpart)   #[self.nLay,self.nWave]
        #mieAbsorp = np.outer(npart,Qsca_Gpart_ForRpart)   #[self.nLay,self.nWave]
        mieScat   = np.outer(npart,Qsca_Gpart_ForRpart)   #[self.nLay,self.nWave]
        mieAbsorp = np.outer(npart,Qabs_Gpart_ForRpart)   #[self.nLay,self.nWave]

        if 0:
            mieExtinct = mieScat + mieAbsorp
            #Validation plot
            #taugridCumSum = np.cumsum(mieExtinct[:,iwave]*self.dz)
            taugrid = np.zeros([self.nLay+1], dtype=self.numerical_precision)
            for i in range(self.nLay):
                taugrid[i+1]=np.trapz(mieExtinct[:i+1,iwave],x=-self.z[:i+1])
            fig,ax=plt.subplots()
            ax.plot(taugrid,self.p/100,'red')
            #ax.plot(taugridCumSum,self.pgrid/100)
            ax.axhline(p1/100,color='k',ls='--')
            for pres in self.p:
                ax.axhline(pres/100,ls='--',lw=0.1,color='black')
            ax.set_xlabel('Optical depth due to Mie Clouds [1]')
            ax.set_ylabel('Pressure [mbar]')
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.set_xlim([-0.3,3])
            fig.savefig(self.filebase+'MieCloudsAtWaveRef.pdf')   
#            mieExtinct = mieScat + mieAbsorp
#            #Validation plot
#            taugridCumSum = np.cumsum(mieExtinct[:,iwave]*self.dz)
#            taugrid = np.zeros([self.nLay])
#            for i in range(self.nLay):
#                taugrid[i]=np.trapz(mieExtinct[:i+1,iwave],x=-self.z[:i+1])
#            fig,ax=plt.subplots()
#            ax.plot(taugrid,self.z/1000)
#            ax.plot(taugridCumSum,self.z/1000)
#            ax.axhline(z1/1000)
#            for zz in self.zgrid:
#                ax.axhline(zz/1000,ls='--',lw=0.1,color='black')
#            ax.set_xlabel('Optical depth due to Mie Clouds [1]')
#            ax.set_ylabel('z [km]')
#            ax.set_xlim([-0.3,3])
#            fig.savefig(self.filebase+'MieCloudsAtWaveRef.pdf')        
        
        return mieScat, mieAbsorp


    def calcMieOpacFromNPart(self,npartConds):
        
        #initialize
        mieScat = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        mieAbsorp = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        
        #calculate Mie scat and absorp coefficients for each condensate and sum them
        for iCond in range(len(self.conds)):
            
            cond=self.conds[iCond]

            Qabs_Gpart=self.mieLUT[cond]['Qabs_Gpart']
            Qsca_Gpart=self.mieLUT[cond]['Qsca_Gpart']
            npart=self.npartConds[cond]             #see readCarmaFile
                    
            #loop over each layer each time (i.e. for each cond)
            for iLay in range(self.nLay):
                mieScat[iLay,:]=mieScat[iLay,:]+np.sum((Qsca_Gpart*np.tile(npart[iLay,:][np.newaxis,:],(self.nWave,1))),axis=1) #[1/m]; need to tile number densities (same for each wave!) AND sum over all reff!!!
                mieAbsorp[iLay,:]=mieAbsorp[iLay,:]+np.sum((Qabs_Gpart*np.tile(npart[iLay,:][np.newaxis,:],(self.nWave,1))),axis=1) #[1/m]
        
        return mieAbsorp, mieScat
      

    
    #%% Calculate planetary spectra
    
    
#     def calcTransitSpectrum(self,extinctCoef,nb=None,saveOpac=False):
#         '''
#         Computes transmission spectrum
#         '''

#         if nb is None:
#             nb=self.nLay
#         #print 'nb =',nb
        
#         #Find zCloud of thick cloud deck
#         if self.modelSetting['CloudTypes'][0]:
#             zCloud=ut.interp1dEx(np.log(self.p),self.z,np.log(self.params['pCloud']))  # [m] ad-hoc flat cloud deck in TransmissionSpectrum
#         else:
#             zCloud = 0.0 #m

#         #Minimum impact parameters of grazing beams
#         if self.modelSetting['CloudTypes'][0]: #0 = flat (opaque) cloud (i.e. sharp cutoff)
#             bmin=self.RpBase+zCloud #m
#         else:
#             bmin=self.RpBase #m
#         r=self.RpBase+self.z #m  # Radius from center of the planet

#         #Impact parameter
#         bgrid=np.linspace(self.RpBase+self.z[1],bmin,nb+1) #m
#         b=0.5*(bgrid[0:-1]+bgrid[1:])   #m
#         db=-np.diff(bgrid)            #m
        
#                         #        pdb.set_trace()
#                         #        inp=dict()
#                         #        inp['extinctCoef']=extinctCoef
#                         #        inp['Rstar']=self.Rstar
#                         #        inp['bgrid']=bgrid
#                         #        inp['r']=r
#                         #        ut.savepickle(inp,'testTransmissionCalc.pkl')

#         tau=np.zeros([nb,self.nWave], dtype=self.numerical_precision) #no dim
#         for ib in range(0,nb):
            
#             #Integrate along line of sight with impact parameters b
#             l=np.zeros([self.nLay], dtype=self.numerical_precision) #m
#             j=0
#             while (r[j]>b[ib]):
#                 l[j]=np.sqrt(r[j]**2-b[ib]**2) #m
#                 j=j+1
#             dl=-np.diff(l[0:j+1]) #m

#             if 1:            
#                 #slanted path coordinate (from 0 at outer layer to sum(dl) at tangential point)
#                 s=np.r_[0.0,np.cumsum(dl)]
#                 #Extinction coeffcient at tangential point (interpolated)
#                     #extAtTangentialPoint = 10**ut.interp1bb(r[::-1],np.log10(extinctCoef[::-1,:]),b[ib])
#                 #interpolate the extinction coefficients at the impact parameter
#                 #extAtTangentialPoint = np.exp(ut.interp1bb(np.flip(r),np.flip(np.log(extinctCoef),axis=0),b[ib]))
                
#                 extAtTangentialPoint = extinctCoef[j,:]
                
#                 #Extiction coefficient along slanted path
#                 ext = np.vstack([extinctCoef[0:j,:],extAtTangentialPoint])
#             else:
#                 #Finer sampling near tangential point, but has little effects 
#                 dl=dl[::-1]
# #                #slanted path coordinate (from 0 at outer layer to sum(dl) at tangential point)
# #                s=np.cumsum(dl)  #np.r_[0.0,np.cumsum(dl)]
# #                #Extinction coefficient points near tangential point (interpolated)
# #                addedS=np.linspace(0,s[0],4)[:-1]
# #                addedR=np.sqrt(b[ib]**2+addedS**2)
# #                extNearTangentialPoint=np.zeros([len(addedR),extinctCoef.shape[1]])
# #                for i in range(len(addedR)):
# #                    extNearTangentialPoint[i,:] = 10**ut.interp1bb(r[::-1],np.log10(extinctCoef[::-1,:]),addedR[i])
# #                s=np.r_[addedS,s]
# #                ext = np.vstack([extNearTangentialPoint,extinctCoef[j-1::-1,:]])

#             #Integrate to get tau along slanted path

#             tau[ib,:]=ut.trapzMultiTimesTwo(x=s,y=ext)

#             #        print ib, (b[ib]**2)/(self.Rstar**2)*1e6,'ppm  tau=',tau[ib,2700]  #corresponds to transit depth
#             #        if ib==40:
#             #            pdb.set_trace()
    
#             #        if np.any(np.isnan(l)):
#             #            print('ERROR: nan values in l. Probably rgrid extends to negative values.')
#             #            pdb.set_trace()

#         #Transmission 
        
#         transmis=np.exp(-tau) #no dim
        
#         #print tau[:,5140]
#         #print Transmis[:,5140]
        
#         dEpsilon= (2*pi*b*db)[:,np.newaxis] *   (1-transmis) #m^2
        
#         #Area blocked by atmo
#         Epsilon=np.sum(dEpsilon,axis=0) #m^2
        
#         #Transit depth = Area blocked by atmo + (effective) planet radius
#         TransitDepth=(Epsilon+pi*bmin**2)/(pi*self.Rstar**2) #no dim
#         dppm=TransitDepth*1e6
        
#         if saveOpac:
#             self.opac['tauTransit']     =tau
#             self.opac['transmisTransit']=transmis
        
#         if np.any(np.isnan(dppm)):
#             print(self.params)
#             pdb.set_trace()
#         return dppm


    def calcTransitSpectrum(self,extinctCoef,nb=None,saveOpac=False):
        '''
        Computes transmission spectrum, now with matrices
        '''
        if nb is None:
            nb=self.nLay
        #Find zCloud of thick cloud deck
        if self.modelSetting['CloudTypes'][0]:
            zCloud=ut.interp1dEx(np.log(self.p),self.z,np.log(self.params['pCloud']))  # [m] ad-hoc flat cloud deck in TransmissionSpectrum
        else:
            zCloud = 0.0 #m
        #Minimum impact parameters of grazing beams
        if self.modelSetting['CloudTypes'][0]: #0 = flat (opaque) cloud (i.e. sharp cutoff)
            bmin=self.RpBase+zCloud #m
        else:
            bmin=self.RpBase #m
        r=self.RpBase+self.z #m  # Radius from center of the planet
        #Impact parameter
        # bgrid=np.linspace(self.RpBase+self.z[1],bmin,nb+1) #m
        # b=0.5*(bgrid[0:-1]+bgrid[1:])   #m
        b=np.linspace(self.RpBase+self.z[0],bmin,nb, dtype=self.numerical_precision) #m
        db=-np.diff(b)            #m
        #Building extinction coefficient matrix and Y matrix for fast trapz rule of the path
        Sigma=(extinctCoef[:-1]+extinctCoef[1:])
        Y=r[None,:]**2-b[:,None]**2
        sliceY=Y<0
        Y[sliceY]=Y[sliceY]*0.0
        Y=np.sqrt(Y)
        Y=-np.diff(Y,axis=1)
        #trapz rule of the path (tau)
        tau=np.matmul(Y,Sigma)
        #Transmis
        Absorption=-np.expm1(-tau)
        #trapz rule over altitude
        Epsilon=((Absorption)*b[:,None])
        Epsilon=Epsilon[:-1]+Epsilon[1:]
        Epsilon=np.pi*np.dot(db,Epsilon)
        TransitDepth=(Epsilon+pi*bmin**2)/(pi*self.Rstar**2) #no dim
        dppm=TransitDepth*1e6
        if saveOpac:
            self.opac['tauTransit']     =tau
            self.opac['absorptionTransit']=Absorption
        if np.any(np.isnan(dppm)):
            pdb.set_trace()
        return dppm
    
    
    
    def calcTransitSpectrumFiner(self,extinctCoef,nb=None,saveOpac=False):
        '''
        Computes transmission spectrum
        '''

        if nb is None:
            nb=self.nLay
        #print 'nb =',nb
        
        #Find zCloud of thick cloud deck
        if self.modelSetting['CloudTypes'][0]:
            zCloud=ut.interp1dEx(np.log(self.p),self.z,np.log(self.params['pCloud']))  # [m] ad-hoc flat cloud deck in TransmissionSpectrum
        else:
            zCloud = 0.0 #m

        #Minimum impact parameters of grazing beams
        if self.modelSetting['CloudTypes'][0]: #0 = flat (opaque) cloud (i.e. sharp cutoff)
            bmin=self.RpBase+zCloud #m
        else:
            bmin=self.RpBase #m
        r=self.RpBase+self.z #m  # Radius from center of the planet

        #Impact parameter
        bgrid=np.linspace(self.RpBase+self.z[1],bmin,nb+1) #m
        b=0.5*(bgrid[0:-1]+bgrid[1:])   #m
        db=-np.diff(bgrid)            #m
        
                        #        pdb.set_trace()
                        #        inp=dict()
                        #        inp['extinctCoef']=extinctCoef
                        #        inp['Rstar']=self.Rstar
                        #        inp['bgrid']=bgrid
                        #        inp['r']=r
                        #        ut.savepickle(inp,'testTransmissionCalc.pkl')

        tau=np.zeros([nb,self.nWave]) #no dim
        precision=1
        for ib in range(0,nb):
            
            #Integrate along line of sight with impact parameters b

            rTemp=self.r[r>b[ib]]
            ext=extinctCoef[r>b[ib]]
            ext=np.append(ext,extinctCoef[len(ext)][None,:],axis=0)
            
            rForSplit=np.append(rTemp,b[ib])
            rTemp=np.append(rTemp,r[len(rTemp)])
            step=(rForSplit[:-1]-rForSplit[1:])/precision
            split=np.linspace(precision,0,precision+1)
            finerR=step[:,None]*split[None,:]+rForSplit[1:][:,None]
            finerR=finerR.flatten()
            
            extInter=np.zeros((finerR.shape[0],ext.shape[1]))
            finerX=np.sqrt((finerR)**2-b[ib]**2)

            for i in range(0,len(finerR)):
                extInter[i]=ut.interp1bb(np.log10(rTemp),ext,np.log10(finerR[i]))


            #extInter=ut.interp1bb(np.log10(rTemp),ext,np.log10(finerR))
            
           # rCalc=np.append(rTemp,self.r[len(rTemp)])
            
            #split=np.linspace(precision,0,precision+1)
            #size=len(rCalc)-1
            #finerIntegrand=np.zeros((size*(precision+1),ext.shape[1]))
            #finerX=np.zeros(size*(precision+1))
            #pdb.set_trace()

            # for i in range(0,size):
            #     finerR=step[i]*split+rForSplit[i+1]
            #     for j in range(0,precision+1):
            #       finerIntegrand[i*(precision+1)+j]=ext[i+1]*(((ext[i]/ext[i+1])))**(((finerR[j]-rCalc[i+1])/(rCalc[i]-rCalc[i+1])))
            #     finerX[i*(precision+1):(i+1)*(precision+1)]=np.sqrt(finerR**2-b[ib]**2)
            
           #  pdb.set_trace()
            
           # # finerR=np.einsum('i',step,split,rForSplit[1:])
            
                 
        
           #  finerR=step[:,None]*split[None,:]+rCalc[1:][:,None]
           #  rCalc=rCalc[:,None]
           #  exponent=((finerR-rCalc[1:])/(rCalc[:-1]-rCalc[1:]))
           #  base=(ext[:-1]/ext[1:])
            
            
            # z=np.append(zTemp,self.z[len(zTemp)])
            # z=z[:,None]
            

            # finerIntegrand=ext[1:][:,None,:]*(((ext[:-1]/ext[1:]))[:,None,:])**(((finerZ-z[1:])/(z[:-1]-z[1:]))[:,:,None])
            # finerX=np.sqrt((finerZ+self.RpBase)**2-b[ib]**2)
            # finerX=finerX.flatten()

            # finerIntegrand=finerIntegrand.reshape(finerIntegrand.shape[0]*finerIntegrand.shape[1],finerIntegrand.shape[2])

            

            tau[ib,:]=ut.trapzMultiTimesTwo(x=-finerX,y=extInter)

            #        print ib, (b[ib]**2)/(self.Rstar**2)*1e6,'ppm  tau=',tau[ib,2700]  #corresponds to transit depth
            #        if ib==40:
            #            pdb.set_trace()
    
            #        if np.any(np.isnan(l)):
            #            print('ERROR: nan values in l. Probably rgrid extends to negative values.')
            #            pdb.set_trace()

        #Transmission 
        transmis=np.exp(-tau) #no dim
        
        #print tau[:,5140]
        #print Transmis[:,5140]
        
        dEpsilon= (2*pi*b*db)[:,np.newaxis] *   (1-transmis) #m^2
        
        #Area blocked by atmo
        Epsilon=np.sum(dEpsilon,axis=0) #m^2
        
        #Transit depth = Area blocked by atmo + (effective) planet radius
        TransitDepth=(Epsilon+pi*bmin**2)/(pi*self.Rstar**2) #no dim
        dppm=TransitDepth*1e6
        
        if saveOpac:
            self.opac['tauTransit']     =tau
            self.opac['transmisTransit']=transmis
        
        if np.any(np.isnan(dppm)):
            print(self.params)
            pdb.set_trace()
        return dppm
    
    def calcTransitSpectrumC(self):
        return FinerTransmission.calcTransitSpectrumFiner(self.extinctCoef,self.RpBase,self.r,10,self.Rstar)
        



    def calcEmissionSpectrum(self, extinctCoef, saveOpac=False, muObs=np.array([0.5773502691896258]), FluxUnit='W/(m**2*um)'):
        '''
        Computes thermal emission considering the presence of clouds
        We assume emission coming from beneath the cloud deck is completely 
        absorbed in the cloud.
        --------------------------
        calcEmissionSpectrum(muObs,OutputUnit)
        muObs set to nominal value unless specified
        OutputUnit set to W/(m**2**um) unless specified
        
        Output is the intensity of the thermal emission in shape: (self.nWave,) OR (self.nWave,len(muObs)) if len(muObs)>1 
        '''
       
        # I first define this standard self.dz which is used in other functions since I don't
        # want to implement this new z grid in all of the code
        self.dz = -np.diff(self.z)
        
        # We define a new grid (Pressure, Temperature, Altitude) which will
        # stop at the cloud deck as if it was our ground. 
        
        # From the cloud pressure pCloud we interpolate to get the cloud temperature TCloud
        # and the cloud altitude zCloud. To do so, we use linear interpolation defined in utilities
        # but with the pressure in log scale.
        
        # Here is the if statement which defines the different grids depending
        # on the presence of clouds. 
        
        includeGrayCloudDeck=self.includeGrayCloudDeckThermal

        #---Set the local variable (T, z, lay) for the thermal emission integration---------------
        if includeGrayCloudDeck and self.modelSetting['CloudTypes'][0]:
            TCloud = ut.interp1dEx(np.log(self.p),self.T,np.log(self.params['pCloud']))
            zCloud = ut.interp1dEx(np.log(self.p),self.z,np.log(self.params['pCloud']))
            
            # We are now ready to create our new grid, which will stop at the cloud. In 
            # this calculation, we only need to define it for T and z. 
            T = np.append(self.T[0:np.searchsorted(self.p, self.params['pCloud'], side='left')], TCloud)
            z = np.append(self.z[0:np.searchsorted(self.p, self.params['pCloud'], side='left')], zCloud)
            
            lay = len(T) # This is the length of our new grid
        else:
            T = self.T
            z = self.z
            lay = self.nLay
            
        
        #---Compute thermal emission spectrum------------------------------------------------------

        # Defn req parameters
        TSurf     = T[-1]                       # Surface temp defn'ed at last entry of T -> T[-1]
        dz   = -np.diff(z)        
        #dtau = extinctCoef * np.tile(dz[:,np.newaxis],(1,self.nWave))

        #dtau = extinctCoef * np.outer(dz,np.ones(self.nWave))
        dtau = 0.5*(extinctCoef[:lay-1,:]+extinctCoef[1:lay,:]) * np.outer(dz,np.ones(self.nWave, dtype=self.numerical_precision))
        
        #if (np.array_equal(dtau,dtau2)):
            #print "Success!"
            #alternatively just type this into python console
        
        # Build grid of cumulative sums of tau from TOA to ground/cloud, starting at 0
        tau=np.vstack([np.zeros(self.nWave, dtype=self.numerical_precision),np.cumsum(dtau,axis=0)])
        # Ensure taugrid > 0
        tau*=(1+(np.arange(1,(lay+1))[:,np.newaxis])*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-10)+(np.arange(1,(lay+1))[:,np.newaxis]*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-99)

        if self.doTransit:

            ## Compute black body radiation for T and TSurf with radutils
            B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,self.nWave)),np.tile(self.wave[np.newaxis,:],(len(T),1)),'um',FluxUnit,'rad')
            #B = rad.PlanckFct(T,self.wave,'um',FluxUnit,'rad')
            Bsurf = rad.PlanckFct(TSurf,self.wave[np.newaxis,:],'um',FluxUnit,'rad')
        else:
            ## Compute black body radiation for T and TSurf with radutils
            B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,self.nWave)),np.tile(self.wave[np.newaxis,:],(len(T),1)))
            #B = rad.PlanckFct(T,self.wave,'um',FluxUnit,'rad')
            Bsurf = rad.PlanckFct(TSurf,self.wave[np.newaxis,:])

        ## Calculate thermal emission for each muObs 
        IntensityThermalEmission=np.zeros([len(muObs),self.nWave], dtype=self.numerical_precision)   # Create empty array

        for i in range(len(muObs)):         # Loop through all muObs
        
            transmis=np.exp((-tau)/muObs[i])      # e^(-Tv/u) term
            #------------Calculate second term of integral equation: Integration over optical depth------------
            IntensityThermalEmissionAtm=-np.trapz(y=B,x=transmis,axis = 0)
            
            #------------Calculate surface contribution (first term integral equation)------------
            IntensityThermalEmissionGround=Bsurf*transmis[-1,:]
            
            #------------Sum atmospheric and surface contributions ------------
            IntensityThermalEmission[i,:]=IntensityThermalEmissionAtm+IntensityThermalEmissionGround
            
        # This is the thermal emission for each wavelength    
        thermal=np.squeeze(IntensityThermalEmission)
        
        # Check total thermal flux
        #TotalThermalFlux=np.trapz(IntensityThermalEmission[:,0]*np.pi,self.wave)
        #AverageReradiationTemp=(TotalThermalFlux/sigmaSB)**(1/4)                 #only true if entire thermal emission is simulated
        
        # -----------------------------------------------------------------------------------------------------------------------------
        if saveOpac:
            # Here, we compute the full tau and transmission grid associated to the full range of
            # the atmosphere. We do so in order to save the full grid in savespectrum/makestruc
    
            # And here, we save in the struc the full grid
    
            #dtau = extinctCoef * np.tile(self.dz[:,np.newaxis],(1,self.nWave))
            #dtau = extinctCoef * np.outer(self.dz,np.ones(self.nWave))
            dtaufull = 0.5*(extinctCoef[:self.nLay-1,:]+extinctCoef[1:self.nLay,:]) * np.outer(self.dz,np.ones(self.nWave, dtype=self.numerical_precision))
            # Build grid of cumulative sums of tau from TOA to surface, starting at 0
            taufull=np.vstack([np.zeros(len(self.wave), dtype=self.numerical_precision),np.cumsum(dtaufull,axis=0)])
            # Ensure taugrid > 0
            taufull=taufull*(1+(np.arange(1,(self.nLay+1))[:,np.newaxis])*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-10)+(np.arange(1,(self.nLay+1))[:,np.newaxis]*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-99)
    
            for i in range(len(muObs)):                                      # Loop through all muObs
                transmisfull=np.exp((-taufull)/muObs[i])                 # e^(-Tv/u)

            self.opac['tau']        =taufull
            self.opac['transmis']   =transmisfull
            #self.opac['IntensityThermalEmissionAtm']   =IntensityThermalEmissionAtm
            #self.opac['IntensityThermalEmissionGround']=IntensityThermalEmissionGround
        #----------------------------------------------------------------------------------------------------------------------------------    
        # Return spectrum with effective clouds
        return thermal
    
    
                                                            #    def calcEmissionSpectrum(self, extinctCoef, saveOpac=False, muObs=np.array([0.5773502691896258]), FluxUnit='W/(m**2*um)'):
                                                            #        '''
                                                            #        Computes thermal emission
                                                            #        --------------------------
                                                            #        calcEmissionSpectrum(muObs,OutputUnit)
                                                            #        muObs set to nominal value unless specified
                                                            #        OutputUnit set to W/(m**2**um) unless specified
                                                            #        
                                                            #        Output is the intensity of the thermal emission in shape: (self.nWave,) OR (self.nWave,len(muObs)) if len(muObs)>1 
                                                            #        '''
                                                            #        # Defn req parameters
                                                            #        TSurf     = self.T[-1]                       # Surface temp defn'ed at last entry of T -> T[-1]
                                                            #        self.dz   = -np.diff(self.z)        
                                                            #        #dtau = extinctCoef * np.tile(self.dz[:,np.newaxis],(1,self.nWave))
                                                            #
                                                            #        #dtau = extinctCoef * np.outer(self.dz,np.ones(self.nWave))
                                                            #        dtau = 0.5*(extinctCoef[:self.nLay-1,:]+extinctCoef[1:self.nLay,:]) * np.outer(self.dz,np.ones(self.nWave))
                                                            #        
                                                            #        #if (np.array_equal(dtau,dtau2)):
                                                            #            #print "Success!"
                                                            #            #alternatively just type this into python console
                                                            #        
                                                            #        
                                                            #        # Build grid of cumulative sums of tau from TOA to surface, starting at 0
                                                            #        tau=np.vstack([np.zeros(len(self.wave)),np.cumsum(dtau,axis=0)])
                                                            #        # Ensure taugrid > 0
                                                            #        tau=tau*(1+(np.arange(1,(self.nLay+1))[:,np.newaxis])*np.ones([1,self.nWave])*1e-10)+(np.arange(1,(self.nLay+1))[:,np.newaxis]*np.ones([1,self.nWave])*1e-99)
                                                            #        
                                                            #        ## Compute black body radiation for T and TSurf with radutils
                                                            #        B = rad.PlanckFct(np.tile(self.T[:,np.newaxis],(1,self.nWave)),np.tile(self.wave[np.newaxis,:],(len(self.T),1)),'um',FluxUnit,'rad')
                                                            #        #B = rad.PlanckFct(self.T,self.wave,'um',FluxUnit,'rad')
                                                            #        Bsurf = rad.PlanckFct(TSurf,self.wave[np.newaxis,:],'um',FluxUnit,'flux')
                                                            #
                                                            #        ## Calculate thermal emission for each muObs 
                                                            #        IntensityThermalEmission=np.zeros([len(muObs),self.nWave])       # Create empty array
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
                                                            #        #TotalThermalFlux=np.trapz(IntensityThermalEmission[:,0]*np.pi,self.wave)
                                                            #        #AverageReradiationTemp=(TotalThermalFlux/sigmaSB)**(1/4)                 #only true if entire thermal emission is simulated
                                                            #        
                                                            #        if saveOpac:
                                                            #            self.opac['tau']        =tau
                                                            #            self.opac['transmis']   =transmis
                                                            #            #self.opac['IntensityThermalEmissionAtm']   =IntensityThermalEmissionAtm
                                                            #            #self.opac['IntensityThermalEmissionGround']=IntensityThermalEmissionGround
                                                            #            
                                                            #        # Return spectrum
                                                            #        thermal=np.squeeze(IntensityThermalEmission)
                                                            #        return thermal
    

    #%% Scattering
    #multiScatToon
    def multiScatToon(self,IrradStar,extinctCoef,scatCoef,temperature_profile,asym = 0,u0 = 0.5773502691896258, intTopLay=0,method_for_therm ='Hemispheric Mean', method_for_ref='Quadrature', hard_surface=False, surf_reflect=0, mid=False, ftau_cld=0, b_top_for_ref=0, ref=True, therm=True):
        print('Running Toon')
        
        nlev=self.nLay # number of levels (60 in Scarlet)
        nlay=self.nLay-1 # Number of layers (Interfaces between Levels)
        w0=scatCoef/extinctCoef # Single scattering albedo on each level
        w0=0.5*(w0[:nlev-1,:]+w0[1:nlev,:]) #Single scattering albedo interpolated for on each layer
        nWave=self.nWave # size of wavelength grid (unit of wavelengths: um)
        
        T = temperature_profile # Temperature grid (units of temperatures: K)
        z = self.z # altitude grid (units: m)
        
        u1=0.5
        
        dz   = -np.diff(z) # vertical thickness of each layer (units: m)

        #build grid of tau on each layer (Same code as in calcEmissionSpectrum)
        dtau = 0.5*(extinctCoef[:nlev-1,:]+extinctCoef[1:nlev,:]) * np.outer(dz,np.ones(self.nWave, dtype=self.numerical_precision)) # 
        #if (np.array_equal(dtau,dtau2)):
            #print "Success!"
            #alternatively just type this into python console
            
        # Build grid of cumulative sums of tau from TOA to ground/cloud, starting at 0 (Same code as in calcEmissionSpectrum)
        tau=np.vstack([np.zeros(self.nWave, dtype=self.numerical_precision),np.cumsum(dtau,axis=0)])
        # Ensure taugrid > 0 (Same code as in calcEmissionSpectrum)
        tau*=(1+(np.arange(1,(nlev+1))[:,np.newaxis])*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-10)+(np.arange(1,(nlev+1))[:,np.newaxis]*np.ones([1,self.nWave], dtype=self.numerical_precision)*1e-99)
        
        if therm==True: # calculates upwards and downwards fluxes on each layer (adapted from Picasso)
            bb_all=np.zeros((nlev,self.nWave))
            for i in range(nlev):
                bb_all[i]=rad.PlanckFct(T[i],self.wave,InputUnit='um',OutputUnit='W/(m**2*um)',RadianceOrFlux='rad') #Planck Function calculated on each level in units of W/(m**2*um)
        
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
        
            exptrm_positive = np.exp(exptrm) 
            exptrm_minus = 1.0/exptrm_positive

            #for flux heating calculations, the energy balance solver 
            #does not like a fixed zero at the TOA. 
            #to avoid a discontinuous kink at the last atmospher
            #layer we create this "fake" boundary condition
            #we imagine that the atmosphere continus up at an isothermal T and that 
            #there is optical depth from above the top to infinity 
            tau_top = dtau[0,:]*self.p[0]/(self.p[1]-self.p[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
            #print(list(tau_top))
            #tau_top = 26.75*plevel[0]/(plevel[1]-plevel[0]) 
            b_top = (1.0 - np.exp(-tau_top / u1 )) * bb_all[0,:] * np.pi #  Btop=(1.-np.exp(-tautop/ubari))*B[0]
        
            if hard_surface:
                b_surface = bb_all[-1,:]*pi #for terrestrial, hard surface  
            else: 
                b_surface= (bb_all[-1,:] + b1[-1,:]*u1)*pi #(for non terrestrial)
            
            A, B, C, D = setup_tri_diag(nlay,self.nWave,  C_plus_up, C_minus_up, 
                                    C_plus_down, C_minus_down, b_top, b_surface, surf_reflect,
                                    Gam, dtau, 
                                    exptrm_positive,  exptrm_minus) 
        
        
            positive = np.zeros((nlay, self.nWave))
            negative = np.zeros((nlay, self.nWave))

            #========================= Start loop over wavelength =========================
            L = 2*nlay
            for w in range(self.nWave):
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

            flux_minus_all = np.zeros((nlev,self.nWave))
            flux_plus_all = np.zeros((nlev,nWave))
            if mid==True:
                flux_minus_mdpt_all = np.zeros((nlev,nWave))
                flux_plus_mdpt_all = np.zeros((nlev,nWave))
        
                exptrm_positive_mdpt = np.exp(0.5*exptrm) 
                exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

            #================ START CRAZE LOOP OVER ANGLE #================
            flux_at_top_all = np.zeros((self.nWave))
            flux_down_all = np.zeros((self.nWave))

        
            ulist=np.array([0.09853,0.30453,0.56202,0.80198,0.96019])
            gaussweights=np.array([0.015747,0.073908,0.146386,0.167174,0.096781])
            
            for g in range(len(ulist)):
            
                flux_at_top = np.zeros((self.nWave))
                flux_down = np.zeros((self.nWave))
                
                flux_minus = np.zeros((nlev,self.nWave))
                flux_plus = np.zeros((nlev,nWave))
                if mid==True:
                    flux_minus_mdpt = np.zeros((nlev,nWave))
                    flux_plus_mdpt = np.zeros((nlev,nWave))
                
                iubar = ulist[g]
                weight=gaussweights[g]

                if hard_surface:
                    flux_plus[-1,:] = bb_all[-1,:] *2*pi # terrestrial flux /pi = intensity #2 removed
                else:
                    flux_plus[-1,:] = ( bb_all[-1,:] + b1[-1,:] * iubar)*2*pi#no hard surface #2 removed     
                    
                flux_minus[0,:] = (1 - np.exp(-tau_top / iubar)) * bb_all[0,:] *2*pi  #2 removed
                
                exptrm_angle = np.exp( - dtau / iubar)
                if mid==True:
                    exptrm_angle_mdpt = np.exp( -0.5 * dtau / iubar) 

                for itop in range(nlay):

                    #Equation 56 in Toon et Al.
                    flux_minus[itop+1,:]=(flux_minus[itop,:]*exptrm_angle[itop,:]+
                                            (J[itop,:]/(Lam[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                            (K[itop,:]/(Lam[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                            sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                            sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau[itop,:]-iubar) )

                    #Equation 56 in Toon et Al.
                    if mid==True:
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
                    if mid==True:
                        flux_plus_mdpt[ibot,:]=(flux_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                            (G[ibot,:]/(Lam[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                            (H[ibot,:]/(Lam[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                            alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                            alpha2[ibot,:]*(iubar+0.5*dtau[ibot,:]-(dtau[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )


                    if mid==True:
                        #Upwards flux at top of atmosphere
                        flux_at_top[:] = flux_plus_mdpt[0,:] #nlevel by nwno 
                    
                flux_minus_all+=flux_minus*weight            
                flux_plus_all+=flux_plus*weight
                
                if mid==True:
                    flux_minus_mdpt_all+=flux_minus_mdpt*weight          
                    flux_plus_mdpt_all+=flux_plus_mdpt*weight
                    flux_at_top_all+=flux_at_top*weight
                
            
    
            if mid==False:
                #thermal_component = flux_plus_mdpt_all,flux_minus_mdpt_all
                thermal_component = flux_plus_all,flux_minus_all
            else:
                thermal_component = (flux_plus_all, flux_minus_all), (flux_plus_mdpt_all,flux_minus_mdpt_all)
            
        if ref==True:
        
            delta_approx = True
            if delta_approx == True :
                dtau=dtau*(1.-w0*asym**2)
                tau[0]=tau[0]*(1.-w0[0]*asym**2)
                for i in range(nlay):
                    tau[i+1]=tau[i]+dtau[i]
            
            ##### --SM-- need to correct the tau arrays first and the w0 and cosb arrays later
                w0=w0*((1.-asym**2)/(1.-w0*(asym**2)))
                asym=asym/(1.+asym)
            
            if method_for_ref == 'Eddington': #Eddington Method
                gam1  = (7-w0*(4+3*ftau_cld*asym))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #Table 1 Toon et Al. 
                gam2  = -(1-w0*(4-3*ftau_cld*asym))/4 #(sq3*w0*0.5)*(1.-cosb)       #Table 1 Toon et Al.
                gam3  = (2-3*ftau_cld*asym*u0)/4 #Table 1 Toon et Al.
            elif method_for_ref == 'Quadrature':#quadrature
                gam1  = (np.sqrt(3)*0.5)*(2. - w0*(1.+ftau_cld*asym)) #Table 1 Toon et Al.
                gam2  = (np.sqrt(3)*w0*0.5)*(1.-ftau_cld*asym)        #Table 1 Toon et Al.
                gam3  = 0.5*(1.-np.sqrt(3)*ftau_cld*asym*u0) #Table 1 Toon et Al.
            
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
            x = np.exp(-taumid/u0)
        
            C_plus_mid=IrradStar*w0* (gam3*(gam1 - 1.0/u0) +gam2*gam4 ) / denominator * np.exp(-taumid/u0) #Equation 23 Toon et Al.  
            C_minus_mid=IrradStar*w0* (gam4*(gam1 + 1.0/u0) +gam2*gam3 ) / denominator * np.exp(-taumid/u0) #Equation 24 Toon et Al.
        
        
            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = Lam*dtau
            #save from overflow 
            exptrm = slice_gt (exptrm, 35.0) 

            exptrm_positive = np.exp(exptrm) #EP
            exptrm_minus = 1.0/exptrm_positive#EM

            #boundary conditions 
            b_top = b_top_for_ref                                      

            b_surface = 0. + surf_reflect*u0*IrradStar*np.exp(-tau[-1, :]/u0)  #Toon et Al. Equation 37 

            #Now we need the terms for the tridiagonal rotated layered method
            #if tridiagonal==0:
            A, B, C, D = setup_tri_diag(nlay,nWave,  C_plus_up, C_minus_up, 
                                    C_plus_down, C_minus_down, b_top, b_surface, surf_reflect,
                                    Gam, dtau, 
                                    exptrm_positive,  exptrm_minus)    

            positive = np.zeros((nlay, self.nWave))
            negative = np.zeros((nlay, self.nWave))
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
                    
            if mid==True:
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
            flux_minus = flux_minus + u0*IrradStar*np.exp(-tau/u0)

            if mid==True:
                #now get midpoint values 
                exptrm_positive_midpt = np.exp(0.5*exptrm) #EP
                exptrm_minus_midpt = 1.0/exptrm_positive_midpt#EM
                    
                #fluxes at the midpoints 
                flux_minus_midpt[:-1,:]= Gam*positive*exptrm_positive_midpt + negative*exptrm_minus_midpt + C_minus_mid #Equation 32 Toon et Al.
                flux_plus_midpt[:-1,:]= positive*exptrm_positive_midpt + Gam*negative*exptrm_minus_midpt + C_plus_mid #Equation 31 Toon et Al.
                #add in midpoint downwelling radiation
                flux_minus_midpt[:-1,:] = flux_minus_midpt[:-1,:] + u0*IrradStar*np.exp(-taumid/u0)
        
            if mid==False:
                #reflected_component = flux_plus_midpt*0.5,flux_minus_midpt*0.5
                reflected_component = flux_plus*0.5,flux_minus*0.5
            else:
                reflected_component = (flux_plus_midpt,flux_minus_midpt), (flux_plus, flux_minus)
        if (therm==True and ref==True):
            if mid==False:
                return thermal_component[0]+reflected_component[0],thermal_component[1]+reflected_component[1]
            else:
                return (thermal_component[0][0]+reflected_component[0][0],thermal_component[0][1]+reflected_component[0][1]),(thermal_component[1][0]+reflected_component[1][0],thermal_component[1][1]+reflected_component[1][1])
        elif therm==True:
            return thermal_component
        elif ref==True:
            return reflected_component
        else:
            print('aa')

    
    
    
    def calcAlbedo(self,method = 'Eddington',omega = -1,geo = False):
        extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
        scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
        IrradStar = np.ones_like(self.wave, dtype=self.numerical_precision)

        Fupw, Fdwn, Fnet = self.multiScatToon(IrradStar,extinctCoef,scatCoef,refl = True, thermal = False, method = method,w=omega)
        albedo1D = Fupw[0,0]
        
        if geo:
            uArr = np.arange(1,11)
            albedo = np.zeros([len(uArr)], dtype=self.numerical_precision)
            
            for u in uArr:
                Fupw, Fdwn, Fnet = self.multiScatToon(IrradStar,extinctCoef,scatCoef,u0 = (u/10.0),refl = True, thermal = False, method = method,w=omega)
                albedo[u-1] = Fupw[0,0]
            geoAlbedo = np.trapz(y=albedo,x=uArr/10.0)
            #geoAlbedo = np.mean(albedo)
        
            return albedo1D,geoAlbedo
        else:
            return albedo1D
        
    def calcDisort(self,IrradStar,extinctCoef,scatCoef,asym = 0,thermal=False,
                   umu0 = 1./np.sqrt(2.),phi0 = 0.,
                   prnt = np.array([False, False, False, False, False])):

        dTau = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:]) * np.outer(self.dz,np.ones(self.nWave, dtype=self.numerical_precision))
        N_tau = len(dTau)
        iphas  = np.ones(N_tau,dtype='int')*2   # Rayleigh Scattering
        gg     = np.ones(N_tau)*asym            # Asymmetry parameter

        fbeam  = IrradStar/umu0 # Ensures fluxes to be normalized to one
        
        albedo = self.BondAlbedo  #shouldn't this be the ground albedo?
        w0 = scatCoef/extinctCoef
        umu    = np.array([-1.,-0.5,0.5,1.])
        phi    = np.array([0.,60.,120.])
        
        temp   = self.T
        btemp = self.T[-1]
        ttemp = self.T[0]
        wvnmlo = 999.
        wvnmhi = 1000.
        
        Fupw = np.zeros([self.nWave], dtype=self.numerical_precision)
        for i in range(self.nWave):
            print(i,)
            [rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed] =\
                                          disort.run(dTau[:,i], w0=w0[:,i], iphas=iphas, gg=gg,
                                                     umu0=umu0,phi0=phi0,albedo=albedo, fbeam=fbeam,
                                                     #utau=uTau[:,i],
                                                     umu=umu, phi=phi, 
                                                     plank=thermal,prnt=prnt,temp=temp,ttemp=ttemp,btemp=btemp, wvnmlo=wvnmlo,wvnmhi=wvnmhi,
                                                     verbose = False)
            Fupw[i] = flup[0]
        
        if thermal:
            return Fupw
        else:
            return Fupw#/fbeam
    
    def testDisort(self,IrradStar,extinctCoef,scatCoef,asym = 0,u0 = 0.5773502691896258,thermal=False,wv = 134,w0 = None,
                   phi0=0.,phi=0.,umu0=1.,umu=1.,
                   temp=300.,btemp=300.,ttemp=300.,wvnmlo = 999.,wvnmhi = 1000.,
                   prnt=np.array([False, False, False, False, False])):

        dTau = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:]) * np.outer(self.dz,np.ones(self.nWave, dtype=self.numerical_precision))
        N_tau = len(dTau)
        iphas  = np.ones(N_tau,dtype='int')*2
        gg     = np.zeros(N_tau, dtype=self.numerical_precision)
        fbeam  = IrradStar  # Ensures fluxes to be normalized to one
        albedo = self.BondAlbedo
        
        if w0 is None:
            w0 = scatCoef/extinctCoef
        else:
            w0 =w0*np.ones_like(dTau, dtype=self.numerical_precision)
        
        [rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed] =\
                                      disort.run(dTau[:,wv], w0=w0[:,wv], iphas=iphas, gg=gg,
                                                 umu0=umu0,phi0=phi0, albedo=albedo, fbeam=fbeam,umu=umu, phi=phi, 
                                                 plank=thermal,prnt=prnt,temp=temp,ttemp=ttemp,btemp=btemp, wvnmlo=wvnmlo,wvnmhi=wvnmhi,
                                                 verbose = False)

        Fupw = flup[0]/fbeam[wv]
        return Fupw

    
    
    #%% Self-Consistent Tp Profiles


    def calcNonGrayTpProf(self,modelSetting,params,firstIter,LucyUnsold,runConvection):
    
        #Option for testing: start on a selected temperature structure
        if 0:
            T = ut.loadpickle('test.pkl')
            return T
        
        # Lucy Unsold
        if LucyUnsold:
            print('Lucy Unsold Brown Dwarf:')
            T = deepcopy(self.T)
            #T=np.interp(np.log(self.p),np.log(params['Tprof'][0]),params['Tprof'][1])
            
            self.qmol_lay                                                                                = self.calcComposition(modelSetting,params,T)
            self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r  = self.calcHydroEqui(modelSetting,params,T)
            self.extinctCoef,self.absorbCoef, self.scatCoef                                              = self.calcOpacities(modelSetting,params,T)

#            Lstar = 4.0*np.pi*sigmaSB*self.Rstar**2*self.Teffstar**4
#            self.Teff = (Lstar*(1-self.BondAlbedo)/(16.0*np.pi*sigmaSB*self.ap**2))**0.25
            #IrradStar = rad.PlanckFct(self.Teffstar,self.wave,'um','W/(m**2*Hz)','rad')*(self.Rstar/au)**2*(au/self.ap)**2/4.0

            #----------------
            #B,J,K,H,mExtinctCoef,mAbsorbCoef = self.solveRTE(T,modelSetting,params,IrradStar,TpCorrLucy= True)
            B,J,K,H,mExtinctCoef,mAbsorbCoef = self.solveRTE(T,modelSetting,params,TpCorrLucy = True)
            TList,dTList,dHList = self.TpCorrection(T,modelSetting,params,B,J,K,H,mExtinctCoef,mAbsorbCoef)
            T = TList[:,-1]
            self.TList = TList
            self.dTList = dTList
            self.dHList = dHList
            ut.savepickle(T,'test.pkl')
            
        # Linearized Tp model
        if (modelSetting['maxNonGrayIter'] is None) or (modelSetting['maxNonGrayIter']> self.nonGrayIter):
            print('Calculate Self-Consistent TP:')
            
            if self.Teq<400:
                print('Quadrupling the max # of iterations to accomodate low Teq of planet')
                modelSetting['maxIterForNonGrayTpCalc'] = modelSetting['maxIterForNonGrayTpCalc']*4
            
            elif self.Teq<800:
                print('Tripling the max # of iterations to accomodate low Teq of planet')
                modelSetting['maxIterForNonGrayTpCalc'] = modelSetting['maxIterForNonGrayTpCalc']*3
            
            elif modelSetting['ComposType']=='ChemKinetic':
                print('Tripling the max # of iterations to accomodate iterations with chemical kinetics')
                modelSetting['maxIterForNonGrayTpCalc'] = modelSetting['maxIterForNonGrayTpCalc']*3
            
            #Set initital conditions for temperature profile T                
            if LucyUnsold:
                # Use Lucy Unsold tp profile as starting point if it was used 
                T = ut.loadpickle('test.pkl')       # Load from Lucy Unsold
            else:
                if (firstIter is True):
                    if 'Tprof' in params:
                        T = np.interp(np.log(self.p),np.log(params['Tprof'][0]),params['Tprof'][1])
                    else:
                        T=self.Teq*np.ones_like(self.p, dtype=self.numerical_precision)*1.4
                        
                    # Initialize chemistry, hydrostatic equilibrium, and opacities via ChemEqui
                    if self.verbose: print('\nInitializing chemistry in calcNonGrayTpProf()\n')
                    
                    self.qmol_lay                                                                                = self.calcComposition(modelSetting,params,T)
                    self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r  = self.calcHydroEqui(modelSetting,params,T)
                    self.rho                                                                                     = self.ntot * self.MuAve
        
                    self.extinctCoef,self.absorbCoef, self.scatCoef                                              = self.calcOpacities(modelSetting,params,T)
        
                    # calculate integration weights so that later we can sum instead of integrate    
                    # weights = self.calcWeight()
                    
                    # modelSetting['ComposType'] = finalComposType
                    
                else:
                    T=self.T  #Use temperature structure from previous iteration as initital condition 

            # # Initialize chemistry, hydrostatic equilibrium, and opacities
            # self.qmol_lay                                                                                = self.calcComposition(modelSetting,params,T)
            # self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r  = self.calcHydroEqui(modelSetting,params,T)
            # self.rho                                                                                     = self.ntot * self.MuAve

            # self.extinctCoef,self.absorbCoef, self.scatCoef                                              = self.calcOpacities(modelSetting,params,T)

            # # calculate integration weights so that later we can sum instead of integrate    
            weights = self.calcWeight()

                                #Stellar irradiation from the top (needs to be intensity in 'W/(m**2*Hz)')
                                        #IrradStar=self.IrradStarEffIntensityPerHz
                                        #    IrradStarOld = rad.PlanckFct(self.Teffstar,self.wave,'um','W/(m**2*Hz)','rad') * (self.Rstar/au)**2*(au/self.ap)**2 * params['HeatDistFactor'] * (1-self.BondAlbedo)
                                        #    fig,ax=plt.subplots()
                                        #    ax.plot(self.wave,IrradStar)#self.IrradStarTpProfCalc)
                                        #    ax.plot(self.wave,IrradStarOld)
            
            # Run RTE solver
            B,J,K,H,fn,Gfn,dtau,lmbda,extinctCoef,mExtinctCoef,absorbCoef = self.solveRTE(T,modelSetting,params,self.IrradStarEffIntensityPerHz,TpCorrLin = True)
                                #self.taugrid = np.vstack([np.zeros(len(self.wave)),np.cumsum(dtau,axis=0)])
            
            # Save iterations of T and dT
            TList = deepcopy(T)
            dTList = np.zeros([self.nLay], dtype=self.numerical_precision)
            errorList = np.zeros([self.nLay], dtype=self.numerical_precision)

            # Iterate until convergence criteria of T are met
            runConvection = False # Don't run convection until after 1st convergence for stability

            #print 'Only 1 Loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            for loop in range(2):
                i = 1
                not_converged = True
                prev_best_dT = 500.
                
                prev_iter = np.zeros(self.nLay, dtype=self.numerical_precision)
                
                while(not_converged):
                    print(i,)
                    
                    nNonConvergedLay = 0
                    
#                    # fills up holes in the convective zones if they are preventing convergence 
#                    if i > 20:
#                        force_conv_lay = True
                    
                    # Call Tp linearization method
                    dT,dJ= self.correcT(T,B,J,fn,Gfn,self.IrradStarEffIntensityPerHz,dtau,lmbda,mExtinctCoef,extinctCoef,absorbCoef,weights,loop,runConvection)
#                    dT,dJ= self.correcT2(T,B,J,fn,Gfn,self.IrradStarEffIntensityPerHz,dtau,lmbda,mExtinctCoef,extinctCoef,absorbCoef)

                    #print dT                    

                    #Slow down jumps if dT is too large (for stability)
                    for k in range(self.nLay):
                        if np.abs(dT[k])> 0.6*T[k]:
                            dT[k] = 0.6*T[k]*np.sign(dT[k])
                        if self.Teq<400:
                            if np.abs(dT[k])>0.3*T[k]:
                                dT[k] = 0.3*T[k]*np.sign(dT[k])
                        
                        if runConvection and self.Teq<800: # Slow down even more when running convection
#                            if np.abs(dT[k])>0.15*T[k]:
#                                dT[k] = 0.15*T[k]*np.sign(dT[k])
                             if np.abs(dT[k])>0.4*T[k]:
                                dT[k] = 0.4*T[k]*np.sign(dT[k])
                                
                        elif runConvection and self.Teq<400:
                            if np.abs(dT[k])>0.2*T[k]:
                                dT[k] = 0.2*T[k]*np.sign(dT[k])
                        
                        # If iteration is -ve of the previous iteration, only apply 1/2 of the dT
                        if np.round(dT[k]) == np.round(prev_iter[k])*-1.:
                            dT[k] = dT[k] / 2.
                    
                    prev_iter = deepcopy(dT)
                    # Calculate new T (in this next iteration)
                    T = T+dT
                                            
                    # Check temperature convergence of all layers
                    for d in range(self.nLay):
                        if np.abs(dT[d]/(T[d]-dT[d])) < 1e-3:   #Convergence criteria
                            nNonConvergedLay += 0
                        else: 
                            nNonConvergedLay += 1
                            
                    if nNonConvergedLay == 0:
                        #--> Tp profile is converged for this molecular composition
                        not_converged = False
                        if loop == 0:
                            print('')
                            print('Checking chemistry/hydrostatic equilibrium convergence',)
                            self.T = self.TList[:,-1]
                            self.makeStruc()
                            if modelSetting['TempType']=='NonGrayConv':
                                print('and adding convection')
                                runConvection = True
                                self.convLay = '(Not yet computed)'
                            print('')
                        if loop == 1:
                            print('')
                            print('dJ: ', np.amax(np.abs(dJ)))
                            print('Successfully converged')
                            
                    elif np.amax(np.abs(dT)) < prev_best_dT:
                        # Save best iteration
                        best_T = deepcopy(T)
                        prev_best_dT = np.amax(np.abs(dT))
                        if runConvection:
                            prev_best_convLay = self.convLay
                        
                            
                    # DO NOT ALLOW FOR INVERSIONS
                    if runConvection and self.Teq<400: # Slow down even more when running convection
                        if len(self.convLay) != 0:
                            for k in range(self.nLay):
                                if k >= self.convLay[0] and T[k] < T[k-1]:
                                    dT[k] = T[k-1]-T[k]
                                    T[k] = T[k-1]


                    # Save iterations of T and dT
                    TList = np.c_[TList,T]
                    dTList = np.c_[dTList,dT]

                    errorList = np.c_[errorList,np.abs(dT/(T-dT))]
                    
                    self.errorList = errorList
                    self.dTList = dTList
                    self.TList = TList     

                    if not_converged:
                        # Iterate
                        # if loop >= 1: # Add stability by changing chemistry/hydroequilibrium only after 1st convergence
                            
                        #     # if (firstIter is True):
                        #     #     print('\n Going through firstIter!!!!!\n')
                        #     #     finalComposType = modelSetting['ComposType']
                        #     #     modelSetting['ComposType'] = 'ChemEqui'
                                
                        #     #     self.qmol_lay                                                                                = self.calcComposition(modelSetting,params,T)
                        #     #     self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r  = self.calcHydroEqui(modelSetting,params,T)
                                
                        #     #     modelSetting['ComposType'] = finalComposType
                                
                        #     # else:
                        #     #     print('\n going wrong branch!!!\n')
                        #     self.qmol_lay                                                                                = self.calcComposition(modelSetting,params,T)
                        #     self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r  = self.calcHydroEqui(modelSetting,params,T)
                                
                        
                        B,J,K,H,fn,Gfn,dtau,lmbda,extinctCoef,mExtinctCoef,absorbCoef = self.solveRTE(T,modelSetting,params,self.IrradStarEffIntensityPerHz,TpCorrLin=True)
                        i+=1
                    
                    if self.plotTpChangesEveryIteration:
                        self.plotTpChanges(save=True,close=True,loop=loop)

                    #Display some convergence measures (just for user to see convergence)
                    IdownPerHz = self.IrradStarEffIntensityPerHz
                    IupPerHz   = 4.0 * H[0,:] - IdownPerHz
                    FupPerHz   = pi * IupPerHz
                    TotalFluxUpTOA  = - np.trapz(x=self.f,y=FupPerHz)                                 #Minus sign because self.f goes from high to low
                    TotalIrradTOA   = - np.trapz(x=self.f,y=self.IrradStarEffIntensityPerHz) * np.pi  #Minus sign because self.f goes from high to low
                    netFluxTOA=TotalFluxUpTOA-TotalIrradTOA
                    print('\nTotal flux up at TOA:     ({:6g} W/m**2)'.format(TotalFluxUpTOA))
                    print('Total irradiation at TOA: ({:6g} W/m**2)'.format(TotalIrradTOA))
                    if runConvection:
                        print('dTmax = {:7g} |  dJmax = {:7g} |  flux residual: {:7g} W/m**2 ({:5g}%) | Conv={}'.format(np.amax(np.abs(dT)),np.amax(np.abs(dJ)),netFluxTOA,netFluxTOA/TotalIrradTOA*100,str(runConvection) + ' for layers:' + str(self.convLay)))
                    else:
                        print('dTmax = {:7g} |  dJmax = {:7g} |  flux residual: {:7g} W/m**2 ({:5g}%) | Conv={}'.format(np.amax(np.abs(dT)),np.amax(np.abs(dJ)),netFluxTOA,netFluxTOA/TotalIrradTOA*100,runConvection))

                    #fig,ax=plt.subplots(); ax.plot(self.wave, IrradStar); ax.semilogx(cLight/self.f*1e6,H[0,:])
                    #fig,ax=plt.subplots(); ax.plot(self.f, IrradStar); ax.plot(self.f,H[0,:])

                    #Check whether maximum number of iterations is reached (user specified)
                    if i == modelSetting['maxIterForNonGrayTpCalc']:
                        print('Reached max # of iterations: using Tp profile with lowest dT (dTmax = {:7g})'.format(prev_best_dT))
                        self.TList = TList            
                        not_converged=False
                        T = deepcopy(best_T)
                        if runConvection:
                            self.convLay = prev_best_convLay
                        
            
            #plt.close('all')
            self.TList = TList
            self.nonGrayIter += 1
#        else:
#            T = self.TList[:,-1]
            
                    #--------------------------------------------
            
#        elif modelSetting['TempType']=='GrayNoIrrad':
#            
#            Tprof=np.vstack([ self.p , params['Tint']*np.ones(self.nLay) ])   
#
#            fig,ax=ut.newFig(xlabel='T',ylabel='Pressure [bar]',log=[False,True],reverse=[False,True])
#            for i in range(3):          
#                [T,_,zgrid,_,_,_,_,_,extinctCoef] = self.mlab.AtmosphereModel(inputs,qmol_lay,Tprof,carmaFile,
#                            modelSetting['ComposType'],'parameters',modelSetting['CloudTypes'],
#                            np.array([0,0,0,0,0,0]),self.muObs[0],0,1,1,nout=9)
#                zgrid       = zgrTid
#                extinctCoef = extinctCoef[:,::-1]
#                dz          = -np.diff(zgrid,axis=0)
#            
#                dtau = dz * extinctCoef
#                dtauR = rad.rosselandMean(self.wave,dtau,T,waveUnit='um')
#                
#                tauR = np.cumsum(dtauR)
#                Tnew = rad.TpGrey(tauR,Teff=params['Tint'])          
#                Tprof=np.vstack([ self.p , Tnew ])   
#                
#                ax.plot(Tprof[1,:],Tprof[0,:])
#
#            Tprof=np.vstack([ self.p , Tnew ])   
#            TempType = 'parameters'

        return T

    def calcNonGrayTpTradProf(self,modelSetting,params,firstIter,LucyUnsold,runConvection):   
        
        print('running TpTrad')
        print(f'modelSetting: {modelSetting}')
        print(f'params: {params}')
        print(f'firstIter: {firstIter:}')
        print(f'LucyUnsold: {LucyUnsold}')
        print(f'runConvection: {runConvection}')
        
        T = self.Teq*np.ones_like(self.p, dtype=self.numerical_precision)*1.4
        N=self.nLay
        levels=deepcopy(T)    
        self.TList =np.array([np.array([level]) for level in deepcopy(levels)])
        
        #self.calcAtmosphere(modelSetting,params,updateOnlyDppm=False,updateTp=False,disp2terminal=False,returnOpac=False,
        #               thermalReflCombined=None,thermalOnly=None,albedoOnly=None, low_res_mode=False)
        
        self.qmol_lay                                                                               = self.calcComposition(modelSetting,params,levels, firstIter=False)
        #self.executeOption()
        self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r = self.calcHydroEqui(modelSetting,params,levels)
        self.extinctCoef,self.absorbCoef, self.scatCoef                                             = self.calcOpacities(modelSetting,params,levels,saveOpac=self.saveOpac, low_res_mode=False)
        
        extinctCoef=self.extinctCoef.copy()
        scatCoef=self.scatCoef.copy()       
    
        #------------------------------------------------------------------------#
        #---------------Calculate temperature perturbations----------------------#
        #------------------------------------------------------------------------#
     
        deltas=np.ones(N)*11
        count=0
        while np.max(np.abs(deltas))>0.2: #0.5:
            deltas,levels=self.take_step_tp_trad(self.TList,levels,N,extinctCoef,scatCoef,loop=False)
            self.TList = np.c_[self.TList,levels]
            
            #self.T=levels 
            #self.calcAtmosphere(modelSetting,params,updateOnlyDppm=False,updateTp=False,disp2terminal=False,returnOpac=False,
            #           thermalReflCombined=None,thermalOnly=None,albedoOnly=None, low_res_mode=False)
            #extinctCoef=self.extinctCoef.copy()
            #scatCoef=self.scatCoef.copy()   
            
            self.qmol_lay                                                                               = self.calcComposition(modelSetting,params,levels, firstIter=False)
            #if self.verbose: print('Going to executeOption(), calcHydroEqui and calcOpacities in calcAtmosphere()\n')
            #self.executeOption()
            self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r = self.calcHydroEqui(modelSetting,params,levels)
            self.extinctCoef,self.absorbCoef, self.scatCoef                                             = self.calcOpacities(modelSetting,params,levels,saveOpac=self.saveOpac, low_res_mode=False) 
            
            extinctCoef=self.extinctCoef.copy()
            scatCoef=self.scatCoef.copy()
            
            count+=1
            print(count)         
        
        print('Temperature Profile Converged')
        
        return levels  
    
    def take_step_tp_trad(self,TList,in_levels,N,extinctCoef,scatCoef,loop=False): #*m
        #plotTp(forceLabelAx=True)
        #plotTpChanges(TList)
        toon= self.multiScatToon(self.IrradStar,extinctCoef,scatCoef,in_levels)
    
        if self.plotTpChangesEveryIteration:
            self.plotTpChanges(save=True,close=True,loop=loop)
        
        flux0=toon[0]-toon[1] 
        delflux=-1*flux0
        
        A=np.zeros((N,N),dtype=self.numerical_precision)
    
        for level_i in range(N):
            print('*****')
            print(level_i)
            print('*****')
            deltaT=0.001
            levels_ptb=in_levels.copy()
            levels_ptb[level_i]=in_levels[level_i]+deltaT
            flux_ptb_all=self.multiScatToon(self.IrradStar,extinctCoef,scatCoef,levels_ptb)
        
            flux_ptb=flux_ptb_all[0]-flux_ptb_all[1]        
            #flux_ptb_up=flux_ptb_all[0]
            #flux_ptb_down=flux_ptb_all[1]
            A_level_i=(flux_ptb-flux0)/deltaT
            for layer_i in range(N):
                A[layer_i][level_i]=np.trapz(x=self.wave,y=A_level_i[layer_i])        
    
        deltafluxsum=np.zeros(N)
        for i in range(N):
            deltafluxsum[i] = np.trapz(x=self.wave,y=delflux[i])
        delta_levels_lin = np.linalg.lstsq(A, sigmaSB*self.params['Tint']**4+deltafluxsum, rcond=None)[0]
        delta_levels_reduced=delta_levels_lin*0.10
        maxabs=np.max(np.abs(delta_levels_reduced))
        if maxabs>250:
            delta_levels_reduced=delta_levels_reduced*250/maxabs
        new_levels=in_levels+delta_levels_reduced   
    
        return delta_levels_reduced,new_levels
         
        
        



    def solveRTE(self,T,modelSetting,params,IrradStar=0,TpCorrLucy=False,TpCorrLin=False, refl=False):
        if refl:
            B = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        else:
            B = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,self.nWave)),np.tile(self.f[np.newaxis,:],(len(T),1)),'Hz','W/(m**2*Hz)','rad')
        
        extinctCoef,absorbCoef, scatCoef = self.calcOpacities(modelSetting,params,T)
        rho = np.tile((self.ntot*self.MuAve)[:,np.newaxis],(1,self.nWave))
        
        mExtinctCoef = extinctCoef/rho
        mAbsorbCoef = absorbCoef/rho
        #mScatCoef = scatCoef/rho
        lmbda = mAbsorbCoef/mExtinctCoef  # *0.0 +0.000001

        dtau = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        dtau[0,:] = (mExtinctCoef[0,:]*self.p[0])/self.grav[0]
        for i in range(1,self.nLay):
            dtau[i,:] = 0.5*(mExtinctCoef[i,:] + mExtinctCoef[i-1,:])*(self.p[i]-self.p[i-1])/self.grav[i]
        
        J,K,H,fn,Gfn = self.feautrierRTE(lmbda,dtau,B,Hext=IrradStar)

        if TpCorrLucy:
            return B,J,K,H,mExtinctCoef,mAbsorbCoef
        if TpCorrLin:
            return B,J,K,H,fn,Gfn,dtau,lmbda,extinctCoef,mExtinctCoef,absorbCoef
        
        return B,J,K,H

    
    def feautrierRTE(self,lmbda,dtau,B,Hext = 0, mu = np.array([0.8872983346,0.5,0.1127016654]), weight = np.array([0.27777777,0.4444444,0.27777777])):
        
        [imu,l,w] = [mu.size,self.nLay,self.nWave]
        
        plnk = B
        J = np.zeros([l,w], dtype=self.numerical_precision)
        K = np.zeros([l,w], dtype=self.numerical_precision)
        H = np.zeros([l,w], dtype=self.numerical_precision)

        fn = np.zeros([l,w], dtype=self.numerical_precision)
        Gfn = np.zeros([w], dtype=self.numerical_precision)


        # Tier 1: mu
        A = np.zeros([imu,w], dtype=self.numerical_precision)
        C = np.zeros([imu,w], dtype=self.numerical_precision)
        L = np.zeros([imu,w], dtype=self.numerical_precision)
        X = np.zeros([imu,w], dtype=self.numerical_precision)

        # Tier 2: mu*mu
        B = np.zeros([imu,imu,w], dtype=self.numerical_precision)
        N = np.zeros([imu,imu,w], dtype=self.numerical_precision)

        # Tier 3: layered
        U = np.zeros([imu,l,w], dtype=self.numerical_precision)
        R = np.zeros([imu,l,w], dtype=self.numerical_precision)
        D = np.zeros([imu,imu,l,w], dtype=self.numerical_precision)
        
        # TOA _______________________________________
        for j in range(imu):
            C[j,:] = -mu[j]/dtau[1,:]
            L[j,:] = -1.0/2.0*dtau[1,:]/mu[j]*lmbda[0,:]*plnk[0,:] - Hext
            
            for k in range(imu):
                B[j,k,:] = 1.0/2.0*dtau[1,:]/mu[j]*(1.0-lmbda[0,:])*weight[k]
            
            B[j,j,:] = B[j,j,:] + C[j,:] - 1.0 - 1.0/2.0*dtau[1,:]/mu[j]
        
        #Inverse B
        Bflip = B#dbg
        for i in range(imu-1,0,-1):
            Bflip = np.swapaxes(Bflip,i-1,i)
        B = np.linalg.inv(Bflip)
        for i in range(0,imu-1):
            B = np.swapaxes(B,i,i+1)

        for j in range(imu):
            for k in range(imu):
                D[j,k,0,:] = B[j,k,:]*C[k,:]

        for j in range(imu):
            R[j,0,:] = 0
            for k in range(imu):
                R[j,0,:] = R[j,0,:] + B[j,k,:]*L[k,:]
        # Middle _______________________________________
        for d in range(1,l-1):
            for j in range(imu):
                A[j,:] = -mu[j]**2/(dtau[d,:]*1.0/2.0*(dtau[d,:]+dtau[d+1,:]))
                C[j,:] = -mu[j]**2/(dtau[d+1,:]*1.0/2.0*(dtau[d,:]+dtau[d+1,:]))
                L[j,:] = -lmbda[d,:]*plnk[d,:]
                for k in range(imu):
                    B[j,k,:] = (1.0-lmbda[d,:])*weight[k]
                B[j,j,:] = B[j,j,:] + A[j,:] + C[j,:] - 1
            
            for j in range(imu):
                for k in range(imu):
                    N[j,k,:] = B[j,k,:] - A[j,:]*D[j,k,d-1,:]
            
            for j in range(imu):
                X[j,:] = A[j,:]*R[j,d-1,:]+L[j,:]
            
            # Inverse T
            Nflip = N#dbg
            for i in range(imu-1,0,-1):
                Nflip = np.swapaxes(Nflip,i-1,i)
            N = np.linalg.inv(Nflip)
            for i in range(0,imu-1):
                N = np.swapaxes(N,i,i+1)
            
            for j in range(imu):
                for k in range(imu):
                    D[j,k,d,:] = N[j,k,:]*C[k,:]
            
            for j in range(imu):
                for k in range(imu):
                    R[j,d,:] = R[j,d,:] + N[j,k,:]*X[k,:]

        # Bottom _______________________________________
        for j in range(imu):
            U[j,l-1,:] = plnk[l-1,:]
        
        # Reverse Substitution _______________________________
        for d in range(l-2,-1,-1):
            for j in range(imu):
                for k in range(imu):
                    U[j,d,:] = U[j,d,:] + D[j,k,d,:]*U[k,d+1,:]
                U[j,d,:] = U[j,d,:] + R[j,d,:]
        
        
        # Calculate J,K,H
        for d in range(l):
            for j in range(imu):
                J[d,:] = J[d,:]+weight[j]*U[j,d,:]
                K[d,:] = K[d,:]+weight[j]*mu[j]**2*U[j,d,:]
            fn[d,:] = K[d,:]/J[d,:]

        for d in range(1,l):
            H[d,:] = (K[d,:] - K[d-1,:])/dtau[d,:]
        
        for j in range(imu):
            H[0,:] = H[0,:] + weight[j]*mu[j]*U[j,0,:]
            
        Gfn = H[0,:]/J[0,:] 
        return J,K,H,fn,Gfn
    


    ### Gray Model -------------------------------------------    
    def grayModel(self):
        #T=np.interp(np.log(self.p),np.log(params['Tprof'][0]),params['Tprof'][1])
        T=self.Teq*np.ones_like(self.p, dtype=self.numerical_precision)
        
        extinctCoef,absorbCoef, scatCoef = self.calcOpacities(self.modelSetting,self.params,T)
        rho = np.tile((self.ntot*self.MuAve)[:,np.newaxis],(1,self.nWave))
        mExtinctCoef = extinctCoef/rho
        
        dtau = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        dtau[0,:] = (mExtinctCoef[0,:]*self.p[0])/self.grav[0]
        for i in range(1,self.nLay):
            dtau[i,:] = 0.5*(mExtinctCoef[i,:] + mExtinctCoef[i-1,:])*(self.p[i]-self.p[i-1])/self.grav[i]
        taugrid = np.average(np.cumsum(dtau,axis=0),axis=1)
        
        # From semi-Gray Model
        T = ((3.0/4.0)*self.Teq**4*(taugrid+(0.7104-0.1331*np.exp(-3.4488*taugrid))))**(1.0/4.0)
        return T
    


    
    
    ### Lucy Unsold ------------------------------------------
    def TpCorrection(self,T,modelSetting,params,B,J,K,H,mExtinctCoef,mAbsorbCoef,errordH = 5e-3,errordT = 1e-3,stop = 250):
        
        TList = T
        dTList = np.zeros([self.nLay], dtype=self.numerical_precision)
        dHList = np.zeros([self.nLay], dtype=self.numerical_precision)
        #RTENList = np.zeros([self.nLay])
        #HC = sigmaSB*self.Teq**4/(4.0*np.pi)
        HC = sigmaSB*self.params['Tint']**4/(4.0*np.pi)

        
        stefan = True
        i = 0
        while(stefan):
            nNonConvergedLay = 0
            dT,dH = self.calcDT(T,B,J,K,H,HC,self.f,mExtinctCoef,mAbsorbCoef)
            T = T+dT
                        
            #RTENorm = -np.trapz((self.MassAbsCoef)*(self.J-self.B),self.fGrid)/HC
            TList = np.c_[TList,T]
            dTList = np.c_[dTList,(dT/(T-dT))]
            dHList = np.c_[dHList,dH/HC]
            #RTENList = np.c_[RTENList,RTENorm]
            
            for d in range(self.nLay):
                if np.abs(dH[d]/HC) < errordH and np.abs(dT[d]/(T[d]-dT[d])) < errordT:  
                    nNonConvergedLay += 0
                else: 
                    nNonConvergedLay += 1
                #print d, " ", e1,e2
            if nNonConvergedLay == 0:
                stefan = False
                print('Successfully converged')
                
            if stefan:
                self.qmol_lay                                                                         = self.calcComposition(modelSetting,params,T)
                self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase,self.r = self.calcHydroEqui(modelSetting,params,T)
                B,J,K,H,mExtinctCoef,mAbsorbCoef= self.solveRTE(T,modelSetting,params,TpCorrLucy = True)
                i+=1
                print(i,)

            if i == stop:
                print('Reached max # of iterations')
                return TList,dTList[:,1:],dHList[:,1:]#,RTENList[:,1:],PList
            
        return TList[:,:-1],dTList[:,1:],dHList[:,1:] #RTENList[:,1:]
    
    def calcDT(self,T,B,J,K,H,HC,f,extCoef,absCoef):
        l = self.nLay
        #f = f[::-1]
        for d in range(1,l-1):
            H[d,:] = (H[d,:] + H[d+1,:])/2.0
        
        KJTop = -np.trapz(absCoef*J,f,axis=1)
        Jinteg = -np.trapz(J,f,axis=1)
        KJ = KJTop/Jinteg
        
        KPTop = -np.trapz(absCoef*B,f,axis=1)
        Binteg = -np.trapz(B,f,axis=1)
        KP = KPTop/Binteg
        
        XFTop = -np.trapz(extCoef*H,f,axis=1)
        Hinteg = -np.trapz(H,f,axis=1)
        XF = XFTop/Hinteg
        
        dH = (HC - Hinteg)
        #factor = 30.0   #dT too large
        
        dT = np.zeros([l], dtype=self.numerical_precision)
        for d in range(l):
            
            if d == 0:
                integral0 = XF[d]*dH[d]*self.p[d]/(self.grav[d])
                integral = integral0
            else:
                integral = np.trapz(y=(XF[:d+1]*dH[:d+1]/self.grav[d]),x=self.p[:d+1]) + integral0
            #print integral
            # Pgrid update
            #dB = KJ[d]/KP[d]* (Jinteg[d] + 3.0*integral + 2.0*dH[0]) - Binteg[d]
            #Taugrid update
            dB = (Jinteg[d]*KJ[d] - Binteg[d]*KP[d])/self.grav[d] + KJ[d]/KP[d]* (3.0*integral + 2.0*dH[0])
            
            dT[d] = (np.pi/(4.0*sigmaSB*T[d]**3)*dB)#/factor
            
            while np.abs(dT[d]/T[d]) > 0.02:  # restrain to +- 5% change in temp
                dT[d] = dT[d]/5
        return dT,dH

    
    ### Linearized -----------------------------------------------
    def correcT(self,T,plnk,J,fn,Gfn,Hext,dtau,lmbda,omega,chi,kappa,weights,loop,convection=True):
        '''
        mExtinctCoef = X/density!!!! = omega w in Mihalas    = omega = chi/rho
        extinctCoef = X ---> not div by rho!                 = chi
        absCoef = kappa                                      = kappa
        lmbda = k/X = epsilon E in Mihalas
        Stefan must always equal True
        
        Outputs: dT, dJ for next iteration
        '''
        # calculate all the necessary derivatives at the current T, P
        dchi_dT,dkappa_dT,domega_dT,dlambda_dT,deta_dT,dplnk_dT = self.diffVar(T,plnk,lmbda,omega,chi,kappa,loop)
        
                ################################### TRANSFER EQUATION  ################################
        # solve the radiative transfer equation
        ######################## Top of atmosphere #########################        
        U = np.zeros([self.nLay,self.nLay,self.nWave], dtype=self.numerical_precision)
        V = np.zeros([self.nLay,self.nLay,self.nWave], dtype=self.numerical_precision)
        E = np.zeros([self.nLay,self.nWave], dtype=self.numerical_precision)
        #    Top Layer
        # at the top layer A = 0 and so we only have B and C
        # B              equation 18.64a
        U[0,0,:] = fn[0,:]/dtau[1,:] + Gfn + dtau[1,:] * lmbda[0,:] / 2.0  # B at top
        # C               equation 18.64b
        U[0,1,:] = fn[1,:]/dtau[1,:] # C at top.  #U[0,1,:] = -1.0*fn[1,:]/dtau[1,:] # C at top
        # B               equation 18.65a
        V[0,0,:] = ((-1.0*(fn[0,:]*J[0,:] - fn[1,:]*J[1,:])/(dtau[1,:]**2)) + 0.5*lmbda[0,:]*(J[0,:] - plnk[0,:]))*(dtau[1,:]*domega_dT[0,:]/(omega[0,:] + 
                     omega[1,:])) + (0.5*dtau[1,:])*(dlambda_dT[0,:]*J[0,:] - lmbda[0,:]*plnk[0,:]*(deta_dT[0,:]/(kappa[0,:]*plnk[0,:]) + dchi_dT[0,:]/chi[0,:]))
        # C                equation 18.65b
        V[0,1,:] = -1.0*((fn[0,:]*J[0,:] - fn[1,:]*J[1,:])/(dtau[1,:]**2))*(dtau[1,:]*domega_dT[1,:]/(omega[0,:] + omega[1,:]))
        # L in Mihalas        equation 18.66
        E[0,:] = (-1.0*(fn[0,:]*J[0,:] - fn[1,:]*J[1,:])/dtau[1,:]) - Gfn*J[0,:] + Hext - (0.5*dtau[1,:]*lmbda[0,:]*(J[0,:] - plnk[0,:]))
        
        ######################## Middle of atmosphere #########################
        mid_lyrs = np.arange(self.nLay-2)+1 # all layers excepts the top and bottom ones
        # A                equation  18.68a
        U[mid_lyrs,mid_lyrs-1,:] = -1.0*(fn[mid_lyrs-1,:]/(dtau[mid_lyrs,:]*0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:])))
        # B                equation 18.68b
        U[mid_lyrs,mid_lyrs,:] = (fn[mid_lyrs,:]/(0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:])))*(1.0/dtau[mid_lyrs,:] + 1.0/dtau[mid_lyrs+1,:]) + lmbda[mid_lyrs,:]
        # C                equation 18.68c
        U[mid_lyrs,mid_lyrs+1,:] = -1.0*(fn[mid_lyrs+1,:]/(dtau[mid_lyrs+1,:]*0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:])))
        # now for V we need to compute equations 18.70a-e
        alpha_di = (fn[mid_lyrs,:]*J[mid_lyrs,:] - fn[mid_lyrs-1,:]*J[mid_lyrs-1,:])/(dtau[mid_lyrs,:]*0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:]))
        gamma_di = (fn[mid_lyrs,:]*J[mid_lyrs,:] - fn[mid_lyrs+1,:]*J[mid_lyrs+1,:])/(dtau[mid_lyrs+1,:]*0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:]))
        beta_di = alpha_di + gamma_di
        a_di = (alpha_di + 0.5*beta_di*dtau[mid_lyrs,:]/(0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:])))/(omega[mid_lyrs-1,:]+omega[mid_lyrs,:])
        c_di = (gamma_di + 0.5*beta_di*dtau[mid_lyrs+1,:]/(0.5*(dtau[mid_lyrs,:]+dtau[mid_lyrs+1,:])))/(omega[mid_lyrs+1,:]+omega[mid_lyrs,:])
        # A                 equation 18.69a
        V[mid_lyrs,mid_lyrs-1,:] = -1.0*(a_di*domega_dT[mid_lyrs-1,:])
        # B                equation 18.69b
        V[mid_lyrs,mid_lyrs,:] = -1.0*(a_di+c_di)*domega_dT[mid_lyrs,:] + dlambda_dT[mid_lyrs,:]*J[mid_lyrs,:] - lmbda[mid_lyrs,:]*plnk[mid_lyrs,:]*(deta_dT[mid_lyrs,:]/(kappa[mid_lyrs,:]*plnk[mid_lyrs,:]) - dchi_dT[mid_lyrs,:]/chi[mid_lyrs,:] )
        # C                equation 18.69c
        V[mid_lyrs,mid_lyrs+1,:] = -1.0*(c_di*domega_dT[mid_lyrs+1,:])
        # L                equation 18.71
        E[mid_lyrs,:] = -1.0*beta_di - lmbda[mid_lyrs,:]*(J[mid_lyrs,:] - plnk[mid_lyrs,:])
        
        ###################### Bottom of atmosphere
        # A                equation 18.72b
        U[-1,-2,:] = -1.0*fn[-2,:]/dtau[-1,:]
        # B                equation 18.72a
        U[-1,-1,:] = fn[-1,:]/dtau[-1,:] + 0.5 + 0.5*dtau[-1,:]*lmbda[-1,:]
        # for V first calculate b_i (equation 18.73)
        b_i = (1.0/3.0)*(plnk[-1,:]-plnk[-2,:])/(dtau[-1,:]**2)
        # A                equation 18.72d
        V[-1,-2,:] = -1.0*((((fn[-1,:]*J[-1,:] - fn[-2,:]*J[-2,:])/(dtau[-1,:]**2) - b_i)*dtau[-1,:]*domega_dT[-2,:]/(omega[-1,:]+omega[-2,:])) - dplnk_dT[-2,:]/(3.0*dtau[-1,:]))
        # B                equation 18.72c
        V[-1,-1,:] = ((-1.0*((fn[-1,:]*J[-1,:] - fn[-2,:]*J[-2,:])/(dtau[-1,:]**2))) + b_i + 0.5*lmbda[-1,:]*(J[-1,:]-plnk[-1,:]))*(dtau[-1,:]*domega_dT[-1,:]/(omega[-1,:]+omega[-2,:])) + 0.5*dtau[-1,:]*(dlambda_dT[-1,:]*J[-1,:] - lmbda[-1,:]*plnk[-1,:]*(deta_dT[-1,:]/(kappa[-1,:]*plnk[-1,:]) - dchi_dT[-1,:]/chi[-1,:])) - dplnk_dT[-1,:]*(0.5 + 1.0/(3.0*dtau[-1,:]))                         
        # L                 equation 18.74
        E[-1,:] = -1.0*((fn[-1,:]*J[-1,:] - fn[-2,:]*J[-2,:])/dtau[-1,:]) - 0.5*(J[-1,:] - plnk[-1,:]) + (plnk[-1,:] - plnk[-2,:])/(3.0*dtau[-1,:]) - 0.5*dtau[-1,:]*lmbda[-1,:]*(J[-1,:] - plnk[-1,:])
        
        ################################## RADIATIVE EQUILIBRIUM EQUATION ###############################
        X = np.zeros([self.nLay,self.nLay,self.nWave], dtype=self.numerical_precision)
        A = np.zeros([self.nLay,self.nLay], dtype=self.numerical_precision)
        F = np.zeros([self.nLay], dtype=self.numerical_precision)
        # split atmosphere in differential and integral forms
        # need to do this for numerical stability (integral form breaks at BOA while differential form breaks at TOA)
        # differential form when tau > 1, integral when tau < 1
        # according to Gandhi & Madhusudhan 2017, tau = 1 is around the 1 bar level
        #########   note that tau = 1 as the transition broke the test code, tau = 10 was better
        diff_bot = np.where(self.p >= 1e5)[0] # differential form layers
        #diff_bot = np.arange(self.nLay)[-2:]
        int_top = np.where(self.p < 1e5)[0] # integral form layers
        #int_top = np.arange(self.nLay)[:-2]
        #self.transition = diff_bot.min() # transition layer between integral and differential forms
        
        # integral form at top half of atmosphere 
        #         B             equation 18.78a
        X[int_top,int_top,:] = weights*kappa[int_top,:]
        # B             equation 18.78b - note that we think there should be a sum here
        #        A[int_top,int_top] = sign_tracker*np.trapz(dkappa_dT[int_top,:]*J[int_top,:] - deta_dT[int_top,:], self.freqs)
        A[int_top,int_top] = np.sum((dkappa_dT[int_top,:]*J[int_top,:] - deta_dT[int_top,:])*weights, axis=1)
        # L             equation 18.78c
        #        F[int_top] = -1.0*sign_tracker*np.trapz(kappa[int_top,:]*(J[int_top,:] - plnk[int_top,:]), self.freqs)
        F[int_top] = -1.0*np.sum(weights*kappa[int_top,:]*(J[int_top,:] - plnk[int_top,:]), axis=1)
            
        # Differential form at bottom part of the atmosphere
        # A              equation 18.79a
        X[diff_bot,diff_bot-1,:] = -1.0*weights*fn[diff_bot-1,:]/dtau[diff_bot,:]
        # B              equation 18.79b
        X[diff_bot,diff_bot,:] = weights*fn[diff_bot,:]/dtau[diff_bot,:]
        # A              equation 18.79c
        #        A[diff_bot,diff_bot-1] = -1.0*sign_tracker*np.trapz(((fn[diff_bot,:]*J[diff_bot,:] - fn[diff_bot-1,:]*J[diff_bot-1,:])/(dtau[diff_bot,:]**2))*(domega_dT[diff_bot-1,:]*dtau[diff_bot,:]/(omega[diff_bot,:]+omega[diff_bot-1,:])),self.freqs)
        A[diff_bot,diff_bot-1] = -1.0*np.sum(weights*((fn[diff_bot,:]*J[diff_bot,:] - fn[diff_bot-1,:]*J[diff_bot-1,:])/(dtau[diff_bot,:]**2))*(domega_dT[diff_bot-1,:]*dtau[diff_bot,:]/(omega[diff_bot,:]+omega[diff_bot-1,:])),axis=1)
        # B              equation 18.79d
        A[diff_bot,diff_bot] = -1.0*np.sum(weights*((fn[diff_bot,:]*J[diff_bot,:] - fn[diff_bot-1,:]*J[diff_bot-1,:])/(dtau[diff_bot,:]**2))*(domega_dT[diff_bot,:]*dtau[diff_bot,:]/(omega[diff_bot,:]+omega[diff_bot-1,:])),axis=1)
        # L              equation 18.80
        F[diff_bot] = (sigmaSB*self.params['Tint']**4)/(4.0*np.pi) - np.sum(weights*(fn[diff_bot,:]*J[diff_bot,:] - fn[diff_bot-1,:]*J[diff_bot-1,:])/dtau[diff_bot,:],axis=1)
        
        # top boundary condition of differential method - generally not applicable for combined form since differential form is only used at BOA
        top = np.where(diff_bot==0)[0]
        # B              equation 18.81a
        #        X[top,top,:] = Gfn
        X[top,top,:] = weights*Gfn
        # L              equation 18.81b
        #F[top] = (sigmaSB*self.Teq**4)/(4.0*np.pi) - np.sum(weights*(Gfn*J[0,:] - Hext))
        F[top] = (sigmaSB*self.params['Tint']**4)/(4.0*np.pi) - np.sum(weights*(Gfn*J[0,:] - Hext))
        
        # now we add convection to the layers that satisfy Delt > 2/7 = Delt_ad
        Q = np.zeros([self.nLay,self.nLay], dtype=self.numerical_precision)
        R = np.zeros([self.nLay,self.nLay], dtype=self.numerical_precision)
            #A2 = deepcopy(A)

        #print 'convection: ', convection, '   loop: ', loop, '--> ', (convection and loop == 1)
        if convection and loop == 1:
#            ################## STEFAN
#            # calculate Logarithmic gradient of temperature (convection stability criteria) at every layer   Eq. 18.87 Hubeny & Mihalas 2014
#            # convection turns on when Grad > Grad_ad, where Grad_ad is the adiabatic gradient (2/7 for ideal gas)
#            Grad = np.zeros(self.nLay)
#            for lyr in np.arange(self.nLay-1)+1:
#                Grad[lyr] = ((T[lyr] - T[lyr-1])/(T[lyr] + T[lyr-1])) *((self.p[lyr] + self.p[lyr-1])/(self.p[lyr] - self.p[lyr-1]))
#            # top of atmosphere is ill defined, set it as second layer (shouldn't matter since there is generally no convection at the TOA)
#            Grad[0] = Grad[1]
#    #        self.Grad = Grad
#            # adiabatic gradient
#            self.Grad_ad = 0.15#2/7. # value used in GENESIS paper
#            
#            
#            # calculate T, p at middle of layers for convection
#            p_mid = np.zeros(self.nLay)
#            T_mid = np.zeros(self.nLay)
#            p_mid[0] = self.p[0]
#            T_mid[0] = T[0]
#            for lyr in np.arange(self.nLay-1)+1:
#                p_mid[lyr] = 0.5*(self.p[lyr] + self.p[lyr-1])
#                T_mid[lyr] = 0.5*(T[lyr] + T[lyr-1])
#        
#            # Get H_conv (F_conv/4pi) at half layers (page 641 Hubeny & Mihalas 2014) and its partial derivatives
#            Rho = self.ntot*self.MuAve 
#            self.conv_lyrs = np.where(Grad >= self.Grad_ad)[0]
#            print self.conv_lyrs
#            H_conv, Grad_minus_Grad_el = self.H_convection(T_mid, p_mid, Grad,chi)
#            dH_conv_dT, dH_conv_dp = self.H_conv_Deriv(T_mid, p_mid, Grad, H_conv,chi)
#            conv_lyrs = np.where(Grad >= self.Grad_ad)[0]
#            #print 'conv_lyrs:', conv_lyrs
#            # add convective flux to all layers where convection applies
#            # note that for each layer, H and its derivatives are calculated at layer - 1/2
#            for lyr in conv_lyrs:
#                # if layer uses differential form
#                if np.where(diff_bot == lyr)[0].size == 1:
#                    # A                    equation 18.88b
#                    A[lyr,lyr-1] += (-1.0)*(-0.5*(dH_conv_dT[lyr]) - 0.5*(dH_conv_dp[lyr])*self.p[lyr-1]/T[lyr-1] )
#                    # B                    equation 18.88d
#                    A[lyr,lyr] += 0.5*(dH_conv_dT[lyr]) + 0.5*(dH_conv_dp[lyr])*self.p[lyr]/T[lyr]
#                    # B                    equation 18.88e
#                    Q[lyr,lyr] = 1.5*H_conv[lyr]/(Grad_minus_Grad_el[lyr])
#                    # L                    equation 18.88f
#                    F[lyr] += -1.0*H_conv[lyr]
#                # if layer uses integral form
#                if np.where(int_top == lyr)[0].size == 1:
#                    # A                    equation 18.90b
#                    A[lyr,lyr-1] += -1.0*(dH_conv_dT[lyr] + dH_conv_dp[lyr]*self.p[lyr-1]/T[lyr-1])*Rho[lyr]                                                /(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])
#                    # A                    equation 18.90c
#                    Q[lyr,lyr-1] = -1.0*(1.5*H_conv[lyr]/(Grad_minus_Grad_el[lyr]))*Rho[lyr]                                                                /(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])
#                    # B                    equation 18.90e
#                    A[lyr,lyr] += (dH_conv_dT[lyr+1] + dH_conv_dp[lyr+1]*self.p[lyr]/T[lyr] - dH_conv_dT[lyr] - dH_conv_dp[lyr]*self.p[lyr]/T[lyr])*Rho[lyr]/(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])    
#                    # B                    equation 18.90g
#                    Q[lyr,lyr] = (H_conv[lyr+1]/Grad_minus_Grad_el[lyr+1] - H_conv[lyr]/Grad_minus_Grad_el[lyr])*3.0*Rho[lyr]                               /(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])
#                    '''FIX'''#              equation 18.90i
#                    A[lyr,lyr+1] += -1.0*-1.0*(dH_conv_dT[lyr+1] + dH_conv_dp[lyr+1]*self.p[lyr]/T[lyr])*Rho[lyr]                                           /(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])    
#                    # C                    equation 18.90j
#                    Q[lyr,lyr+1] =  (H_conv[lyr+1]/Grad_minus_Grad_el[lyr+1])*3.0*Rho[lyr]                                                                  /(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])
#                    # L                    equation 18.90k
#                    F[lyr] += -1.0*(H_conv[lyr+1] - H_conv[lyr])*2.0*Rho[lyr]                                                                               /(self.p[lyr+1]/self.grav[lyr+1]-self.p[lyr-1]/self.grav[lyr-1])
#                
#                #### Convection Equation
#                # A                        equation 18.91b  
#                R[lyr,lyr-1] = 2.0*T[lyr]*Grad[lyr]/(T[lyr]**2 - T[lyr-1]**2) - (2.0*self.p[lyr]*Grad[lyr]/(self.p[lyr]**2 - self.p[lyr-1]**2))*(self.p[lyr-1]/T[lyr-1])
#                # B                        equation 18.91d
#                R[lyr,lyr] = 2.0*T[lyr-1]*Grad[lyr]/(T[lyr]**2 - T[lyr-1]**2) - (2.0*self.p[lyr-1]*Grad[lyr]/(self.p[lyr]**2 - self.p[lyr-1]**2))*(self.p[lyr]/T[lyr])
#                # L                        equation 18.92
#                Should_Be_Zero = Grad[lyr] - ((T[lyr] - T[lyr-1])/(T[lyr] + T[lyr-1])) * ((self.p[lyr]+self.p[lyr-1])/(self.p[lyr]-self.p[lyr-1]))
#                if Should_Be_Zero != 0.0:
#                    print 'Warning, Convection equation is fishy'
#                F[lyr] += Should_Be_Zero
#            
            
            print('CONVECTION')
            
            ################# JONATHAN
            grad = np.zeros([self.nLay], dtype=self.numerical_precision)
            # not lnT!!!
            
#            grad = np.zeros([self.nLay])
#            lnT = np.log(T)
#            lnP = np.log(self.p)
#            for i in range(1,self.nLay):
#                grad[i]=(((lnT[i]-lnT[i-1])/(lnT[i]+lnT[i-1]))*
#                         ((lnP[i]+lnP[i-1])/(lnP[i]-lnP[i-1])))
#            grad[0] = grad[1]
            
            grad = np.zeros([self.nLay], dtype=self.numerical_precision)
            grad[1:] = np.diff(np.log(T))/np.diff(np.log(self.p))
            grad[0] = grad[1]
            
#            grad_ad = 0.15
            grad_ad = 2./7. # Genesis
            convLay = np.where(grad >= grad_ad)[0]
#            if self.first_conv_run == True:
#                self.first_conv_lay = deepcopy(convLay)
#                self.first_conv_run = False
#            if force_conv_lay:
#                convLay = deepcopy(self.first_conv_lay)
            self.convLay = deepcopy(convLay)

                        
            Tmid = np.hstack([T[0],0.5*(T[:self.nLay-1]+T[1:self.nLay])])
            Pmid = np.hstack([self.p[0],0.5*(self.p[:self.nLay-1]+self.p[1:self.nLay])])
            
           # rho = self.ntot*self.MuAve
            
            Hconv,grad_diff,rho = self.getHconv(Tmid,Pmid,grad,grad_ad,chi)
            dHconv_T,dHconv_P = self.getdHconv(Tmid,Pmid,grad,grad_ad,Hconv,chi)

            # Remove nans
            grad_diff[np.isnan(grad_diff)] = 0
            Hconv[np.isnan(Hconv)] = 0
            dHconv_T[np.isnan(dHconv_T)] = 0
            dHconv_P[np.isnan(dHconv_P)] = 0
            
            for d in convLay:
                # If layer uses differential form
                if np.where(diff_bot == d)[0].size == 1:
                    A[d,d-1] += -(-(0.5*dHconv_T[d])-(0.5*dHconv_P[d])*(self.p[d-1]/T[d-1])) #A
                    A[d,d]   +=    (0.5*dHconv_T[d])+(0.5*dHconv_P[d])*(self.p[d]/T[d])      #B
                   
                    Q[d,d] = (3./2. * Hconv[d]) / (grad_diff[d])
                   
                    F[d] += -(Hconv[d])
                
                #if layer uses integral form
                if np.where(int_top == d)[0].size == 1:
                    A[d,d-1] += -((dHconv_T[d]+dHconv_P[d]*(self.p[d-1]/T[d-1]))*0.5*rho[d]/           (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d]))
                    A[d,d]   += (((dHconv_T[d+1]+dHconv_P[d+1]*(self.p[d]/T[d])) - 
                                   (dHconv_T[d-1]+dHconv_P[d-1]*(self.p[d]/T[d])))*0.5*rho[d]/          (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d]))
                    A[d,d+1] += ((dHconv_T[d+1]+dHconv_P[d+1]*(self.p[d-1]/T[d-1]))*0.5*rho[d]/           (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d]))
                    
                    
                    if grad_diff[d+1] == 0: # Do not allow division by 0
                        Q[d,d-1]  = -((Hconv[d]/grad_diff[d])*(1.5*rho[d])/                            (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d]))
                        Q[d,d]    =  ((0-Hconv[d]/grad_diff[d])*(3.*rho[d]/     (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d])))
                        Q[d,d+1]  = 0
                    else:
                        Q[d,d-1]  = -((Hconv[d]/grad_diff[d])*(1.5*rho[d])/                            (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d]))
                        Q[d,d]    =  ((Hconv[d+1]/grad_diff[d+1]-Hconv[d]/grad_diff[d])*(3.*rho[d]/     (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d])))
                        Q[d,d+1]  = -((Hconv[d+1]/(grad_diff[d+1]))*(3.*rho[d]/      (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d])))
                        
                    F[d] += -(Hconv[d+1]-Hconv[d]*2.)*rho[d]/                                      (self.p[d+1]/self.grav[d]-self.p[d-1]/self.grav[d])
                    
                R[d,d-1] = ((2*T[d]*grad[d])/(T[d]**2-T[d-1]**2) - 
                                (2*self.p[d]*grad[d])/(self.p[d]**2-self.p[d-1]**2)*(self.p[d-1]/T[d-1]))
                   
                R[d,d]   = ((2*T[d-1]*grad[d])/(T[d]**2-T[d-1]**2) -
                                (2*self.p[d-1]*grad[d])/(self.p[d]**2-self.p[d-1]**2)*(self.p[d]/T[d]))
        
        ## Now we solve for delta T  ---  equation 36  Gandhi & Madhusudhan 2017
        U_moved = np.moveaxis(U,-1,0)
        U_inv_moved = np.linalg.inv(U_moved)
        V_moved = np.ascontiguousarray(np.moveaxis(V,-1,0))
        UV_moved = np.matmul(U_inv_moved,V_moved)
        UV = np.moveaxis(UV_moved,0,-1)
        X_moved = np.ascontiguousarray(np.moveaxis(X,-1,0))
        XUV = np.matmul(X_moved, UV_moved)
        XUV_summed = np.sum(XUV,0)
#            XUV_summed = sign_tracker*np.trapz(XUV,self.freqs,axis=0)
        A_conv = A + np.matmul(Q,R)
#        A_conv = A2 + np.matmul(Q,R)
        LHS = A_conv - XUV_summed
        U_inv = np.moveaxis(U_inv_moved,0,-1)
#        UE = np.zeros_like(E)
#        for i in range(self.NF):
#            UE[:,i] = np.dot(U_inv[:,:,i],E[:,i])
#        XUE = np.zeros_like(E)
#
#        for i in range(self.NF):
#            XUE[:,i] = np.dot(X[:,:,i],UE[:,i])
#        XUE_summed = np.sum(XUE,1)
        UE = np.einsum('ijk,jk->ik', U_inv, E)
        XUE = np.einsum('ijk,jk->ik', X, UE)
        XUE_summed = np.sum(XUE, axis=1)

#            XUE_summed = sign_tracker*np.trapz(XUE,self.freqs, axis=-1)
        RHS = F - XUE_summed
        deltaT = np.dot(np.linalg.inv(LHS),RHS)
        
        dT = np.tile(deltaT[:,np.newaxis],(1,self.nWave))
#        UVT = np.zeros_like(E)
#        for i in range(self.NF):
#            UVT[:,i] = np.dot(UV[:,:,i],dT[:,i])
        UVT = np.einsum('ijk,jk->ik', UV, dT)
        
        deltaJ = UE - UVT
#        
#        if 1:
#            
#            XUV = np.zeros_like(A)
#            for i in range(self.NF):
#                UV = np.matmul(inv(U[:,:,i]),V[:,:,i])
#                XUV += np.matmul(X[:,:,i],UV)
#            LHS = inv(A-XUV)
#            
#            XUE = np.zeros_like(F)
#            for i in range(self.NF):
#                UE = np.dot(inv(U[:,:,i]),E[:,i])
#                XUE += np.dot(X[:,:,i],UE)
#            RHS = F - XUE
#            
#            deltaT = np.dot(LHS,RHS)
#                
#                
#
        return deltaT, deltaJ
    
    
    def diffVar(self,T,plnk,lmbda,mExtinctCoef,extinctCoef,absorbCoef,loop):
        newT = T*1.01
        diff = newT-T
        diff = np.tile(diff[:,np.newaxis],(1,self.nWave))

        if loop == 1: # Add stability by calculation chemistry/hydroequil only after 1st convergence
            if self.verbose: print('Computing chemistry in calcNonGrayTpProf/correcT/diffVar()\n')
            self.qmol_lay                                                                         = self.calcComposition(self.modelSetting,self.params,newT)
            self.z,self.dz,self.grav,self.ntot,self.nmol,self.MuAve,self.scaleHeight,self.RpBase, self.r  = self.calcHydroEqui(self.modelSetting,self.params,newT)
        extinctCoef2,absorbCoef2, scatCoef                                                    = self.calcOpacities(self.modelSetting,self.params,newT)
        rho = np.tile((self.ntot*self.MuAve)[:,np.newaxis],(1,self.nWave))
        
        mExtinctCoef2 = extinctCoef2/rho

        lmbda2 = absorbCoef2/extinctCoef2

        B2 = rad.PlanckFct(np.tile(newT[:,np.newaxis],(1,self.nWave)),np.tile(self.f[np.newaxis,:],(len(newT),1)),'Hz','W/(m**2*Hz)','rad')
        
        diffExt = (extinctCoef2 - extinctCoef)/diff
        diffAbs = (absorbCoef2 - absorbCoef)/diff
        diffMExt = (mExtinctCoef2 - mExtinctCoef)/diff
        diffLmbda = (lmbda2 - lmbda)/diff
        diffN = (absorbCoef2*B2 - absorbCoef*plnk)/diff
        diffPlnk = (B2 - plnk)/diff

        return diffExt,diffAbs,diffMExt,diffLmbda,diffN,diffPlnk


    def calcWeight(self):
        weight = np.zeros([self.nWave], dtype=self.numerical_precision)
        for i in range(1,self.nWave-1):
            weight[i] = 0.5*(self.f[i+1]-self.f[i-1])
        
        weight[0] = 0.5*(self.f[1]-self.f[0])
        weight[-1] = 0.5*(self.f[-1]-self.f[-2])
        
        if self.f[0] > self.f[-1]:
            return weight * -1.0             #FLIPPED FREQ GRID SO -VE 
        else:
            return weight
        
    
    def H_convection(self, T,p, Grad,chi):
        Q = 1.0 # dln(rho)/dln(T) holding pressure fixed, equal to unity for an ideal gas
        rho = self.ntot*self.MuAve # density [kg/m^3]
        Hp = p/(self.grav*rho) # scale height assuming ideal gas
        massLay = rho * self.z
        heatCap = 20.79/1.01e-3 * massLay
        heatCap[-1] = heatCap[-2]
#        c_p = heatCap#(7./2)*(12.5*2.0) # heat capacity at constant pressure
        c_p = 1000#(7./2)*(12.5*2.0) # heat capacity at constant pressure
        mixing_length = Hp
        dplnk_dT = self.Planck_deriv(T)
        tau_el = np.zeros(self.nLay, dtype=self.numerical_precision)
        for i in range(self.nLay):
            tau_el[i] = self.ChiRoss_not_over_rho(dplnk_dT[i,:],chi[i])*mixing_length[i]
        Beta =  ((16.0*(2.0**0.5)*sigmaSB*(T**3)) / (rho*c_p*((self.grav*Q*Hp)**0.5)*(mixing_length/Hp) )) * (tau_el/(1.0 + 0.5*tau_el*tau_el))
        # equation 18.38 Hubeny & Mihalas
        Grad_minus_Grad_el = Grad - self.Grad_ad + 0.5*Beta*Beta - Beta*(0.25*Beta*Beta + Grad - self.Grad_ad)**0.5
        F_conv = ((self.grav * Q * Hp/32.)**0.5)*(rho*c_p*T)*((Grad_minus_Grad_el)**1.5)*((mixing_length/Hp)**2)
        H_conv = F_conv/(4.0*np.pi)
        return H_conv, Grad_minus_Grad_el
    
    
    def H_conv_Deriv(self, T, p, Grad, H_conv,chi):
        
        dT = T*1e-6
        dp = p*1e-6
        
        chiNew = self.calcOpacities(self.modelSetting,self.params,T+dT)[0]

        H_conv_T, dif_Grad_T = self.H_convection(T + dT, p, Grad,chiNew)
        H_conv_p, dif_Grad_p = self.H_convection(T, p + dp, Grad,chi)

        dH_dT = (H_conv_T - H_conv)/dT
        dH_dp = (H_conv_p - H_conv)/dp
        
        return dH_dT, dH_dp
    
    def getHconv(self,T,P,grad,grad_ad,chi):
        z,dz,grav,ntot,nmol,MuAve,scaleHeight,RpBase,r  = self.calcHydroEqui(self.modelSetting,self.params,T)
        
        rho = ntot*MuAve
        Hp = scaleHeight #P/(self.grav*rho) #!!!!!!!????? 
        Q = 1. # Genesis
        sizel = Hp # Genesis

        MuAveLay = np.hstack([MuAve[0],0.5*(self.MuAve[:self.nLay-1]+self.MuAve[1:self.nLay])])
        Rconst = 8.3144598 /(MuAveLay/uAtom * 0.001)#J/mol*K / kg/mol = J/kg*K
        Cp = 7./2. * Rconst # 5./2.*R Monotomic, 7./2.*R #Diatomic, 4*R #Polyatomic   units of J/kg K

#        Cp = 14300
#        Cp = 100*1e3   # Testing unphysical Cp
        dplnk_dT = self.Planck_deriv(T)
        tau_el=np.zeros([self.nLay], dtype=self.numerical_precision)
        for i in range(self.nLay):
            tau_el[i] = self.ChiRoss_not_over_rho(dplnk_dT[i,:],chi[i])*sizel[i]

        beta = ((16.*np.sqrt(2.)*sigmaSB*T**3.)/(rho*Cp*(self.grav*Q*Hp)**0.5*(sizel/Hp)))*(tau_el/(1.+0.5*tau_el**2.))
        grad_diff = (grad-grad_ad) + (0.5*beta**2. - beta*(0.25*beta**2.+(grad-grad_ad))**0.5)
        Fconv = (self.grav*Q*Hp/32.)**0.5*(rho*Cp*T)*(grad_diff)**1.5*(sizel/Hp)**2.
        Hconv = Fconv/(4.*np.pi)
#        where_nan = np.isnan(Hconv)
#        Hconv[where_nan] = 0
        return Hconv,grad_diff,rho

    def getdHconv(self,T,P,grad,grad_ad,Hconv,chi):
        dT = T*0.0001
        dp = P*0.0001
        
        chiNew = self.calcOpacities(self.modelSetting,self.params,T+dT)[0]
        Hconv_T, grad_diff,rho = self.getHconv(T + dT,P,grad,grad_ad,chiNew)
        Hconv_P, grad_diff,rho = self.getHconv(T,P + dp,grad,grad_ad,chi)
    
        dHconv_T = (Hconv_T - Hconv)/dT
        dHconv_P = (Hconv_P - Hconv)/dp
        return dHconv_T,dHconv_P
    
    def Planck_deriv(self,T):
       diff = 5
       newT = T + diff
#       newT = T*1.01
       
       plnk = rad.PlanckFct(np.tile(T[:,np.newaxis],(1,self.nWave)),np.tile(self.f[np.newaxis,:],(len(T),1)),'Hz','W/(m**2*Hz)','rad')

       B2 = rad.PlanckFct(np.tile(newT[:,np.newaxis],(1,self.nWave)),np.tile(self.f[np.newaxis,:],(len(newT),1)),'Hz','W/(m**2*Hz)','rad')
       
       diffPlnk = (B2 - plnk)/diff

       return diffPlnk
   

    def ChiRoss_not_over_rho(self,planck_d,chi_nu):
       bottom = np.trapz(planck_d, self.f)
       top = np.trapz(planck_d/chi_nu, self.f)
       ross = bottom/top
       return ross
   
        

    
      
    #%% Saving/Loading Methods

    def save(self,filename=None):
        # Saves self to pickle
        if filename is None:
            filename=self.filebase+self.runName+'.atm'
        
        doNotSaveKeys=np.array(['mlab','thermalSecEclppm','albedoSecEclppm','secEclppm','extinctCoef','absorbCoef','scatCoef',
                                'LookUpSigma','AbunLUT','ciaH2LUT','ciaHeLUT','sigmaAtm','mieLUT',
                                'g1','g2','g3','g4','omega','lmbda','gamma','Fs','u0','solution','specs','taugrid']) #multiScatToon variables
        
        
        #Copy all variables (except the ones not to save) in a dictionary called obj
        obj=dict()
        for key in self.__dict__.keys():
            if np.any(key==doNotSaveKeys)==False:
                try:
                    obj[key]=self.__dict__[key]                                                                               
                    if ut.get_size(self.__dict__[key])>10000:
                        print(key, ut.get_size(self.__dict__[key]) )
                except:
                    print('Could not save: self.'+key+' --> Skipping!')
                    
        #Save the dictionary obj
        lastsave = ut.savepickle(obj, filename)
        
        return lastsave
       

    def load(self,filename):
        # Loads pickle data to self
        obj=ut.loadpickle(filename)
        
        for key in obj.keys():
            self.__dict__[key] = obj[key]

        #defaults for plotting
        self.colors={'He':'yellow', 'H2':'orange','N2':'gray',
                     'CH4':'purple', 'C2H2':'lightgreen', 'O2':'gray', 'OH':'lightgray', 'H2O':'blue', 'CO':'red', 'CO2':'green',
                     'NH3':'brown', 'HCN':'peru', 'H2S':'C8', 'PH3':'C1', 'Na':'C9', 'K':'C6',
                     'TiO':'#1f77b4', 'SiO':'#ff7f0e', 'H-':'#2ca02c', 'VO':'#d62728', 'HDO':'#9467bd', 'FeH':'#8c564b', 'O3':'#8c564b', 'SO2':'gold', 'AlO':'gold', 'CrH':'gray', 'CrO':'r', 'CrO2':'g', 'CrO3':'k', 'VO2':'gray', 'TiO2':'gray', 'TiS':'gray', 'TiH':'gray',
                     'H':'C1', 'He+':'C2', 'Li':'C3', 'Li+':'C4', 'Be':'C5', 'Be+':'C6', 'Be++':'C7', 'B':'C5', 'B+':'C6', 'B++':'C7', 'C':'C8', 'C+':'C9', 'C++':'C10',
                     'N':'C11', 'N+':'C12', 'N++':'C13', 'O':'C14', 'O+':'C15', 'O++':'C16', 'F':'C17', 'F+':'C18', 'F++':'C19', 'Ne':'C20',
                     'Ne+':'C21', 'Ne++':'C22', 'Na+':'C23', 'Na++':'C24', 'Mg':'C25', 'Mg+':'C26', 'Mg++':'C27', 'Al':'C28', 'Al+':'C29', 'Al++':'C30',
                     'Si':'C31', 'Si+':'C32', 'Si++':'C33', 'P':'C31', 'P+':'C32', 'P++':'C33', 'S':'C31', 'S+':'C32', 'S++':'C33', 'Cl':'C34', 'Cl+':'C35', 'Cl++':'C36', 'Ar':'C37', 'Ar+':'C38', 'Ar++':'C39', 'K+':'C40',
                     'K++':'C41', 'Ca':'C42', 'Ca+':'C43', 'Ca++':'C44', 'Sc':'C45', 'Sc+':'C46', 'Sc++':'C47', 'Ti':'g', 'Ti+':'m', 'Ti++':'C50',
                     'V':'c', 'V+':'C52', 'V++':'C53', 'Cr':'C54', 'Cr+':'C55', 'Cr++':'C56', 'Mn':'C57', 'Mn+':'C58', 'Mn++':'C59', 'Fe':'r',
                     'Fe+':'orange', 'Fe++':'y', 'Co':'C63', 'Co+':'C64', 'Co++':'C65', 'Ni':'C66', 'Ni+':'C67', 'Ni++':'C68', 'Cu':'C69', 'Cu+':'C70', 'Cu++':'C70',
                     'Zn':'C71', 'Zn+':'C72', 'Zn++':'C73', 'Ga':'C74', 'Ga+':'C75', 'Ga++':'C76', 'Ge':'C77', 'Ge+':'C78', 'As':'C79', 'Se':'C80',
                     'Br':'C81', 'Kr':'C82', 'Rb':'C83', 'Sr':'C84', 'Sr+':'C84', 'Y':'C85', 'Y+':'C86', 'Y++':'C87', 'Zr':'C88', 'Zr+':'C89', 'Zr++':'C90',
                     'Nb':'C91', 'Nb+':'C92', 'Nb++':'C93', 'Mo':'C94', 'Mo+':'C95', 'Mo++':'C95', 'Tc':'C96', 'Tc+':'C96', 'Ru':'C97', 'Ru+':'C98', 'Ru++':'C98', 'Rh':'C99', 'Rh+':'C100', 'Rh++':'C100',
                     'Pd':'C101', 'Pd+':'C102', 'Ag':'C103', 'Ag+':'C104', 'Cd':'C105', 'Cd+':'C106', 'In':'C107', 'In+':'C108', 'Sn':'C109', 'Sn+':'C110',
                     'Sb':'C111', 'Te':'C112', 'I':'C113', 'Xe':'C114', 'Xe+':'C115', 'Cs':'C116', 'Ba':'C117', 'Ba+':'C118', 'La':'C119', 'La+':'C120',
                     'La++':'C121', 'Ce':'C122', 'Ce+':'C123', 'Ce++':'C124', 'Pr':'C125', 'Pr+':'C126', 'Pr++':'C127', 'Nd':'C128', 'Nd+':'C129', 'Nd++':'C130',
                     'Pm':'C131', 'Sm':'C132', 'Sm+':'C133', 'Sm++':'C134', 'Eu':'C135', 'Eu+':'C136', 'Eu++':'C137', 'Gd':'C138', 'Gd+':'C139', 'Gd++':'C140',
                     'Tb':'C141', 'Tb+':'C142', 'Tb++':'C143', 'Dy':'C144', 'Dy+':'C145', 'Dy++':'C146', 'Ho':'C147', 'Ho+':'C148', 'Ho++':'C149', 'Er':'C150',
                     'Er+':'C151', 'Er++':'C152', 'Tm':'C153', 'Tm+':'C154', 'Tm++':'C155', 'Yb':'C156', 'Yb+':'C157', 'Yb++':'C158', 'Lu':'C159', 'Lu+':'C160',
                     'Lu++':'C161', 'Hf':'C162', 'Hf+':'C163', 'Hf++':'C164', 'Ta':'C165', 'Ta+':'C166', 'W':'C167', 'W+':'C168', 'Re':'C169', 'Re+':'C170',
                     'Os':'C171', 'Os+':'C172', 'Ir':'C173', 'Ir+':'C174', 'Pt':'C175', 'Pt+':'C176', 'Pt++':'C177', 'Au':'C178', 'Au+':'C179', 'Au++':'C180',
                     'Hg':'C181', 'Hg+':'C182', 'Hg++':'C183', 'Tl':'C184', 'Pb':'C185', 'Pb+':'C186', 'Bi':'C187', 'Bi+':'C188', 'Po':'C189', 'At':'C190',
                     'Rn':'C191', 'Fr':'C192', 'Ra':'C193', 'Ac':'C194', 'Th':'C195', 'Th+':'C196', 'Th++':'C197', 'Pa':'C198', 'U':'C199', 'U+':'C200'}
        self.offset=0.0
        self.color=None
        
        self.filename=filename
        self.filebase=self.filename[:-len(self.runName)-4]   #in case the file was moved, use the new position when saving new outputs
        
        self.label=self.autoLabel() #self.runName

        if 'opacSources' not in self.__dict__:
            self.opacSources={'molLineAbsorb':True,
                                        'cia':True,
                               'rayleighScat':True,
                               'fineHazeScat':True,
                              'paramMieCloud':True,
                              'carmaMieCloud':True}

        # Re-create variables that were not saved (for file size reasons)
#        if 'modelSetting' in self.__dict__:          
#            self.calcSecEclppm()


    def makeStruc(self,refWaveRangeForOptDepth=[1.2,1.8]):
        #Structure only
        struc=pd.DataFrame(self.p,columns=['Pressure'])
        struc['Temperature']=self.T
        struc['z']=self.z
        struc['grav']=self.grav
        struc['ntot']=self.ntot
        struc['MuAve']=self.MuAve/uAtom
        struc['rho']=self.MuAve*self.ntot
        struc['scaleHeight']=self.scaleHeight
        struc.to_csv(self.filebase+self.runName+'_strucTp.csv')

        #Structure + AbsMol
        comp = pd.DataFrame(self.qmol_lay[:,self.AbsMolInd],columns=self.AbsMolNames)
        self.struc = pd.concat([struc,comp], axis=1)
        self.struc.to_csv(self.filebase+self.runName+'_struc.csv')

        #Structure + all molecules + optical depth
        allcomp = pd.DataFrame(self.qmol_lay,columns=self.MolNames)
        self.strucAll = pd.concat([struc,allcomp], axis=1)
        
        if self.saveOpac and 'opac' in self.__dict__.keys():
            iwave=np.arange(bisect(self.wave,refWaveRangeForOptDepth[0]),bisect(self.wave,refWaveRangeForOptDepth[1]))
            # for quantity in ['extinctCoef','tau','transmis','tauTransit','transmisTransit']:
            for quantity in ['extinctCoef','tau','transmis','tauTransit','absorptionTransit']:
                self.strucAll[quantity+'_{:g}-{:g}'.format(*refWaveRangeForOptDepth)]=np.median(self.opac[quantity][:,iwave],axis=1)

        self.strucAll.to_csv(self.filebase+self.runName+'_strucAll.csv')
        
        

    def saveSpectrum(self,format='ascii.ecsv',filename=None,saveSettings=None,kernel=None,norm=False,saveMolNames=['H2O','CO','CH4'],makeCopy=False):
        '''
        saveSettings=np.array([1,0,0,0,0,0])   #[CalcTransSpec,CalcThermSpec,CalcReflectSpec,MolByMolPlot,MolByMolPlotThermo,MolByMolPlotReflect]
        '''
        
        #Default value
        if saveSettings is None:
            saveSettings=np.array([1,0,0,0,0,0])
            if self.modelSetting['thermalOnly']!=[] and self.modelSetting['thermalOnly'][0]!='thermalNoScatMol':
                saveSettings[1]=1
            if self.modelSetting['albedoOnly']!=[]:
                saveSettings[2]=1
            if 'dppmMol' in self.modelSetting['transit']:
                saveSettings[3]=1
            if 'thermalNoScatMol' in self.modelSetting['thermalOnly']:
                saveSettings[4]=1


        if filename is None:
            if kernel is None:
                filename=self.filebase+self.runName+'_Spectrum_FullRes'
                self.specFile = filename
            else:                    
                filename=self.filebase+self.runName+'_Spectrum'
                self.specFile = filename

            if (saveSettings==np.array([1,0,0,0,0,0])).all():
                filename=filename+'_dppm'    

        self.makeStruc()

        t = Table()

        if kernel is None:
            t['wave'] = self.wave;  t['wave'].unit='um'
            if saveSettings[0]: 
                t['dppm'] = self.dppm;  t['dppm'].unit='ppm'
            if saveSettings[1]:
                t['thermal'] = self.thermal;  t['thermal'].unit = 'W/(m**2*um)'
                if saveSettings[1]==2:
                    for i,muObs in enumerate(self.muObs):
                        t['mu={:.2f}'.format(muObs)] = self.thermalMuObs[i,:];  t['mu={:.2f}'.format(muObs)].unit = 'W/(m**2*um)'
            if saveSettings[2]:
                t['albedoSpec'] = self.albedoSpec;  t['albedoSpec'].unit = '1'
            if saveSettings[3]:
                for MolName in saveMolNames: 
                    t['dppm['+MolName+']'] = self.dppmMol[self.molInd[MolName],:];  t['dppm['+MolName+']'].unit = 'ppm'
            if saveSettings[4]:
                for MolName in saveMolNames: 
                    t['thermal['+MolName+']'] = self.thermalNoScatMol[self.molInd[MolName],:];  t['thermal['+MolName+']'].unit = 'W/(m**2*um)'
        else:
            l = kernel.shape[0]/2
            t['wave'] = self.wave[l:-l];   t['wave'].unit='um'
            if saveSettings[0]: 
                smoothedSpec = convolve(self.dppm, kernel)
                t['dppm'] = smoothedSpec[l:-l];  t['dppm'].unit='ppm'
            if saveSettings[1]:
                smoothedSpec = convolve(self.thermal, kernel)
                t['thermal'] = smoothedSpec[l:-l];  t['thermal'].unit = 'W/(m**2*um)'
            if saveSettings[2]:
                smoothedSpec = convolve(self.albedoSpec, kernel)
                t['albedoSpec'] = smoothedSpec[l:-l];  t['albedoSpec'].unit = 'W/(m**2*um)'
            if saveSettings[3]:
                for MolName in saveMolNames: 
                    smoothedSpec = convolve(self.dppmMol[self.molInd[MolName],:], kernel)
                    t['dppm['+MolName+']'] = smoothedSpec[l:-l];  t['dppm['+MolName+']'].unit = 'ppm'
            if saveSettings[4]:
                for MolName in saveMolNames: 
                    smoothedSpec = convolve(self.thermalMol[self.molInd[MolName],:], kernel)
                    t['thermal['+MolName+']'] = smoothedSpec[l:-l];  t['thermal['+MolName+']'].unit = 'W/(m**2*um)'

        for key in self.params:
            if (type(self.params[key])==float) or (type(self.params[key])==int) or (type(self.params[key])==str):
                t.meta[key]=self.params[key]
#            elif type(self.params[key]).__module__ == np.__name__:
#                t.meta[key]=float(t.meta['params'][key])
        t.meta['resPower'] = float((self.resPower))
        t.meta['waveMax'] = float(np.max(self.wave))
        t.meta['waveMin'] = float(np.min(self.wave))
        t.meta['chiSquared'] = float(self.chi2)
        t.meta['MoleculeListFile'] = self.MoleculeListFile
        t.meta['casename'] = self.filename
        t.meta['citation'] = 'Benneke & Seager 2012, 2013; Benneke 2015; Benneke et al. 2019a,b'

        t.write(filename+'.csv',format=format,delimiter=',',overwrite=True)
        
        if makeCopy: #Make an identical copy one directory up
            filename2=os.path.dirname(os.path.dirname(filename)) + '/' + os.path.basename(filename)
            t.write(filename2+'.csv',format=format,delimiter=',',overwrite=True)
        
        


    #%% Plotting Methods 
    
    def plotSpectrum(self,ax=None,spectype='dppm',resPower=None,kernel=None,label=None,labelAddChi2=False,save=False,specs=None,
                     xscale='log',xlim=None,ylim=None,setylim=None,forceylim=True,presAxis=True,presLevels=False,
                     iMol=None,iMuObs=None,figsize=None,sat=False,inclExpInAxisLabel=False,
                     offset=0,subtrMeanWaveRange=None,fillCurve=None,**kwargs):
        '''
        fillCurve: None,'below','above'
        '''

        #Making figure
        if ax is None:
            fig,ax=plt.subplots(figsize=figsize)
        else:
            fig=None
        ax.set_xlabel(r'Wavelength [$\mathrm{\mu m}$]')
        ax.spectype=spectype
        
        # select quantity on y-axis and set_ylabel
        if spectype=='dppm':
            ax.set_ylabel(r'Transit Depth [ppm]')
            if iMol is None:
                y = self.dppm
            else:
                y = self.dppmMol[iMol,:]

        elif spectype=='thermal':
            ax.set_ylabel(r'Thermal Emission at TOA $\left[\mathrm{W/m^{2}\mu m}\right]$')
            if iMol is None:
                y = self.thermal
            else:
                y = self.thermalNoScatMol[iMol,:]
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        elif spectype=='Tbright':
            ax.set_ylabel(r'Brightness Temperature (K)')
            y = rad.calcTBright(self.thermal,self.wave,'W/(m**2*um)', 'um')

        elif spectype=='thermalDirect':
            ax.set_ylabel(r'Thermal Emission at TOA (bandpass integrated) $\left[\mathrm{W/m^{2}}\right]$')
            y = np.zeros_like(self.thermal, dtype=self.numerical_precision)  #just make a line at 0

        elif spectype=='totalOutgoingFlux':
            ax.set_ylabel(r'Total Outgoing Flux at TOA $\left[\mathrm{W/m^{2}}\right]$')
            y = self.totalOutgoingFlux() * np.ones_like(self.wave, dtype=self.numerical_precision)

        elif spectype=='thermalPhotons':
            ax.set_ylabel(r'Thermal Emission at TOA $\left[\mathrm{photons/m^{2}\mu m}\right]$')
            if iMol is None:
                y = self.thermal  / (hPlanck*self.f)
            else:
                y = self.thermalMol[iMol,:] / (hPlanck*self.f)
                
        elif spectype=='thermalMuObs':
            ax.set_ylabel(r'Thermal Emission at TOA $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.thermalMuObs[iMuObs,:]

        elif spectype=='thermal2Col':
            ax.set_ylabel(r'Thermal Emission at TOA $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.thermal2Col
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        elif spectype=='secEclppm':
            ax.set_ylabel(r'Eclipse Depth [ppm]')
            self.calcSecEclppm()
            if iMol is None:
                y = self.secEclppm
                
        elif spectype=='secEclppm_halfBB':
            ax.set_ylabel(r'Eclipse Depth [ppm]')
            self.calcSecEclppm()
            flux = rad.PlanckFct(420,self.wave,InputUnit='um',OutputUnit='W/(m**2*um)',RadianceOrFlux='flux')
            secEclppmBB=1e6 * (flux*self.params['Rp']**2) / (self.fStarSurf*self.params['Rstar']**2)
            if iMol is None:
                y = 0.4 * self.secEclppm + 0.6*secEclppmBB
                                
        elif spectype=='thermalSecEclppm':
            ax.set_ylabel(r'Eclipse Depth [ppm]')
            self.calcSecEclppm()
            if iMol is None:
                y = self.thermalSecEclppm
            else:
                #y = 1e6 *self.params['Rp']**2 / self.params['Rstar']**2  * self.thermalMol[iMol,:] / (rad.PlanckFct(self.params['Teffstar'],self.wave,'um' ,'W/(m**2*um)','flux'))
                #y = self.dppm * self.thermalMol[iMol,:] / rad.PlanckFct(self.params['Teffstar'],self.wave,'um' ,'W/(m**2*um)','flux')
                y = self.dppm * self.thermalMol[iMol,:] / self.fStarSurf

        #----------------------------------------------------------        
        elif spectype=='albedo':
            ax.set_ylabel(r'Planetary Albedo')
            self.calcSecEclppm()
            if iMol is None:
                y = self.albedoSpec
            else:
                y = self.albedoSpec[iMol,:]

        elif spectype=='albedoDisort':
            ax.set_ylabel(r'Planetary Albedo from pyDISORT')
            y = self.albedoDisort
            
        elif spectype=='albedoFeautrier':
            ax.set_ylabel(r'Planetary Albedo from Feautrier')
            y = self.albedoFeautrier
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))   
            
        elif spectype=='albedoToon':
            ax.set_ylabel(r'Planetary Albedo from Toon')
            y = self.albedoToon

        elif spectype=='albedoSecEclppm':
            ax.set_ylabel(r'Reflective Light Eclipse Depth [ppm]')
            self.calcSecEclppm()
            if iMol is None:
                y = self.albedoSecEclppm
              
        #----------------------------------------------------------        
        elif spectype=='thermalToon':
            ax.set_ylabel(r'Thermal Emission at TOA (Toon) $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.thermalToon
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        elif spectype=='thermalFeautrier':
            ax.set_ylabel(r'Thermal Emission at TOA (Feautrier) $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.thermalFeautrier
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        #----------------------------------------------------------
        elif spectype=='totalFluxDisort':
            ax.set_ylabel(r'Thermal + Reflected Emission at TOA (Disort) $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.totalFluxDisort
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        elif spectype=='totalFluxFeautrier':
            ax.set_ylabel(r'Thermal + Reflected Emission at TOA (Feautrier) $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.totalFluxFeautrier
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        elif spectype=='totalFluxToon':
            ax.set_ylabel(r'Thermal + Reflected Emission at TOA (Toon) $\left[\mathrm{W/m^{2}\mu m}\right]$')
            y = self.totalFluxToon
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        else:
            raise ValueError("unknown spectype: " + spectype + '(valid options are: dppm,thermal,secEclppm,thermalSecEclppm,albedo,albedoSecEclppm)')

        #Apply offset or subtract median 
        y=y+offset
        if subtrMeanWaveRange is not None:
            yMedian = np.mean(y[np.logical_and(self.wave>subtrMeanWaveRange[0],self.wave<subtrMeanWaveRange[1])])
            y=y-yMedian

        #Colors and curve label
        if label is None:
            if 'label' in self.__dict__.keys():
                label=self.label
            else:
                if iMol is not None:            
                    molName = self.AbsMolNames[iMol]            
                    kwargs['color']=kwargs.get('color',self.colors[molName])
                    if label is None:
                        label = molName
                elif iMuObs is not None:
                    label = 'mu={:.4f}'.format(self.muObs[iMuObs])
                else:
                    label=self.runName
                    if resPower is not None:
                        if resPower<self.resPower:
                            label=label+'_R='+str(resPower)
                    if labelAddChi2:    
                        label=label+'_'+str(self.chi2)
        
        if resPower is not None:
            if resPower<self.resPower:
                kernel=Gaussian1DKernel(self.resPower / resPower / 2.35)    # *2 because FWHM = 2 standard deviation
                
        
        print(spectype)

        #Plotting
        if kernel is None:
            x=self.wave
            hline=ax.plot(x,y,label=label,**kwargs)
            if fillCurve=='below':
                ax.fill_between(x, y, 0, **kwargs, alpha=.1)
            #ax.set_xlim([x[0],x[-1]])
        elif kernel=='niriss':
            x,y = self.niriss()            
            hline=ax.plot(x,y,label=label+'_NIRISS',**kwargs)
            #ax.set_xlim([self.wave[0],self.wave[-1]])
            
        else:
            l = int(kernel.shape[0]/2)
            x = self.wave[l:-l]
            y = convolve(y, kernel)[l:-l]
            hline=ax.plot(x,y,label=label,**kwargs)
            if fillCurve=='below':
                ylimBefore = ax.get_ylim()
                ax.fill_between(x, y, 0, **kwargs, alpha=.1)
                ax.set_ylim(ylimBefore)
            ax.set_xlim([x[0],x[-1]])

        #Set xlim
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            xlim=[self.wave[0],self.wave[-1]]

        if ylim is None:
            if setylim is not None:
                if type(setylim) is float:
                    oldylim=ax.get_ylim()
                    r1=np.min(y[x>xlim[0]]); r2=np.max(y[x<xlim[1]])
                    newylim=[r1-setylim*(r2-r1),r2+setylim*(r2-r1)]
                    goodylim = [  np.min([oldylim[0],newylim[0]])  , np.max([oldylim[1],newylim[1]])  ]
                    #print oldylim, newylim, goodylim                  
                    if forceylim:
                        ax.set_ylim(newylim)
                    else:                
                        ax.set_ylim(goodylim)
                else:  # len(setylim)==2:
                    ax.set_ylim(setylim)
            else:
                if spectype=='thermal':
                    ax.set_ylim(bottom=0)

        if ylim is not None:
            ax.set_ylim(ylim)

        if inclExpInAxisLabel:
            exponent=np.int(np.log10(ax.get_ylim()[1]))
            ax.yaxis.get_offset_text().set_visible(False)
            #y=y/10**exponent
            ylabel=ax.get_ylabel()
            ind=ylabel.find('[')
            ylabel=ylabel[:ind+1] + r'10^{{:d}}'.format(exponent) +'\ ' + ylabel[ind+1:]
            ax.set_ylabel(ylabel)



        #Overplot data points
        if specs is not None:
            for spec in specs:
                if spec.meta['spectype']==spectype:
                    spec.plot(ax=ax,zorder=10,lw=1,xunit='um',showDataIndex=False)
#                    if 'label' in spec.meta:
#                        specLabel=spec.meta['label'][:110]
#                    else:
#                        specLabel=None
#                    spec.plot(ax=ax,color='black',zorder=10,lw=1,xunit='um',showDataIndex=False,label=specLabel)

        ax.legend(fontsize=6)

        self.color=hline[0].get_color()
     
        if 'defaultxlim' in self.__dict__.keys(): 
            if self.defaultxlim is not None:
                ax.set_xlim(self.defaultxlim)
        
        if xscale=='log':
            ax.set_xscale('log')
            ut.xspeclog(ax,level=1)

        if spectype=='dppm' and presLevels:
            ax.set_ylim(ax.get_ylim())
            ax.axhline(self.RpBase**2/self.Rstar**2 *1e6,color='black')
            for zz in self.z:
                ax.axhline((self.RpBase+zz)**2/self.Rstar**2 *1e6,lw=0.1,ls='--',color='black')

        if spectype=='dppm' and presAxis:
            ut.makeSecYAxis(ax,(self.RpBase+self.z)**2/self.Rstar**2 *1e6,self.p/100,np.array([10**i for i in [5,4,3,2,1,0,-1,-2,-3]]),label='Pressure [mbar]',yminorticks2=np.hstack([np.array([1,2,4,6,8])*10**i for i in np.arange(-3.0,5.0)]))

        if fig is not None:
            if save:
                #print 'Saving:  ' +  self.filebase+self.runName+'_Spectrum_'+spectype+'.pdf'            
                fig.savefig(self.filebase+self.runName+'_Spectrum_'+spectype+'.pdf')            
            return fig,ax
        else:
            return hline[0]


    def plotCombo(self,resPower=100,save=False):
        fig,axs=plt.subplots(3,3,dpi=50,figsize=[26,20]); axs=axs.flatten()
        
        self.plotSpectrum(ax=axs[0],spectype='dppm',resPower=resPower,presAxis=False)
        if 'dppmMol' in self.modelSetting['transit']:
            self.plotSpectraByMol(ax=axs[3],spectype='dppm',resPower=resPower,presAxis=False)
        axs[0].get_shared_x_axes().join(axs[0], axs[3])
        #axs[0].get_shared_y_axes().join(axs[0], axs[3])
        
        self.plotSpectrum(ax=axs[1],spectype='secEclppm',resPower=resPower)

        self.plotSpectrum(ax=axs[2],spectype='thermal',resPower=resPower)
        if 'thermalNoScatMol' in self.modelSetting['thermalOnly']:
            self.plotSpectraByMol(ax=axs[5],spectype='thermal',resPower=resPower)
        axs[2].get_shared_x_axes().join(axs[2], axs[5])
        
        self.plotSpectrum(ax=axs[6],spectype='Tbright',resPower=resPower)
        self.plotTp(ax=axs[7],forceLabelAx=True)
        self.plotComp(ax=axs[8])
        
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        if save:
            fig.savefig(self.filebase+self.runName+'_combo.pdf')            
        
        return fig,axs
    

    def plotSpectraByMol(self,ax=None,save=False,spectype='dppm',setylim=None,specs=None,MolNamesToPlot=None,
                         totalLabel='total',lw=0.5,lwTotal=2,fillCurve=None,**kwargs):

        #Defaults
        if fillCurve is None:
            if spectype=='dppm':
                fillCurve='below'
                
        if MolNamesToPlot is None:
            MolNamesToPlot=self.AbsMolNames
        if ax is None:
            fig,ax=self.plotSpectrum(spectype=spectype,specs=specs,color='black',label=totalLabel,setylim=setylim,lw=lwTotal,**kwargs)
        else:
            self.plotSpectrum(ax=ax,spectype=spectype,specs=specs,color='black',label=totalLabel,setylim=setylim,lw=lwTotal,**kwargs)
            fig=ax.get_figure()
        for MolName in MolNamesToPlot:
            iMol=self.getAbsMolIndex(MolName)
            #self.plotSpectrum(ax=ax,spectype=spectype,iMol=iMol,lw=lw,label=ut.chemlatex(MolName),color=self.colors[MolName],**kwargs)  
            self.plotSpectrum(ax=ax,spectype=spectype,iMol=iMol,lw=lw,label=MolName,color=self.colors[MolName],fillCurve=fillCurve,**kwargs)  
        ax.legend(loc=3,fontsize='xx-small')
        if spectype=='thermal':
            ax.set_ylim(bottom=0.0)
        if save:
            fig.savefig(self.filebase+self.runName+'_SpectraByMol_'+spectype+'.png')            
            fig.savefig(self.filebase+self.runName+'_SpectraByMol_'+spectype+'.pdf')  
        return fig,ax
    
    
        
    def plot_fStarSurf(self,save=False):
        fig,ax=ut.newFig(xlabel='Wavelength [um]',ylabel='Flux [W/(m**2*um)]',log=[True,False])
        ax.plot(self.wave,self.fStarSurf,label='fStarSurf')
        ax.plot(self.wave,rad.PlanckFct(self.Teffstar,self.wave,'um','W/(m**2*um)','flux'),label='Planck Function with T=Teffstar')
        L = np.trapz(x=self.wave,y=self.fStarSurf*4*pi*self.Rstar**2)
        ax.set_title('Total luminosity of star: {} W/(m**2)  [params["Lstar"]: {} W/(m**2) ]'.format(L,self.params['Lstar']))
        ax.legend()
        if save:
            fig.savefig(self.filebase+self.runName+'_fStarSurf.pdf')
        return fig,ax

    def plot_IrradStarEff(self,save=False):
        fig,ax=ut.newFig(xlabel='Wavelength [um]',ylabel='Flux [W/(m**2*um)]',log=[True,False])
        ax.plot(self.wave,self.IrradStarEff)
        ax.set_title('Total Effective Incident Flux for 1D model: {} W/(m**2)'.format(self.totalEffIncFlux))
        if save:
            fig.savefig(self.filebase+self.runName+'_IrradStarEff.pdf')
        return fig,ax
    
    def plotThermalMuObs(self,save=False,**kwargs):
        #self.plotSpectrum(spectype='thermal',label='total',color='black',**kwargs)
        fig,ax=self.plotSpectrum(spectype='thermalMuObs',iMuObs=0,lw=1,**kwargs)
        for iMuObs in np.arange(1,len(self.muObs)):
            self.plotSpectrum(ax=ax,spectype='thermalMuObs',iMuObs=iMuObs,lw=1,**kwargs)
        ax.legend(loc='best',fontsize='xx-small')
        if save:
            fig.savefig(self.filebase+self.runName+'_thermalMuObs.pdf')  
        return fig,ax        


    def plotTp(self,ax=None,axisOnly=False,figsize=[8,10],save=False,showTeq=True,marker='x',punit='bar',partialPresMol=None,forceLabelAx=False,**kwargs):
        ## show convection limit
        if ax is None:
            fig, ax = plt.subplots(1,figsize=figsize)
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure ['+punit+']')
            if axisOnly is False:
                if showTeq:
                    plt.axvline(x=self.Teq,color='r',linestyle=':',zorder=-10)
                if partialPresMol is None:
                    ax.semilogy(self.T,self.p*unitfac(punit),marker=marker,**kwargs)  
                    ax.set_ylabel('Pressure ['+punit+']')
                else:
                    ax.semilogy(self.T,self.p*unitfac(punit)*self.getMixRatio(partialPresMol),marker=marker,**kwargs)  
                    ax.set_ylabel('Partial pressure of {} [{}]'.format(partialPresMol,punit))
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_Tp.pdf')            
            return fig,ax
        else:
            if partialPresMol is None:
                ax.semilogy(self.T,self.p*unitfac(punit),marker=marker,**kwargs)  
                ax.set_ylabel('Pressure ['+punit+']')
            else:
                ax.semilogy(self.T,self.p*unitfac(punit)*self.qmol_lay[:,self.getMolIndex(partialPresMol)],marker=marker,**kwargs)  
                ax.set_ylabel('Partial pressure of {} [{}]'.format(partialPresMol,punit))
                
        if forceLabelAx:
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure ['+punit+']')
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()


    def plotTz(self,ax=None,save=False,makeConvAdjust=False,showTeq=True,**kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Altitude [m]')
            if showTeq:
                ax.axvline(x=self.Teq,color='r',linestyle='--')

            z1bar=ut.interp1dEx(np.log(self.p),self.z,np.log(1e5))  # [m] one bar level
            ax.axhline(y=z1bar,color='gray',linestyle='--')

            ax.plot(self.T,self.z,marker='x',**kwargs)  

            dT=np.diff(self.T)
            dTdz = dT / self.dz
                
            #Show slope for onset of convection:
#            heatCap = 14300 #J/kgK   = 14.30 J/gK #https://en.wikipedia.org/wiki/Lapse_rate
#            grav=0.5*(self.grav[:-1]+self.grav[1:])
#            dTdzAdiabat = grav/heatCap  
            MuAveLay = 0.5*(self.MuAve[:-1]+self.MuAve[1:])
            Rconst = 8.3144598 /(MuAveLay/uAtom * 0.001)#J/mol*K
            heatCap = 7./2. * Rconst
            grav=0.5*(self.grav[:-1]+self.grav[1:])
            dTdzAdiabat = grav/heatCap  

            slopeTooSteep = np.where(dTdz > dTdzAdiabat)[0]
            if len(slopeTooSteep)>0:
                iLay=slopeTooSteep[0]
                Tline = self.T[iLay] - dTdzAdiabat[iLay] * (self.z - self.z[iLay]) 
                print(dTdzAdiabat[iLay]*1000, 'K/km')
                ax.plot(Tline,self.z,color='gray',lw=0.5, label = 'Slope is too steep at layer ' + str(iLay))

                if makeConvAdjust:
                    for i in range(iLay,len(self.p)-1):
                        self.T[i+1] = self.T[i] + dTdzAdiabat[i] * self.dz[i]
            else:
                iLay=self.nLay-2
                Tline = self.T[iLay] - dTdzAdiabat[iLay] * (self.z - self.z[iLay]) 
                print(dTdzAdiabat[iLay]*1000, 'K/km')
                ax.plot(Tline,self.z,color='gray',lw=0.5,ls='--')
            
            ##Jonathan            
            Tline = 0.5*((self.T[-1] - dTdzAdiabat[-1] * (self.z - self.z[-1])) + (self.T[-2] - dTdzAdiabat[-2] * (self.z - self.z[-2])))
            
            ax.plot(Tline,self.z,color='red',lw=0.5,label = 'Cp = '+str(heatCap[-1]))

            ax.set_xlim([0,self.T[-1]+500])  
            
            if self.modelSetting['TempType']=='NonGrayConv':
                if 'convLay' in self.__dict__.keys():
                    if len(self.convLay) != 0:
                        ax.plot(self.T[self.convLay],self.z[self.convLay],marker = 'x', color = 'r', label = 'Convective Layer(s)')
            
            ax.legend(loc='best')
            ###Jonathan
            
            ax.set_ylim([self.z[-1],self.z[0]])
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_Tz.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.z,marker='x',**kwargs) 
            

    def plot_dlnP_dlnT(self,ax=None,save=False,**kwargs):
           if ax is None:
               fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
               ax.set_xlabel('d(lnP)/d(lnT)')
               ax.set_ylabel('Pressure [bar]')
    
               grad = np.zeros([self.nLay], dtype=self.numerical_precision)
               lnT = np.log(self.T)
               lnP = np.log(self.p)
               
               for i in range(1,self.nLay):
                   grad[i]=(((lnT[i]-lnT[i-1])/(lnT[i]+lnT[i-1]))*
                            ((lnP[i]+lnP[i-1])/(lnP[i]-lnP[i-1])))
               grad[0] = grad[1]
               ax.semilogy(grad,self.p/1e5,marker='x',label='TP Gradient',**kwargs)

               ax.legend(loc=3,fontsize='xx-small')
               ax.set_ylim([1e-12,1e4])
               ax.invert_yaxis()
               ax.minorticks_on()

               if save:
                   fig.savefig(self.filebase+self.runName+'_TpGradient.pdf')

               fig2,ax2=plt.subplots()    
               ax2.plot(lnP,lnT,marker='x',label='TP Gradient',**kwargs)



               return fig,ax


           else:
               ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs)

            
    def plotComp(self,ax=None,ppm=False,save=False,MolNames=None):
        
        if MolNames is None:
            MolNames=self.MolNames
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
        else:
            fig=None

        ax.invert_yaxis()

        if not ppm:
            ax.set_xlabel('Mixing ratio [1]')
            ax.set_xlim([1e-10,1.1e0])
        else:
            ax.set_xlabel('Mixing ratio [ppm]')
            ax.set_xlim(np.array([1e-10,1.1e0])*1e6)
        ax.set_ylabel('Pressure [bar]')
        ax.set_ylim([1e-10,1e2])
        ax.invert_yaxis()
        #ax.grid(b=True, which='major', axis='both',color='lightgrey')
        ax.grid(which='major',linestyle='-')
        #ax.axhline(y=10,color='k',linestyle='--')
        ax.axhline(y=1,color='k')
        for MolName in MolNames:
            self.plotMixRatio(ax,MolName,ppm=ppm)
        if len(MolNames) > 24:
            ax.legend(loc=3,fontsize='xx-small', ncol=3)
        if len(MolNames) > 12:
            ax.legend(loc=3,fontsize='xx-small', ncol=2)
        else:
            ax.legend(loc=3,fontsize='xx-small', ncol=1)

        if fig is not None:
            if save:
                fig.savefig(self.filebase+self.runName+'_MixRatios.pdf')            
            return fig,ax


    def plotMixRatio(self,ax,MolName,ppm=False,label=None,linewidth=1.5,**kwargs):
        if MolName in self.colors.keys():
            kwargs['color']=kwargs.get('color',self.colors[MolName])
        else:
            kwargs['color']='gray'
        if label is None:
            label = MolName
        if MolName not in self.AbsMolNames:
            linewidth=0.5*linewidth
        iMol=np.where(self.MolNames==MolName)[0][0]
        if ppm:
            factor=1e6
        else:
            factor=1
        ax.loglog(self.qmol_lay[:,iMol]*factor,self.p/1e5,marker='+', label=label, linewidth=linewidth, **kwargs)

        
    def getMolIndex(self,MolName):
        iMol=np.where(self.MolNames==MolName)[0][0]
        return iMol

    def getAbsMolIndex(self,MolName):
        iMol=np.where(self.AbsMolNames==MolName)[0][0]
        return iMol
    
    def getMixRatio(self,MolName):
        iMol=np.where(self.MolNames==MolName)[0][0]
        return self.qmol_lay[:,iMol]
        

#    def plotBlack(self,ax,wave=None,T=[1600,1400,1200,1000,800,600,400,200], **kwargs):
#        if wave is None:
#            wave=self.wave
#        for Ti in T:
#            ax.plot(wave,rad.PlanckFct(Ti,wave,'um' ,'W/(m**2*um)','flux'), label=T, **kwargs)

    def plotBB(self,ax,T,spectype='thermal',wave=None,InputUnit='um',OutputUnit='W/(m**2*um)',label=None,scalefactor=1.0,**kwargs):
        
        if wave is None:
            wave=self.wave
        if label is None:
            label='T = {:g} K'.format(T)
        
        flux = rad.PlanckFct(T,wave,InputUnit='um',OutputUnit='W/(m**2*um)',RadianceOrFlux='flux')
        
        if spectype=='thermal':
            ax.plot(wave,flux*scalefactor,label=label,**kwargs)
        elif spectype=='secEclppm' or spectype=='thermalSecEclppm':
            # if 'fStarSurf' in self.params:
            #     y=1e6 * (flux*self.params['Rp']**2) / (self.fStarSurf*self.params['Rstar']**2)
            #     ax.plot(wave,y,label=label,**kwargs)
            # else:
            #     y=1e6 * (flux*self.params['Rp']**2) / (rad.PlanckFct(self.params['Teffstar'],self.wave,'um' ,'W/(m**2*um)','flux')*self.params['Rstar']**2)    
            #     ax.plot(wave,y,label=label,**kwargs)

            # self.fStarSurf is already either a star model or planck function (see in runModel)
            y=1e6 * (flux*self.params['Rp']**2) / (self.fStarSurf*self.params['Rstar']**2)
            ax.plot(wave,y*scalefactor,label=label,**kwargs)

        else:
            print('WARNING: Unknown spectype in atmosphere.plotBB')


    def plotIrradStar(self,ax=None,save=False):
        fig,ax=ut.newFig(xlabel=r'$\log_{10}(r_{part})$ [um]',ylabel='Irradiance [W/(m**2*um)]',reverse=[False,False],figsize=(10,7))
        ax.plot(self.wave,self.IrradStar,label='IrradStar (spectral irradiance at subsolar point)')
        if 'IrradStarTpProfCalc' in self.__dict__.keys():
            ax.plot(self.wave,self.IrradStarTpProfCalc,label='IrradStarTpProfCalc')
        ax.legend()
        if fig is not None:
            if save:
                fig.savefig(self.filebase+self.runName+'cloud_contour.pdf')
            return fig,ax

    def plotCarmaCloud(self,ax=None,save=False):
        #make heatmap/colourmap plot for n(r,p):

        for cond in self.conds:

            fig,ax=ut.newFig(xlabel=r'$\log_{10}(r_{part})$ [um]',ylabel='log10 Pressure [mbar]',reverse=[False,True],figsize=(10,7))

            x=np.log10(self.mieLUT[cond]['reff'])+6  #convert to micron for plotting
            y=np.log10(self.p)-2                     #convert to mbar for plotting
            z=self.npartConds[cond]*10**(-6)         #convert back to 1/cm**3 for plotting
            
            v = np.linspace(np.log10(np.max(z)*1e-5),np.log10(np.max(z)),100)
            cs=ax.contourf(x,y,np.log10(z),v)
            cbar = fig.colorbar(cs,ticks=ticker.MultipleLocator(0.5)) 
            cbar.set_label(r'$\log_{10}$ (Number density of particle [cm-3])')

            #v = np.logspace(np.log10(np.max(z)*1e-5),np.log10(np.max(z)),100) #specify colourbar range, "resolution", base, etc.
            #cs=ax.contourf(x,y,np.log10(z),v)  #,locator=ticker.LogLocator(base=10.0) 
            
            
            #plt.xlim([-4,2])
            ax.set_ylim([4,-4])
            #ut.ylognumbers(ax)
            
        if fig is not None:
            if save:
                fig.savefig(self.filebase+self.runName+'cloud_contour.pdf')
            return fig,ax

    def plotdppmLayer(self,ax=None,save=False,**kwargs):
        dppmLayer = self.r**2 / self.Rstar**2 * 1e6
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Transit Depth [ppm]')
            ax.set_ylabel('Pressure [bar]')
            #plt.axvline(x=self.Teq,color='r',linestyle='--',zorder=-10)
            ax.semilogy(dppmLayer,self.p/1e5,marker='x',**kwargs)  
            ax.axhline(1e-3,color='grey',linestyle='--',zorder=-10)
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_Tp.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs) 
            
            
    def plotRpLayer(self,ax=None,save=False,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel(r'Planet Radius [$R_\oplus$]')
            ax.set_ylabel('Pressure [bar]')
            ax.semilogy(self.r/Rearth,self.p/1e5,marker=None,**kwargs)  
            ax.axhline(1e-3,color='grey',linestyle='--',zorder=-10)
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_Tp.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs) 
            
            
    def plotOpacity(self,kappa,label=None,ax=None,save=False,vrange=None):
        x=self.wave
        y=self.p*1e-5
        z=np.log10(kappa)
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[16,10])
            ax.set_xlabel('Wavelength [um]')
            ax.set_ylabel('Pressure [bar]')
            #plt.axvline(x=self.Teq,color='r',linestyle='--')
            #ax.semilogy(self.T,self.p/1e5,marker='x',**kwargs)  
            
            ut.xspeclog(ax)
            ax.set_yscale('log')
            ax.set_ylim([1e-6,1e2])
            ax.invert_yaxis()
            ax.minorticks_on()
            
            xx,yy=np.meshgrid(x,y)
            if vrange is None:
                cs=ax.pcolormesh(xx, yy, z, cmap='jet', vmin=z[np.isfinite(z)].min(), vmax=z[np.isfinite(z)].max())
            else:
                cs=ax.pcolormesh(xx, yy, z, cmap='jet', vmin=vrange[0],vmax=vrange[1])
            fig.colorbar(cs) 

            if label is not None:
                ax.set_title(label)
            
            if save:
                fig.savefig(self.filebase+self.runName+'_Opacity.pdf')            
            return fig,ax



    def plotOpacityAtWave(self,waveToPlot=2.0,ax=None,rossMean=False,save=False,vrange=None):
        
        if rossMean:
            # ignoreKeys = ['tau','transmis','tauTransit','transmisTransit']
            ignoreKeys = ['tau','transmis','tauTransit','absorptionTransit']
        fig, ax = plt.subplots(figsize=[16,10])
        ax.set_xlabel('Opacity [1/m]')
        ax.set_ylabel('Pressure [bar]')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-6,1e4])
        ax.invert_yaxis()
        ax.minorticks_on()
        
        if rossMean:
            for key in self.opac.keys():
                if key in ignoreKeys:
                    continue
                elif type(self.opac[key]).__module__ == np.__name__:
                    # calculate Rosseland mean opacity
                    rossMeanOpa = np.zeros_like(self.p, dtype=self.numerical_precision)
                    for i in range(self.p.size):
                        rossMeanOpa[i] = rad.rosselandMean(self.wave,self.opac[key][i,:],self.T[i],waveUnit='um')
                    ax.plot(rossMeanOpa,self.p*1e-5,label=key)
            ax.set_title('RosselandMean')
            str_save = '_rossMean'
        else:
            waveInd=bisect(self.wave,waveToPlot)
            for key in self.opac.keys():
                if type(self.opac[key]).__module__ == np.__name__:
                    ax.plot(self.opac[key][:,waveInd],self.p*1e-5,label=key)
            ax.set_title(r'$\lambda$ [um]='+str(waveToPlot))
            str_save = ''
        ax.set_xlim([1e-30,1e1])
        ax.legend()
        
        if save:
            fig.savefig(self.filebase+self.runName+'_OpacityAtWave'+str(str_save)+'.pdf')            
        return fig,ax


    def plotOpacityContributions(self,save=False,vrange=None):
        fig, ax = plt.subplots(figsize=[16,10])
        ax.set_xlabel('Wavelength [microns]')
        ax.set_ylabel('Pressure [bar]')
        
        
        for key in self.opac.keys():
            if type(self.opac[key]).__module__ == np.__name__:
                ax.plot(self.wave,self.calcTransitSpectrum(self.opac[key]),label=key)
        ax.legend()
        ax.axhline(self.pressureToDppm(self.params['miePAtTau1']),ls='--',color='black')
        
        ax.set_ylim(ax.get_ylim())
        ax.axhline(self.RpBase**2/self.Rstar**2 *1e6,color='black')
        for zz in self.z:
            ax.axhline((self.RpBase+zz)**2/self.Rstar**2 *1e6,lw=0.1,ls='--',color='black')
        
        ut.makeSecYAxis(ax,(self.RpBase+self.z)**2/self.Rstar**2 *1e6,self.p/100,np.array([10**i for i in [5,4,3,2,1,0,-1,-2,-3]]),label='Pressure [mbar]')

        ut.xspeclog(ax)
    
    
    
        if save:
            fig.savefig(self.filebase+self.runName+'_OpacityContr.pdf')            
        return fig,ax


    def pressureToDppm(self,p):
        dppm=np.interp(p,self.p,(self.RpBase+self.z)**2/self.Rstar**2 *1e6)
        return dppm
    
    
    def plotAlbedoDisort(self):
        extinctCoef = 0.5*(self.extinctCoef[:self.nLay-1,:]+self.extinctCoef[1:self.nLay,:])
        scatCoef = 0.5*(self.scatCoef[:self.nLay-1,:]+self.scatCoef[1:self.nLay,:])
        
        #IrradStar = rad.PlanckFct(self.Teffstar,self.wave,'um','W/(m**2*um)','flux')*(self.Rstar/au)**2*(au/self.ap)**2
        IrradStar = np.ones(self.nWave, dtype=self.numerical_precision)
        scatAlb = np.arange(0,11)/10.0
        albedo = np.zeros([len(scatAlb)], dtype=self.numerical_precision)

        for omega in range(len(scatAlb)):
            Fupw = self.testDisort(IrradStar,extinctCoef,scatCoef, thermal = False, w0 = scatAlb[omega],wv=0)
            print(scatAlb[omega])
            albedo[omega] = Fupw#/IrradStar[5000]
            #albedo[m,omega] = 
        
        fig,ax = plt.subplots()
        ax.plot(scatAlb,albedo)
        ax.set_xlabel('Scattering albedo')
        ax.set_ylabel('Albedo')
        #ax.legend(loc=2,fontsize='xx-small')
        return fig,ax        

    
    #%% Temperature Correction Plotting & Debug
    def plotTpCorrected(self,ax=None,save=False,**kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            #plt.axvline(x=self.Teq,color='r',linestyle='--')
            ax.semilogy(self.T,self.p/1e5,marker='x',label='Finish',**kwargs)  

            T=np.interp(np.log(self.p),np.log(self.params['Tprof'][0]),self.params['Tprof'][1])
            ax.semilogy(T,self.p/1e5,marker='x',label='From Parameters',**kwargs)  
            
            ###
            ax.legend(loc=3,fontsize='xx-small')
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_TpCorrected.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs) 
            
    def plotTpChanges(self,ax=None,save=False,close=True,loop=None,**kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            plt.axvline(x=self.Teq,color='black',linestyle='--')
            for temp in range(len(self.TList[0])):
                ax.semilogy( self.TList[:,temp],self.p/1e5,**kwargs)  
            ax.semilogy( self.TList[:,temp],self.p/1e5,marker='x',**kwargs)
            ax.set_ylim([1e-10,1e4])
            #ax.set_xlim([self.TList.max*1.2])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                if loop is None:
                    fig.savefig(self.filebase+self.runName+'_TpChange.pdf')
                else:
                    fig.savefig(self.filebase+self.runName+'_TpChange'+str(loop)+'.pdf')
            if close:
                plt.close(fig)
            return fig,ax
        else:
            for temp in range(len(self.TList[0])):
                ax.semilogy( self.TList[:,temp],self.p/1e5,marker='x',label=temp,**kwargs)
   
    def plotTpConvec(self,ax=None,save=False,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            #plt.axvline(x=self.Teq,color='r',linestyle='--')
            ax.semilogy(self.T,self.p/1e5,marker='x',label='TP profile',**kwargs)  

#            T=np.interp(np.log(self.p),np.log(self.params['Tprof'][0]),self.params['Tprof'][1])
#            ax.semilogy(T,self.p/1e5,marker='x',label='From Parameters',**kwargs)  
            
            T_not = np.zeros([self.nLay], dtype=self.numerical_precision)
            T_not[-1] = self.T[-1]
#            T_not[:-1]=self.T[-1]*(self.p[:-1]/self.p[-1])**(2./7.)
            T_not[:-1]=self.T[-1]*(self.p[:-1]/self.p[-1])**(1- (1/(7./5.)))

                            
            ax.semilogy(T_not,self.p/1e5,marker='x',label="T = T(P')*(P/P')**2/7",**kwargs)  
            ax.legend(loc=3,fontsize='xx-small')
            ax.set_ylim([1e-12,1e4])
#            ax.set_xlim([self.T[0]-100,self.T[-1]+100])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_TpConvec.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs)  
    
    
    def plotTpGradient(self,ax=None,save=False,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Gradient')
            ax.set_ylabel('Pressure [bar]')
#            
#            grad = np.zeros([self.nLay])
#            lnT = np.log(self.T)
#            lnP = np.log(self.p)
#            for i in range(1,self.nLay):
#                grad[i]=(((lnT[i]-lnT[i-1])/(lnT[i]+lnT[i-1]))*
#                         ((lnP[i]+lnP[i-1])/(lnP[i]-lnP[i-1])))
#            grad[0] = grad[1]
#            
            
            
            grad = np.diff(np.log(self.T))/np.diff(np.log(self.p))
            
            ax.semilogy(grad,self.p[1:]/1e5,marker='x',label='TP Gradient',**kwargs)

            if self.modelSetting['TempType']=='NonGrayConv':
                if len(self.convLay) != 0:
                    p_not = self.p[1:]/1e5
                    ax.semilogy(grad[self.convLay-1],p_not[self.convLay-1],marker = 'x', color = 'r', label = 'Convective Layer(s)')
            

            ax.legend(loc=3,fontsize='xx-small')
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                fig.savefig(self.filebase+self.runName+'_TpGradient.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs)  
            
    def plotThievedData(self,ax=None,save=False,**kwargs):
        
        data = np.loadtxt('/Users/Jonathan/Downloads/Internet/PastedGraphic-3.txt',delimiter = ',')        
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs)  
            ax.semilogy(data[:,0],data[:,1],marker='x',**kwargs)
            ax.set_ylim([1e-12,1e4])
            ax.invert_yaxis()
            ax.minorticks_on()
            
            if save:
                fig.savefig(self.filebase+self.runName+'_TpThievedData.pdf')            
            return fig,ax
        else:
            ax.semilogy( self.T,self.p/1e5,marker='x',**kwargs) 
            
        return fig,ax
        

    #%% Instrument Response
    
    def prepInstrResp(self,specs, low_res_mode=0):  
        #prepare bandind and bandtra in specs for fast execution of self.instrResp()

        if low_res_mode:
            wave = self.wave_lowres
        else:
            wave = self.wave
            
        self.f=cLight/(wave*1e-6)   # Hz=1/s

        # #stellar flux at 1 parsec
        if self.fStarSurf is None:
            self.calcfStarSurf(self.params)
            
        # self.fStarSurf is already holding the expected spectrum (see runModel())
        self.thermalStar = self.fStarSurf
        if low_res_mode:
            self.thermalStar = self.thermalStar[::low_res_mode]
        
        
        for i,spec in enumerate(specs):
            if spec.meta['spectype'][:7] == 'highres':
                pass
            else:
                #Convert wavelength to um
                if spec['wave'].unit!='um':
                    spec=deepcopy(spec)
                    spec['wave']=['wave'].to('um')
                    spec['waveMin']=['waveMin'].to('um')
                    spec['waveMax']=['waveMax'].to('um')

                #prepare bandind and bandtra for fast execution of self.instrResp()
                spec['bandind']=[np.zeros(2,dtype=int) for i in range(len(spec))]
                spec['bandtra']=object  #ERROR HERE MIGHT INDICATE THAT THE SPECTRAL RESOLUTION OF MODEL IS INSUFFICIENT FOR INSTRUMENT RESPONSE
                for i,pnt in enumerate(spec):
                    if pnt['bandpass']=='uniform' or pnt['bandpass']=='Uniform':
                        pnt['bandind']=np.array([np.argmax(wave>pnt['waveMin']),np.argmax(wave>pnt['waveMax'])-1])
                        pnt['bandtra']=1.0
                    else:
                        #Read and normalize basspass information
                        fits_file       = self.datapath+'/bandpasses/'+pnt['bandpass']+'.fits'
                        f_open          = fits.open(fits_file)

                        #check that the wavelength is in Angstroms
                        if getattr(f_open[1].columns,'units')[0]!='ANGSTROMS':
                            raise ValueError('The first column of the response function does not have ANGSTROMS as units')

                        #get wavelength and transmission
                        wave_um         = f_open[1].data['WAVELENGTH']*1e-4
                        trans_norm      = f_open[1].data['THROUGHPUT']/np.max(f_open[1].data['THROUGHPUT'])
                        t               = Table([wave_um,trans_norm],names=('wave','transmission'))
                        t['normtrans']  = t['transmission']/np.max(t['transmission'])
                        t               = t[t['normtrans']>0.01]

                        #Set bandind and transmission function in that band
                        ind = np.arange(np.argmax(wave>t['wave'][0]),np.argmax(wave>t['wave'][-1])-1)
                        if ind.size>0:
                            #prevlastindex=ind[-1]
                            ind_large = np.r_[[ind[0]-1],ind,ind[1]+2]
                            wave_new  = wave[ind_large] #np.sort(np.array([pnt['waveMin']]+list(self.wave[ind])+[pnt['waveMax']]))
    #                    else :
    #                        ind_large = [prevlastindex, prevlastindex+1]
    #                        wave_new  = np
                        pnt['bandind'] = np.array([ind[0],ind[-1]])
                        pnt['bandtra'] = ut.interp1dEx(t['wave'].data,t['normtrans'].data,wave_new)

                        #Set limits for data point plotting (for display only)
                        t=t[t['normtrans']>0.2]
                        pnt['waveMin']=t['wave'][0]
                        pnt['waveMax']=t['wave'][-1]
                        pnt['wave']=0.5*(pnt['waveMin']+pnt['waveMax'])



    def instrResp(self,specs,spectypeInp=None,iLay=None,low_res_mode=False):
        '''Simulate instrument output'''

        astromodels=[]     
        for i,spec in enumerate(specs):

            if spectypeInp is None:
                spectype=deepcopy(spec.meta['spectype'])
            else:
                spectype=deepcopy(spectypeInp)

            if spectype=='dppm':
                self.nPhotPerSecM2(spec,star=True,misDurTransit=True,low_res_mode=low_res_mode)
                astromodel = spec['nPhotMisDurTransit'] / spec['nPhotStar'] * 1e6

            elif spectype=='secEclppm':
                self.nPhotPerSecM2(spec,star=True,planet=True)
                astromodel = 1.0 / (1.0 + spec['nPhotStar'] / spec['nPhotPlanet']) * 1e6

            elif spectype=='secEclppmTmod':
                self.nPhotPerSecM2(spec,star=True,planetTmod=True,iLay=iLay)
                astromodel = 1.0 / (1.0 + spec['nPhotStar'] / spec['nPhotPlanetTmod']) * 1e6  
 
            elif spectype=='thermal':  # W/(m**2*um) at planet surface weighted by stellar spectrum (equivalent to fitting eclipse depth)
                #thermal flux derived from secondary eclipse
                distance=10*parsec   #cancels out
                self.nPhotPerSecM2(spec,star=True,planet=True,distance=distance)
                astromodel = 2*hPlanck*cLight*spec['nPhotPlanet']*(distance/self.params['Rp'])**2  /  ( (spec['waveMax']*1e-6)**2 - (spec['waveMin']*1e-6)**2 ) * 1e-6  # W/(m**2*um)   that's why there is 1e-6
                             #this is the average thermal flux at TOA in (W/m**2*um) across the bin that would match the detected signal 
                             #necessary to account correctly for strong contrast across a given bin
                #spec['yerrLow'] = astromodel / spec['nPhotPlanet'] * np.sqrt(spec['nPhotStar'])
                #spec['yerrUpp']=spec['yerrLow']

#            elif spectype=='thermal':   # W/m**2 at planet surface (direct in flux integration across bin)
#                astromodel = ut.meanBin(self.wave,self.thermal,xbin=np.column_stack([spec['waveMin'],spec['waveMax']]))  #  W/m**2 = W/(m**2*um) * um

            elif spectype=='totalOutgoingFlux':  # W/(m**2) 
                astromodel=self.totalOutgoingFlux()

            elif spectype[:7] == 'highres':
                astromodel = np.ones_like(spec['waveMin'], dtype=self.numerical_precision)

            else:
                print('ERROR: Unknown spectype in pyspectrum.instrResp')
                print(spectype)
                pdb.set_trace()

            if spectypeInp is None:
                spec['ymodel']=astromodel
                    
            if np.any(np.isnan(astromodel)):
                print('WARNING: Instrument Response is NaN --> probably insufficient wavelength coverage modeled or resolution of model spectrum too low for observation')
                pdb.set_trace()
                
            astromodels.append(astromodel)
            
        return astromodels



    def simulateObs(self,spec,random=True):
        yval=self.instrResp([spec])[0]
        if random:        
            spec['yval']=np.random.normal(yval,spec['yerrLow']*np.float64(random)) 
        else:
            spec['yval']=yval

      
    def averageThermalWithBBcolumn(self,Tbb=2000,cloudFrac=0.5):
        blackBody = rad.PlanckFct(Tbb,self.wave,'um','W/(m**2*um)','flux')
        self.thermal2Col = self.thermal * (1-cloudFrac) + blackBody * cloudFrac


    def calcSecEclppm(self,thermalOnly=None,albedoOnly=None,thermalReflCombined=None):
        
        #Set defaults if not set by function call
        if thermalOnly is None:
            if 'thermalOnly' in self.modelSetting:
                if self.modelSetting['thermalOnly']!=[]:
                    thermalOnly=self.modelSetting['thermalOnly'][0]
            else:
                thermalOnly='old'

        if albedoOnly is None:
            if 'albedoOnly' in self.modelSetting:
                if self.modelSetting['albedoOnly']!=[]:
                    albedoOnly=self.modelSetting['albedoOnly'][0]
            
        if thermalReflCombined is None:
            if 'thermalReflCombined' in self.modelSetting:
                if self.modelSetting['thermalReflCombined']!=[]:
                    thermalReflCombined=self.modelSetting['thermalReflCombined'][0]
                
        #--thermalSecEclppm--------------------
        #Flux at stellar surface
        #self.fStarSurf is already computed when running the atmosphere model 
        
        # if 'fStarSurf' in self.params:
        #     self.fStarSurf = self.params['fStarSurf']
        # else:
        #     #print ('ASSUMING BLACK BODY FOR STAR')
        #     self.fStarSurf = rad.PlanckFct(self.Teffstar,self.wave,'um','W/(m**2*um)','flux')

        factorT = self.dppm / self.fStarSurf
        factorT_UniformRp=1e6 * (self.params['Rp']**2) / (self.fStarSurf * self.params['Rstar']**2)
        #factorT = self.dppm  / (rad.PlanckFct(self.params['Teffstar'],self.wave,'um' ,'W/(m**2*um)','flux'))
        #factorT_UniformRp=1e6 * (self.params['Rp']**2) / (rad.PlanckFct(self.params['Teffstar'],self.wave,'um' ,'W/(m**2*um)','flux')*self.params['Rstar']**2)
                    #        fig,ax=ut.newFig(xlabel='Wavelength [um]',ylabel='',log=[True,False],figsize=None)
                    #        ax.plot(self.wave,  1e6 * (self.params['Rp']**2/self.params['Rstar']**2) *  np.ones_like(self.wave) )
                    #        ax.plot(self.wave,  self.dppm)
                    #        fig,ax=ut.newFig(xlabel='Wavelength [um]',ylabel='',log=[True,False],figsize=None)
                    #        ax.plot(self.wave,factorT_UniformRp)
                    #        ax.plot(self.wave,factorT)

        if thermalOnly=='NoScat':
            self.thermalSecEclppm          = factorT           * self.thermalNoScat
            self.thermalSecEclppmUniformRp = factorT_UniformRp * self.thermalNoScat
        elif thermalOnly=='MuObs':
            self.thermalSecEclppm=factorT*self.thermalMuObs
        elif thermalOnly=='Toon':
            self.thermalSecEclppm=factorT*self.thermalToon
        elif thermalOnly=='Feautrier':
            self.thermalSecEclppm=factorT*self.thermalFeautrier
        elif thermalOnly=='old':
            self.thermalSecEclppm=factorT*self.thermal
        elif thermalOnly=='thermalNoScatMol':
            self.thermalSecEclppm=np.zeros_like(self.wave, dtype=self.numerical_precision)
        elif thermalOnly is None:
            self.thermalSecEclppm=np.zeros_like(self.wave, dtype=self.numerical_precision)
        else:
            print('Unknown thermalOnly: ' + thermalOnly)
            pdb.set_trace()
                    #            fig,ax=ut.newFig(xlabel='Wavelength [um]',ylabel='Eclipse Depth [ppm]',log=[True,False],figsize=None)
                    #            resPower=500
                    #            x,y=ut.convAtResPower(self.wave,self.thermalSecEclppmUniformRp,resPower); ax.plot(x,y)
                    #            x,y=ut.convAtResPower(self.wave,self.thermalSecEclppm         ,resPower); ax.plot(x,y)

        #--albedoSecEclppm-----------------------
            #factorA=1e6 * (self.params['Rp']/self.params['ap']  )**2
        factorA= (self.dppm/1e6 * self.params['Rstar']**2)  /  self.params['ap']**2    *1e6
        
        if albedoOnly=='Disort':
            self.albedoSecEclppm=factorA*self.albedoDisort
        elif albedoOnly=='Feautrier':
            self.albedoSecEclppm=factorA*self.albedoFeautrier
        elif albedoOnly=='Toon':
            self.albedoSecEclppm=factorA*self.albedoToon
        elif albedoOnly=='SetGeometricAlbedo':
            self.albedoSecEclppm=factorA*self.GeometricAlbedo
        elif albedoOnly is None:
            self.albedoSecEclppm=np.zeros_like(self.wave, dtype=self.numerical_precision)
        else:
            print('Unknown albedoOnly: ' + albedoOnly)
            pdb.set_trace()
        
        
        #secEclppm (combined)
        if thermalReflCombined=='Disort':
            self.secEclppm=factorT*self.totalFluxDisort  
        elif thermalReflCombined=='Feautrier':
            self.secEclppm=factorT*self.totalFluxFeautrier
        elif thermalReflCombined=='Toon':
            self.secEclppm=factorT*self.totalFluxToon
        elif thermalReflCombined is None:
            self.secEclppm = self.thermalSecEclppm+self.albedoSecEclppm
        else:
            print('Unknown thermalReflCombined: ' + thermalReflCombined)
            pdb.set_trace()


#
#    def nPhotPerSecM2_OLD(self,spec,star=True,planet=False,misDurTransit=False,planetTmod=False,iLay=None,
#                      distance=10*parsec):
#        # number of photons from star that a 1m2-telescope at 10 parsecs would collect per second within wavelength bin        
#
#        if star:
#            spec['nPhotStar']=0.0
#        if misDurTransit:
#            spec['nPhotMisDurTransit']=0.0
#        if planet:
#            spec['nPhotPlanet']=0.0
#        if planetTmod:
#            spec['nPhotPlanetTmod']=0.0
#        
#        for i,pnt in enumerate(spec):
#            try:
#                ind=np.arange(pnt['bandind'][0],pnt['bandind'][1])
#            except:
#                raise ValueError('ERROR: self.prepInstrResp() needs to be run first!') 
#            if star:
#                pnt['nPhotStar']          = np.trapz(x=self.wave[ind],y=self.thermalStar[ind]     /(hPlanck*self.f[ind])*pnt['bandtra']) * (self.params['Rstar']/distance)**2                         # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
#            if misDurTransit:
#                pnt['nPhotMisDurTransit'] = np.trapz(x=self.wave[ind],y=self.thermalStar[ind]     /(hPlanck*self.f[ind])*pnt['bandtra']  * (self.params['Rstar']/distance)**2  *self.dppm[ind]*1e-6)  # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
#            if planet:
#                pnt['nPhotPlanet']        = np.trapz(x=self.wave[ind],y=self.thermalStar[ind]     /(hPlanck*self.f[ind])*pnt['bandtra']  * (self.params['Rstar']/distance)**2  *self.secEclppm[ind]*1e-6)  # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
#                #pnt['nPhotPlanet']        = np.trapz(x=self.wave[ind],y=self.thermal[iLay,ind]    /(hPlanck*self.f[ind])*pnt['bandtra']) * (self.params['Rp']   /distance)**2                         # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
#            if planetTmod:
#                pnt['nPhotPlanetTmod']    = np.trapz(x=self.wave[ind],y=self.thermalTmod[iLay,ind]/(hPlanck*self.f[ind])*pnt['bandtra']) * (self.params['Rp']   /distance)**2                         # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
#                        
#            #if planet:
#            #    spec['nPhotPlanetFromEclipseMeas'] = spec['nPhotStar']*spec['yval']*1e-6    #photons/(s*m**2) 
#                
#  
    def calcfStarSurf(self,params):
        
        if 'fStarSurf' in params:
            if 'Lstar' in params:
                self.fStarSurf = params['fStarSurf']
                L = np.trapz(x=self.wave,y=self.fStarSurf*4*pi*self.Rstar**2)
                self.fStarSurf *= params['Lstar'] / L
                print ('Normalizing user-provided stellar spectrum (params["fStarSurf"]) to match params["Lstar"] (scaling factor = {})'.format(params['Lstar']/L))
            else:
                self.fStarSurf = params['fStarSurf']

        else:
            if 'Lstar' in params:
                #Create black body spectrum normalized to match Lstar
                self.fStarSurf = rad.PlanckFct(params['Teffstar'],self.wave,'um','W/(m**2*um)','flux')
                fStarSurfTemp = rad.PlanckFct(params['Teffstar'],np.linspace(0.3,100,len(self.wave)),'um','W/(m**2*um)','flux')
                L = np.trapz(x=np.linspace(0.3,100,len(self.wave)),y=fStarSurfTemp*4*pi*self.Rstar**2)
                self.fStarSurf *= params['Lstar'] / L
                print ('Normalizing black-body stellar spectrum to match params["Lstar"] (scaling factor = {})'.format(params['Lstar']/L))
            else:
                #Create black body spectrum simply with Teffstar
                self.fStarSurf = rad.PlanckFct(params['Teffstar'],self.wave,'um','W/(m**2*um)','flux')


    def nPhotPerSecM2(self,spec,star=True,planet=False,misDurTransit=False,planetTmod=False,starTOA=False,iLay=None,
                      distance=10*parsec, low_res_mode=False): 
        # number of photons from star that a 1m2-telescope at 10 parsecs would collect per second within wavelength bin        
        if low_res_mode:
            wave = self.wave_lowres
        else:
            wave = self.wave
            
        if spec[0]['bandind'].size>0:
            prevlastindex=spec[0]['bandind'][0]-1
        else :
            j=0
            while spec[j]['bandind'].size==0:
                j=j+1
            prevlastindex=spec[j]['bandind'][0]-1

        if star:
            spec['nPhotStar']=0.0
        if misDurTransit:
            spec['nPhotMisDurTransit']=0.0
        if planet:
            spec['nPhotPlanet']=0.0
        if planetTmod:
            spec['nPhotPlanetTmod']=0.0
        if starTOA:
            spec['nPhotStarTOA']=0.0

        for i,pnt in enumerate(spec):
            try:
                ind=np.arange(pnt['bandind'][0],pnt['bandind'][1]+1) 
            except:
                raise ValueError('ERROR: self.prepInstrResp() needs to be run first!') 

            if ind.size>0: 
                prevlastindex=ind[-1]
                ind_large = [pnt['bandind'][0]-1]+list(ind)+[pnt['bandind'][1]+1]
                wave_new  = np.sort(np.array([pnt['waveMin']]+list(wave[ind])+[pnt['waveMax']]))
            else : 
                ind_large = [prevlastindex, prevlastindex+1]
                wave_new  = np.array([pnt['waveMin'],pnt['waveMax']])

            f_with_interp   = np.interp(wave_new, wave[ind_large], self.f[ind_large])
            if star:
                thermalStar_with_interp     = np.interp(wave_new, wave[ind_large], self.thermalStar[ind_large])

                pnt['nPhotStar']            = np.trapz(x=wave_new,y=thermalStar_with_interp     /(hPlanck*f_with_interp)*pnt['bandtra']) * (self.params['Rstar']/distance)**2                         # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
            if misDurTransit:
                thermalStar_with_interp     = np.interp(wave_new, wave[ind_large], self.thermalStar[ind_large])
                dppm_with_interp            = np.interp(wave_new, wave[ind_large], self.dppm[ind_large])
                pnt['nPhotMisDurTransit']   = np.trapz(x=wave_new,y=thermalStar_with_interp     /(hPlanck*f_with_interp)*pnt['bandtra']  * (self.params['Rstar']/distance)**2  *dppm_with_interp*1e-6)  # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
            if planet:
                thermalStar_with_interp     = np.interp(wave_new, wave[ind_large], self.thermalStar[ind_large])
                secEclppm_with_interp       = np.interp(wave_new, wave[ind_large], self.secEclppm[ind_large])
                pnt['nPhotPlanet']          = np.trapz(x=wave_new,y=thermalStar_with_interp     /(hPlanck*f_with_interp)*pnt['bandtra']  * (self.params['Rstar']/distance)**2  *secEclppm_with_interp*1e-6)  # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
            if planetTmod:
                thermalTmod_with_interp     = np.interp(wave_new, wave[ind_large], self.thermalTmod[iLay,ind_large])
                pnt['nPhotPlanetTmod']      = np.trapz(x=wave_new,y=thermalTmod_with_interp     /(hPlanck*f_with_interp)*pnt['bandtra']) * (self.params['Rp']   /distance)**2                         # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
                
            if starTOA:
                thermalStar_with_interp     = np.interp(wave_new, wave[ind_large], self.thermalStar[ind_large])
                secEclppm_with_interp       = np.interp(wave_new, wave[ind_large], self.secEclppm[ind_large])
                dppm_with_interp            = np.interp(wave_new, wave[ind_large], self.dppm[ind_large])
                pnt['nPhotStarTOA']          = np.trapz(x=wave_new,y=thermalStar_with_interp     /(hPlanck*f_with_interp)*pnt['bandtra']  /(dppm_with_interp*1e-6))  # [x]=um, [y]=photons/(s*m**2*um) --> [nPhot]=photons/(s*m**2)
     
        

        
    def convertUncertaintyToThermal(self,spec,distance=10*parsec):

        self.nPhotPerSecM2(spec,star=True,planet=True,distance=distance)
        
        nphot=spec['nPhotStar']*spec['yval']*1e-6    #photons/(s*m**2) 
        thermal = 2*hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( (spec['waveMax']*1e-6)**2 - (spec['waveMin']*1e-6)**2 )   *  1e-6
        spec['yval']=thermal; spec['yval'].unit='W/(m**2*um)'

        nphot=spec['nPhotStar']*(spec['yval']-spec['yerrLow'])*1e-6    #photons/(s*m**2) 
        thermalLow = 2*hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( (spec['waveMax']*1e-6)**2 - (spec['waveMin']*1e-6)**2 )   *  1e-6
        spec['yerrLow']=thermal-thermalLow; spec['yerrLow'].unit='W/(m**2*um)'
        
        nphot=spec['nPhotStar']*(spec['yval']+spec['yerrUpp'])*1e-6    #photons/(s*m**2) 
        thermalUpp = 2*hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( (spec['waveMax']*1e-6)**2 - (spec['waveMin']*1e-6)**2 )   *  1e-6
        spec['yerrUpp']=thermalUpp-thermal; spec['yerrUpp'].unit='W/(m**2*um)'


            
    def convertSecEclppmToThermal(self,spec,distance=10*parsec):

        print('prepInstrResp')
        self.prepInstrResp([spec])
        self.instrResp([spec])

        self.nPhotPerSecM2(spec,star=True,planet=True,starTOA=True,distance=distance)
        
        specThermal = deepcopy(spec)
        specThermal.meta['spectype']='thermal'
        nphot=spec['nPhotStarTOA']*spec['yval']*1e-6   #photons/(s*m**2) 
        thermal = 2*hPlanck*cLight*nphot  /  ( ((spec['waveMax']*1e-6)**2- (spec['waveMin']*1e-6)**2))   *  1e-6
        #thermal =2*hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( ((spec['waveMax']*1e-6)**2- (spec['waveMin']*1e-6)**2))   *  1e-6
        #thermal = hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( spec['waveMax'] *1e-6* ((spec['waveMax']*1e-6)- (spec['waveMin']*1e-6)))   *  1e-6
        #thermal = hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( spec['waveMin']*(2700*(((1+1/2700)**99)-1)/100) *1e-6* ((spec['waveMax']*1e-6)- (spec['waveMin']*1e-6)))   *  1e-6
        #thermal = 2*hPlanck*cLight*nphot  * (distance/self.params['Rp'])**2  /  ( (spec['waveMax'] ))   *  1e-6
        specThermal['yval']=thermal; specThermal['yval'].unit='W/(m**2*um)'

        nphot=spec['nPhotStarTOA']*(spec['yval']-spec['yerrLow'])*1e-6    #photons/(s*m**2) 
        thermalLow = 2*hPlanck*cLight*nphot  /  ( (spec['waveMax']*1e-6)**2 - (spec['waveMin']*1e-6)**2 )   *  1e-6
        specThermal['yerrLow']=thermal-thermalLow; specThermal['yerrLow'].unit='W/(m**2*um)'
        
        nphot=spec['nPhotStarTOA']*(spec['yval']+spec['yerrUpp'])*1e-6    #photons/(s*m**2) 
        thermalUpp = 2*hPlanck*cLight*nphot  /  ( (spec['waveMax']*1e-6)**2 - (spec['waveMin']*1e-6)**2 )   *  1e-6
        specThermal['yerrUpp']=thermalUpp-thermal; specThermal['yerrUpp'].unit='W/(m**2*um)'
                
        return specThermal

    def convertThermalToTBright(self,spec):
        TBrightSpec = deepcopy(spec)
        TBrightSpec.meta['spectype']='Tbright'
        TBright=rad.calcTBright(spec['yval'],spec['wave'],'W/(m**2*um)', 'um')
        TBrightPlus=rad.calcTBright(spec['yval']+spec['yerrUpp'],spec['wave'],'W/(m**2*um)', 'um')
        TBrightMinus=rad.calcTBright(spec['yval']-spec['yerrLow'],spec['wave'],'W/(m**2*um)', 'um')
        TBrightSpec['yval']=TBright
        TBrightSpec['yerrLow']=TBright-TBrightMinus
        TBrightSpec['yerrUpp']=TBrightPlus-TBright

        return TBrightSpec
    
  
                        
            
    def thermalPhot(self,distance=10*parsec):
        'return thermal flux in photons/(s * m**2 * um)'
        return self.thermal / (hPlanck*self.f) * (self.params['Rp']/distance)**2  

    def contribution_function(self,specs):
        epsilon = np.finfo(float).eps
        T=copy.deepcopy(self.T)
        specsh=copy.deepcopy(specs)
        for i in range (0,len(self.T)):
            h=np.sqrt(epsilon)*T[i]*10**6

            Tph=T[i]+h
            dT=Tph-T[i]
            
            self.T[i]=Tph
            self.calcAtmosphere(self.modelSetting,self.params,updateTp=False)
            self.calcSecEclppm()
            self.instrResp(specsh)
            self.T[i]=T[i]
            contributiontemp=np.array([])
            for j in range(0,len(specs)):
                contributiontemp=np.append(contributiontemp,(specsh[j]['ymodel']-specs[j]['ymodel'])/dT)
            if i==0:
                contribution=contributiontemp
            else:
                contribution=np.vstack((contribution,contributiontemp))
                
        self.calcAtmosphere(self.modelSetting,self.params)
        self.calcSecEclppm()
        
        return contribution
    
    
    def niriss(self):
        resPowIn=self.resPower
        waveIn=self.wave
        depthIn=self.dppm
        
        iCut=np.searchsorted(waveIn,3.0)
        waveIn=waveIn[:iCut]
        depthIn=depthIn[:iCut]
        dWaveIn=waveIn/resPowIn
        
        #order 1
        dWaveO1=1.25/700
        #waveO1=np.arange(0.85,2.8,dWaveO1)
        fwhmPIX=dWaveO1/dWaveIn
        depthInConv=ut.gaussConv(depthIn,fwhmPIX)
        #depthO1=np.interp(waveO1,waveIn,depthInConv)

        #return waveO1,depthO1
        return waveIn,depthInConv
    
    
    #%% Instrument Plotting Routines        

    def plotInstrTrans(self,specs):
        fig,ax=ut.newFig(xlabel='Wavelength',ylabel='Transmission')        
        for i,spec in enumerate(specs):
            for pnt in spec:
                ind=np.arange(pnt['bandind'][0],pnt['bandind'][1])
                ax.plot(self.wave[ind],pnt['bandtra'],label=pnt['bandpass'])
        
                
    def plot_dmodeldT_instr(self,specs,showTp=True,spectypeInp=None):
        
        for i,spec in enumerate(specs):
            if spectypeInp is None:
                spectype=deepcopy(spec.meta['spectype'])
            else:
                spectype=deepcopy(spectypeInp)
            
            astromodel=self.instrResp([spec],spectype)[0]
            dmodeldT=np.zeros([self.nLay,len(spec)], dtype=self.numerical_precision)
            for iLay in range(self.nLay):
                dmodeldT[iLay,:] = self.instrResp([spec],spectypeInp=spectype+'Tmod',iLay=iLay) - astromodel
                
            fig,ax=ut.newFig(xlabel='d('+spectype+') / dT',ylabel='Pressure [mbar]',log=[False,True],reverse=[False,True], figsize=(6,7))        
            for ipnt,pnt in enumerate(spec):
                ax.plot(dmodeldT[:,ipnt],self.p/100,'x-',label='{:d}: {:s}'.format(ipnt,pnt['instrname']))
            ax.legend(loc='best',fontsize=8).draggable()
            
            if showTp:
                ax2=ax.twiny()
                ax2.plot(self.T,self.p/100,'rx-')

        return fig,ax,dmodeldT
            

    def plot_dmodeldT_model(self,ax=None,spectype='thermal',layers=None,save=False):
        if ax is None:
            fig, ax = ut.newFig(log=[False,True])
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('d('+spectype+') / dT')
        else:
            fig=None
            
        if layers is None:
            layers=np.linspace(0,self.nLay-1,9).astype(int)[1:-1]
            
        for iLay in layers:
            ax.plot(self.wave,self.thermalTmod[iLay,:]-self.thermal,label='{:d}: {:g} mbar'.format(iLay,self.p[iLay]/100))
        ax.legend(loc='best',fontsize=8).draggable()   

        if save:
            fig.savefig(self.filebase+self.runName+'_dthermaldT.pdf')            
        if fig is not None:
            return fig,ax


    def totalOutgoingFlux(self,integrateOverMuObs=False,waveBin=None,muInd=None):

        if integrateOverMuObs is False:
            if muInd is None:
                total = np.trapz(x=self.wave,y=self.thermal)   #W/m**2
            else:
                total = np.trapz(x=self.wave,y=self.thermalMuObs[muInd,:]*np.pi)   #W/m**2
        else:
            intensity = self.thermalMuObs[1:,:]    #intensity [12 muValues x 15000 wavelengths]
            muObs     = self.muObs[1:]           #mu [12 muValues]
            FupAtEachWave = -2*pi* np.trapz(x=muObs, y=muObs[:,np.newaxis]*intensity, axis=0)  #W/(m**2 * um)
            total = np.trapz(x=self.wave,y=FupAtEachWave)   #W/m**2
        
        if waveBin is None:
            return total
        else:
            ind = np.arange(bisect(self.wave,waveBin[0]),bisect(self.wave,waveBin[1]))
            totalInBin = np.trapz(x=self.wave[ind],y=self.thermal[ind])   #W/m**2          
            print(waveBin, totalInBin, '({:.2f})'.format(totalInBin/total*100))
            return 

        
    def checkOutgoingFlux(self):
        
        print('Incoming flux:                      {:f} W/m**2'.format(self.totalEffIncFlux))
        
        if 'Feautrier' in self.modelSetting['thermalReflCombined']:
            total = np.trapz(x=self.wave,y=self.totalFluxFeautrier)   #W/m**2  
            correspondBondAlbedo = 1 - (total / self.totalEffIncFlux) * (1-self.BondAlbedo)
            print('Total   flux up at TOA (Feautrier): {:f} W/m**2 (A_equi={:f})'.format(total, correspondBondAlbedo))
            
        if 'Feautrier' in self.modelSetting['thermalOnly']:
            total = np.trapz(x=self.wave,y=self.thermalFeautrier)   #W/m**2  
            correspondBondAlbedo = 1 - (total / self.totalEffIncFlux) * (1-self.BondAlbedo)
            print('Thermal flux up at TOA (Feautrier): {:f} W/m**2 (A_equi={:f})'.format(total, correspondBondAlbedo))
            
        total = self.totalOutgoingFlux()
        correspondBondAlbedo = 1 - (total / self.totalEffIncFlux) * (1-self.BondAlbedo)
        print('Thermal flux up at TOA (mu=0.577):  {:f} W/m**2 (A_equi={:f})'.format(total, correspondBondAlbedo))
        
        if 'MuObs' in self.modelSetting['thermalOnly']:
            total = self.totalOutgoingFlux(integrateOverMuObs=True)
            correspondBondAlbedo = 1 - (total / self.totalEffIncFlux) * (1-self.BondAlbedo)
            print('Thermal flux up at TOA (muIntegr):  {:f} W/m**2 (A_equi={:f})'.format(total, correspondBondAlbedo))
        
        self.EquivBondAlbedo=correspondBondAlbedo

        if self.wave[0]>1.0 or self.wave[-1]<5.0:
            print('WARNING: Insufficient wavelength coverage to compute totalOutgoingFlux!!!')
        return


    #%% Helper functions

    def calcMolInd(self,molName):
        for i,mol in enumerate(self.AbsMolNames):
            if mol == molName:
                return i
            
            
#%% Helper functions

def spectrum2instr(x,y,xmin,xmax):

    if xmin.size==1:
        imin=bisect_left(x,xmin)
        imax=bisect_right(x,xmax)
        yinstr=np.mean(y[imin:imax])
        
    else:
        n=len(xmin)    
        yinstr=np.zeros(n)    
        for i in range(n):
            imin=bisect_left(x,xmin[i])
            imax=bisect_right(x,xmax[i])
            yinstr[i]=np.mean(y[imin:imax])
            #yinstr[i]=np.trapz(y[imin:imax],x=x[imin:imax]) / (x[imax]-x[imin])
    
    return yinstr

        
def calcNewRadiusToMatchDppms(specs,astromodels,RpOld,Rstar):
    yi=np.array([])
    sigmai=np.array([])
    ym=np.array([])
    for i in range(len(specs)):
        yi=np.append(yi,specs[i]['yval'])
        sigmai=np.append(sigmai,specs[i]['yerrLow'])
        ym=np.append(ym,astromodels[i])
    dDppm = lambda r: (r**2 - RpOld**2) / Rstar**2  * 1e6
    fun = lambda r: np.sum(    ( (yi - (ym+dDppm(r))) / sigmai )**2 )
    #fun = lambda r: np.sum(  (   (yi   -   (ym + ((r**2 - RpOld**2) / Rstar**2)) ) / sigmai )**2  )
    
    RpNew = minimize(fun,RpOld,method='Nelder-Mead')#,options={'disp': True}) # ,options={'gtol': 1e-6, 'disp': True})
    RpNew =RpNew['x'][0]
    
#    print('\n\n')
#    fig,ax=plt.subplots()
#    rs=np.linspace(0.8*RpOld,1.2*RpOld,1000)
#    f=np.zeros_like(rs)
#    for i,r in enumerate(rs):  
#        f[i]=fun(r)
#    ax.plot(rs,f)
#    ax.plot(float(RpNew),fun(RpNew),'ro')
#    fig.savefig('funr.pdf')
#    if RpNew > 4.26e7:
    
    return float(RpNew), dDppm(RpNew), fun(RpNew)


def chi2AsFunctionOfRp(r,self,specsfitRadius):

   # print('DOING ADVANCED RADIUS OPTIMIZATION: Radius = {:g}'.format(r/Rearth))
    print('DOING ADVANCED RADIUS OPTIMIZATION: Radius = '+str(r/Rearth))
    self.params['Rp']=r

    yi=np.array([])
    sigmai=np.array([])
    for i in range(len(specsfitRadius)):
        yi=np.append(yi,specsfitRadius[i]['yval'])
        sigmai=np.append(sigmai,specsfitRadius[i]['yerrLow'])

    #Run Atmosphere Model
    self.calcAtmosphere(self.modelSetting,self.params,updateOnlyDppm=True,low_res_mode=self.fitRadiusLowRes)
    astromodels=self.instrResp(specsfitRadius,low_res_mode=self.fitRadiusLowRes)
#                    self.plotSpectrum(ax=self.ax,save=True,spectype='dppm',specs=specsfitRadius,resPower=500,xscale='log',presAxis=False,presLevels=False,label=str(counter))
#                    self.fig.savefig(self.filebase+self.runName+'_Spectrum_Test.pdf')            

    #Determine new Rp
    chi2 = np.sum(  ( (yi - astromodels)/ sigmai )**2 )
    
    return chi2
    
@jit(nopython=True, cache=True)
def slice_gt(array, lim):
	"""Funciton to replace values with upper or lower limit
	"""
	for i in range(array.shape[0]):
		new = array[i,:] 
		new[np.where(new>lim)] = lim
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
	A = np.zeros((L,nwno)) 
	B = np.zeros((L,nwno )) 
	C = np.zeros((L,nwno )) 
	D = np.zeros((L,nwno )) 

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
	AS, DS, CS, DS,XK = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l) # copy arrays

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



