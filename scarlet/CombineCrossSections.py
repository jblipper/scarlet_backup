#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:21:19 2018

@author: pelletier

Code to make the 'scarlet_LookUpQuickRead' tables used in atmosphere.py to calculate opacities
Works by simply combining the absorption cross sections of many molecules into 1 table.
Changed from the old file types to 

Choose:
 - the molecules you want to include (including from which data base (ExoMol/HITEMP) and at what resolution)
 - wavelength range (in microns)
 - resolution (resolution = 16 will pick 1 in every 16 data points from the full resolution files)

Input tables are currently (07/2018) found on the UdeM cluster maestria in the 'bbenneke/shared/07_CrossSectionLUTs/' folder
There tables for each molecules and they have the following variables:
'LookUpPGrid', the pressure at which the cross sections are computed, shape = (40,1)
'LookUpTGrid', the temperatures at which the cross sections are computed, shape = (6,1)
'nu', the wavenumber (cm^-1) at which the cross sections are computed, shape = (230261, 1)
'sigma_mol_i', the cross section at every nu, T and P, shape = (230261, 6, 40, 1)

Output table variables:
'LookUpPGrid', same as before
'LookUpTGrid', same as before
'nu', now only includes 1 in every 'resolution' elements and only covers the wavelength range specified
'sigma_mol', the cross section at every nu, T and P, and for each molecule, shape = (6, 40, ##, 15)  ---- here ## will depend on your wavelength coverage and resolution
'MolNames', the name of all the molecules, in the order they are in sigma_mol
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import h5py
import os.path
import datetime
from bisect import bisect
import matplotlib.pyplot as plt
import glob
import pdb
import re

# set relative path to go to where the opacity files are (maestria/bbenneke/shared/07_CrossSectionLUTs/)
Input_directory = '../../../../../bbenneke/shared/07_CrossSectionLUTs/' 
# set relative path to your scarlet_LookUpQuickRead folder   
Output_directory = '../../scarlet_LookUpQuickRead/'


# choose the range and resolution you want
Wave_range = [0.5, 3.0] # microns, min = 0.2, max = 20
resolution = 16 # 128 means 1 in every 128 element is kept

########### available setting options, normally stick with 'Res250K_ExoMol_CH4' ##############
#LUTList = 'Res5e4'
#LUTList = 'Res250K_ExoMol_CH4'
LUTList = 'Res250K_ExoMol_H2OCH4'
#LUTList = 'Res1e6'
#LUTList = 'Res1e6ExoMol'


print(LUTList)
print('Wave Range = {:s}'.format(Wave_range))
print('Resolution = {:s}'.format(resolution))
# PreMade_Molecules come from pre-made opacity tables (mostly HITEMP molecules at 250k resolution)
PreMade_Molecules = {}
# Other_Molecules are everything else, mostly ExoMol molecules but also H- which comes from an analytical formula, and O2 which comes from a 50k resolution file
Other_Molecules = {}


# copy of the old matlab format.  Not ideal.
if LUTList[0:7] == 'Res250K':
    if LUTList == 'Res250K_ExoMol_CH4': # i.e. H2O from HITEMP
        PreMade_Molecules['H2O']   = 'CrossSectionLUT_20140809_00h07m_01_H2O_0_Mix_Res250K.mat'
    PreMade_Molecules['CO']     = 'CrossSectionLUT_20140809_00h07m_05_CO_0_Mix_Res250K.mat'
    PreMade_Molecules['C2H2']   = 'CrossSectionLUT_20150527_16h37m_26_C2H2_0_Mix.mat'
    PreMade_Molecules['OH']     = 'CrossSectionLUT_20150608_15h52m_13_OH_0_Mix.mat'
    PreMade_Molecules['H2S']    = 'CrossSectionLUT_20150527_16h37m_31_H2S_0_Mix.mat'
    PreMade_Molecules['PH3']    = 'CrossSectionLUT_20150527_16h37m_28_PH3_0_Mix.mat'
    PreMade_Molecules['K']      = 'CrossSectionLUT_RES250K_20150610_15h22m43s_K.mat'
    PreMade_Molecules['Na']     = 'CrossSectionLUT_RES250K_20150610_15h22m43s_Na.mat'
    # ExoMol molecules
    if LUTList == 'Res250K_ExoMol_H2OCH4': # H2O from ExoMol
        Other_Molecules['H2O'] = 'ExoMol_Files/H2O/'
    Other_Molecules['VO']  = 'ExoMol_Files/VO/'
    Other_Molecules['CO2'] = 'ExoMol_Files/CO2/'
    Other_Molecules['CH4'] = 'ExoMol_Files/CH4/'
    Other_Molecules['FeH'] = 'ExoMol_Files/FeH/'
    Other_Molecules['HDO'] = 'ExoMol_Files/HDO/'
    Other_Molecules['HCN'] = 'ExoMol_Files/HCN/'
    Other_Molecules['NH3'] = 'ExoMol_Files/NH3/'
    Other_Molecules['TiO'] = 'ExoMol_Files/TiO/'
    Other_Molecules['SiO'] = 'ExoMol_Files/SiO/'
    Other_Molecules['H-']  = 'H-'
    Other_Molecules['O2']  = 'CrossSectionLUT_20130205_15h07m_O2.mat' # O2 is actually from HITEMP, but the table is different than the others
    

elif LUTList == 'Res5e4':
    PreMade_Molecules['H2O']  = 'CrossSectionLUT_20130213_10h20m_H2O.mat'
    PreMade_Molecules['CO2']  = 'CrossSectionLUT_20130213_10h20m_CO2.mat'
    PreMade_Molecules['O3']   = 'CrossSectionLUT_20130205_15h07m_O3.mat'
    PreMade_Molecules['N2O']  = 'CrossSectionLUT_20130205_15h07m_N2O.mat'
    PreMade_Molecules['CO']   = 'CrossSectionLUT_20130213_10h20m_H2O.mat'
    PreMade_Molecules['CH4']  = 'CrossSectionLUT_20130205_15h07m_CH4_inclKarkoschka2010.mat'
    PreMade_Molecules['O2']   = 'CrossSectionLUT_20130205_15h07m_O2.mat'
    PreMade_Molecules['SO2']  = 'CrossSectionLUT_20130205_15h07m_SO2.mat'
    PreMade_Molecules['NO2']  = 'CrossSectionLUT_20130205_15h07m_NO2.mat'
    PreMade_Molecules['NH3']  = 'CrossSectionLUT_20130819_20h48m_NH3.mat'
    PreMade_Molecules['C2H2'] = 'CrossSectionLUT_20130205_15h07m_C2H2.mat'
    PreMade_Molecules['K']    = 'CrossSectionLUT_Res5e4_20140808_16h12m35s_K.mat'
    PreMade_Molecules['Na']   = 'CrossSectionLUT_Res5e4_20140808_16h12m35s_Na.mat'

elif LUTList == 'Res1e6':
    PreMade_Molecules['H2O']   = 'CrossSectionLUT_20140809_00h07m_01_H2O_0_Mix.mat'
    PreMade_Molecules['CO']    = 'CrossSectionLUT_20140809_00h07m_05_CO_0_Mix.mat'
    PreMade_Molecules['CH4']   = 'CrossSectionLUT_20140809_00h07m_06_CH4_0_Mix.mat'
    
elif LUTList == 'Res1e6ExoMol':
    PreMade_Molecules['H2O']    = 'CrossSectionLUT_ExoMol_NoPresDep_20140813_15h37m38s_01_H2O_1_1H2-16O.mat'
    PreMade_Molecules['CO2']    = 'CrossSectionLUT_20131227_19h38m_02_CO2_0_Mix.mat'
    PreMade_Molecules['CO']     = 'CrossSectionLUT_20140809_00h07m_05_CO_0_Mix.mat'
    PreMade_Molecules['CH4']    = 'CrossSectionLUT_ExoMol_NoPresDep_20140813_15h37m38s_06_CH4_1_12C-1H4.mat'
    PreMade_Molecules['SO2']    = 'CrossSectionLUT_20131227_19h38m_09_SO2_0_Mix.mat'
    PreMade_Molecules['NO2']    = 'CrossSectionLUT_20131227_19h38m_10_NO2_0_Mix.mat'
    PreMade_Molecules['NH3']    = 'CrossSectionLUT_20131227_19h38m_11_NH3_0_Mix.mat'
    PreMade_Molecules['C2H2']   = 'CrossSectionLUT_20131227_19h38m_26_C2H2_0_Mix.mat'
    PreMade_Molecules['TiO']    = 'CrossSectionLUT_ExoMol_NoPresDep_20140813_15h37m38s_06_TiO_1_48Ti.mat'


# all molecules without HDO, O2 and maybe H2O, those are added later
#molecules = [mole for mole in MoleculeFiles]

now = datetime.datetime.now()
# create the Table in which the information of all the molecules will be stacked in.  Stores info in this file 
Table = h5py.File(Output_directory + 'LookUpQuickRead_MoleculesAll_HITEMP_{:s}_{:s}{:s}{:s}_{:s}_{:s}_{:s}.mat'.format(LUTList, now.year, format(now.month, "02"), format(now.day, "02"), Wave_range[0], Wave_range[-1], resolution), "w")

#%%  H- bound-free and free-free cross sections
C1 = 152.519
C2 = 49.534
C3 = -118.858
C4 = 92.536
C5 = -34.194
C6 = 4.982
wave0 = 1.6419
alpha = 1.439e4 # alpha = hc/k_b = 1.439 cmK = 1.439e4 umK

def f_Hminus(wave):
    return C1 + C2*(1./wave + 1./wave0)**0.5 + C3*(1./wave + 1./wave0)**1. + C4*(1./wave + 1./wave0)**1.5 + C5*(1./wave + 1./wave0)**2. + C6*(1./wave + 1./wave0)**2.5                               

# John 1988, equation 4
def sigma_bound_free_Hminus(wave):
    sigma = np.zeros(np.array(wave).size)
    indices = np.where(wave<wave0)[0]
    sigma[indices] = (1e-18*wave[indices]**3)*((1./wave[indices] - 1./wave0)**1.5)*f_Hminus(wave[indices])
    return sigma
# in cm^2
    
A1_short = 518.1021
B1_short = -734.8666
C1_short = 1021.1775
D1_short = -479.0721
E1_short = 93.1373
F1_short = -6.4285

A2_short = 473.2636
B2_short = 1443.4137
C2_short = -1977.3395
D2_short = 922.3575
E2_short = -178.9275
F2_short = 12.36

A3_short = -482.2089
B3_short = -737.1616
C3_short = 1096.8827
D3_short = -521.1341
E3_short = 101.7963
F3_short = -7.0571

A4_short = 115.5291
B4_short = 169.6374
C4_short = -245.649
D4_short = 114.243
E4_short = -21.9972
F4_short = 1.5097



A2_long = 2483.345
B2_long = 285.827
C2_long = -2054.291
D2_long = 2827.776
E2_long = -1341.537
F2_long = 208.952

A3_long = -3449.889
B3_long = -1158.382
C3_long = 8746.523
D3_long = -11485.632
E3_long = 5303.609
F3_long = -812.939

A4_long = 2200.04
B4_long = 2427.719
C4_long = -13651.105
D4_long = 16755.524
E4_long = -7510.494
F4_long = 1132.738

A5_long = -696.271
B5_long = -1841.4
C5_long = 8624.97
D5_long = -10051.53
E5_long = 4400.067
F5_long = -655.02

A6_long = 88.283
B6_long = 444.517
C6_long = -1863.864
D6_long = 2095.288
E6_long = -901.788
F6_long = 132.985


# John 1988
def kff_short(wave, T):
    return 1e-29*( (5040.0/T)*(A1_short*wave**2 +B1_short + C1_short/wave + D1_short/wave**2 + E1_short/wave**3 + F1_short/wave**4) + ((5040.0/T)**1.5)*(A2_short*wave**2 +B2_short + C2_short/wave + D2_short/wave**2 + E2_short/wave**3 + F2_short/wave**4) + ((5040.0/T)**2)*(A3_short*wave**2 +B2_short + C3_short/wave + D3_short/wave**2 + E3_short/wave**3 + F3_short/wave**4) + ((5040.0/T)**2.5)*(A4_short*wave**2 +B4_short + C4_short/wave + D4_short/wave**2 + E4_short/wave**3 + F4_short/wave**4)  )

def kff_long(wave, T):
    return 1e-29*( ((5040.0/T)**1.5)*(A2_long*wave**2 +B2_long + C2_long/wave + D2_long/wave**2 + E2_long/wave**3 + F2_long/wave**4) + ((5040.0/T)**2)*(A3_long*wave**2 +B2_long + C3_long/wave + D3_long/wave**2 + E3_long/wave**3 + F3_long/wave**4) + ((5040.0/T)**2.5)*(A4_long*wave**2 +B4_long + C4_long/wave + D4_long/wave**2 + E4_long/wave**3 + F4_long/wave**4) + ((5040.0/T)**3)*(A5_long*wave**2 +B5_long + C5_long/wave + D5_long/wave**2 + E5_long/wave**3 + F5_long/wave**4) + ((5040.0/T)**3.5)*(A6_long*wave**2 +B6_long + C6_long/wave + D6_long/wave**2 + E6_long/wave**3 + F6_long/wave**4)      )

def kff(wave, T):
    k_ff = np.zeros(wave.size)
    short_ind = np.where(wave < 0.3645)[0]
    long_ind = np.where(wave >= 0.3645)[0]
    k_ff[short_ind] = kff_short(wave[short_ind], T)
    k_ff[long_ind] = kff_long(wave[long_ind], T)
    return k_ff                /(0.75*(T**-2.5)*np.exp(alpha/(wave0*T)))  # this division term comes from the bound-free absorption coefficient!!  Not sure how to convert to a cross-section otherwise
# in cm^2

def Sigma_Hminus(wave, T):
    sigma_bf = sigma_bound_free_Hminus(wave)
    sigma_ff = kff(wave, T)
    return (sigma_bf + sigma_ff)/1e4 # convert to m^2



#%%
# function to sort a list a strings into order in increasing number
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#temperatures now go up to 4000K.  sets high temp values to 2100K for all files other than H- and ExoMol H2O (and HDO goes up to 3000K)
Temperatures = np.array([250.0, 400.0, 700.0, 1000.0, 1500.0, 2100.0, 2700.0, 3300.0])

# Make a dictionary containing all the molecules, mostly just used to keep track of how many there are.
MoleculeFiles = PreMade_Molecules.copy()  
MoleculeFiles.update(Other_Molecules)

MolNames = []
# loop over HITEMP molecules
print('Combining Molecule Cross-Sections')
for i, HITEMP_molecule in enumerate(PreMade_Molecules):
    print(HITEMP_molecule)
    print(PreMade_Molecules[HITEMP_molecule])
    if os.path.isfile(Input_directory + PreMade_Molecules[HITEMP_molecule]):
        f = h5py.File(Input_directory + PreMade_Molecules[HITEMP_molecule]) # open file of molecule
    else:
        raise Exception('File for molecule does not exist.  Check directory.')
    # convert from wavenumber (1/cm) to wavelength in microns
    wave = 1e4/np.array(f['nu']).flatten()[::-1]
    if i == 0.0: # stuff that only needs to be done once:
        # check if requested wavelength range is outside data range
        if Wave_range[0] < wave.min() or Wave_range[-1] > wave.max():
            raise Exception('Requested wavelengths go outside data wavelength range') 
        # find the indices corresponding to the desired wavenumber/wavelength range
        range_indices = np.arange(bisect(wave, Wave_range[0]), bisect(wave, Wave_range[-1]))
        # pick only 1 in every resolution elements
        coarse_indices = range_indices[0::int(resolution)]
        # append the last point at the end to make sure range is included.  To avoid extrapolation.
        coarse_indices = np.append(coarse_indices, range_indices[-1])
        coarse_wave = wave[coarse_indices]
        # write info that is the same for all files into table, this is done only once 
        Wave_microns = Table.create_dataset("Wave_microns", coarse_wave.shape)
        Wave_microns[...] = coarse_wave
        LookUpTGrid = Table.create_dataset("LookUpTGrid", Temperatures.shape)
        LookUpTGrid[...] = Temperatures#f['LookUpTGrid'][:].flatten()
        LookUpPGrid = Table.create_dataset("LookUpPGrid", f['LookUpPGrid'][:].flatten().shape)
        LookUpPGrid[...] = f['LookUpPGrid'][:].flatten()
        all_css = np.zeros([LookUpTGrid[:].size, LookUpPGrid[:].size, Wave_microns[:].size, len(MoleculeFiles)])
        dt = h5py.special_dtype(vlen=str)
        LookUpMolNames = Table.create_dataset('LookUpMolNames', (len(MoleculeFiles),), dtype=dt)
        
    # inverse the order of the cross sections
    css = f['sigma_mol_i'][:][::-1,:,:,0][coarse_indices]
    # shift the order of the axes to be how scarlet expects them
    css = np.moveaxis(css, 0, -1)
    # set 250.0, 400.0, 700.0, 1000.0, 1500.0, 2100.0 equal to the values in the tables
    all_css[:6,:,:,i] = css
    # now make the temperatures 2700K and 3300K equal to those of 2100K
    all_css[6:,:,:,i] = np.tile(css[-1,:,:],[2,1,1])
    MolNames.append(HITEMP_molecule)
    f.close()        
        
# now do the ExoMol molecules
for j, Other_molecule in enumerate(Other_Molecules):
    print(Other_molecule)
    print(Other_Molecules[Other_molecule])
    # H- is unique, comes from the equations above.  Taken from 
    if Other_molecule == 'H-':
        print('Adding H-')
        Hminus_cross_sections = np.zeros_like(all_css[:,:,:,0], dtype = float)
        for T_ind, temp in enumerate(Temperatures):
            Hminus_css = Sigma_Hminus(coarse_wave, temp)
            for P_ind in range(LookUpPGrid.size):
                Hminus_cross_sections[T_ind,P_ind,:] = Hminus_css
        all_css[:,:,:,i+1+j] = Hminus_cross_sections
    # O2 seems to only be available in 50k resolution, so interpolate it onto the desired grid.  Note that before O2 was zero everywhre
    elif Other_molecule == 'O2':
        print('Adding O2')
        f_O2 = h5py.File(Input_directory + MoleculeFiles['O2'])
        O2_wave = 1e4/f_O2['nu'][:].flatten()[::-1]            
        # O2 at temperatures 2700 and 3300 is set to zero
        O2_cs = np.moveaxis(f_O2['sigma_mol_i'][:][::-1,:,:,0], 0, -1)
        O2_cross_sections = np.zeros_like(all_css[:,:,:,0], dtype = float)
        # O2 should have a pressure dependence 
        for T_ind in range(LookUpTGrid[:6].size): # only do up to 2100K for O2, since that is what is available
            for P_ind in range(LookUpPGrid.size):
                # O2 is not log interpolated bacause it has zeros in it
                O2_cross_sections[T_ind,P_ind,:] = np.interp(coarse_wave, O2_wave, O2_cs[T_ind,P_ind,:] ,  left = 0.0, right = 0.0 )
        O2_cross_sections[T_ind:,:,:] = O2_cross_sections[T_ind,:,:] # set 2700K and 3300K to the values of 2100K
        all_css[:,:,:,i+1+j] = O2_cross_sections
        f_O2.close() 
    else:
        # go into the molecule folder (ExoMol_Files/molecule/) and search for all files with sigma in them (these are the default output files frm the ExoMol website) 
        mol_Files = glob.glob(Input_directory + Other_Molecules[Other_molecule]+ '*sigma*')
        # sort them in increassing order (the smallest should be 296K, the highest varies from molecule to molecule)
        sorted_Files = natural_sort(mol_Files)
        # create an array to store the cross-sections at each temperature. This will be missing some dimensions if not all the temperatures are available 
        mol_css = np.zeros([len(Temperatures), coarse_wave.size])
        for k, temp in enumerate(sorted_Files):
            # extract wavelength (microns) - convert from wavenumber (cm^-1) and invert order
            file_T = np.transpose(np.genfromtxt(temp))
            mol_wave = 1e4/file_T[0][::-1]
            # pick only the wavelength range that you want
            mol_wave_indices = np.arange(bisect(mol_wave, Wave_range[0]), bisect(mol_wave, Wave_range[-1]))
            wave_cut = mol_wave[mol_wave_indices]
            # pick the cross-sections at those wavelengths
            cs = file_T[1][::-1][mol_wave_indices]
            # if there are no zeros use log interpolation
            if np.min(cs) > 0.0:
                log_cs = np.log10(cs/1e4) # convert from cm^2 to m^2
                logcoarse_cs = np.interp(coarse_wave, wave_cut, log_cs,  left = -1e20, right = -1e20) # set edges to -1e20 to that 10**-1e20 gives 0.0
                coarse_cs = 10**logcoarse_cs
            else: # use linear interpolation
                coarse_cs = np.interp(coarse_wave, wave_cut, cs/1e4,  left = 0.0, right = 0.0)
            if np.any(np.isnan(coarse_cs)):
                raise Exception('Warning: NaN detected, not good')
            mol_css[k,:] = coarse_cs  
        # if temperatures don't go to 3300K, set all higher temperatures to the highest one available
        if k < len(Temperatures)-1:
            print('High temperature data unavailable for {:s}, setting high temperature values to those of {:s}K'.format(Other_molecule, Temperatures[k]) )
            mol_css[k:,:] = coarse_cs
        # HITEMP filled up until i
        all_css[:,:,:,i+1+j] = np.moveaxis(np.tile(mol_css, [LookUpPGrid.size,1,1]),0,1)
    MolNames.append(Other_molecule)
    
LookUpMolNames[...] = MolNames
sigma_mol = Table.create_dataset("sigma_mol", all_css.shape)
sigma_mol[...] = all_css

print('Finished!')



#css = CO['sigma_mol_i'][:][::-1,:,:,0][coarse_indices]
# inverse the direction of the cross-sections as is done with the wavelength
#css = css[coarse_indices]
# shift the order of the axes to be how scarlet expects them
#css = np.moveaxis(css, 0, -1)
#all_css[:,:,:,i] = css




#ax.plot(coarse_wave, css[:,0,0], label = '%s' % resolution )






#g = h5py.File('/Users/pelletier/Research/GitHub/scarlet_LookUpQuickRead/LookUpQuickRead_MoleculesAll_HITEMP_Res250K_ExoMol_CH4_20150610_0.4_6_128.mat')
##
##O2 = h5py.File('/Users/pelletier/Research/GitHub/Opacity_Tables/CrossSectionLUT_20130205_15h07m_O2.mat')
##CO = h5py.File('/Users/pelletier/Research/GitHub/Opacity_Tables/CrossSectionLUT_20130205_15h07m_O2.mat')
#
##print(CO.keys())
#
#
#LookUpMolNames = []
#for iMol in range(len(g['MolNames'])):
#    LookUpMolNames.append(''.join(chr(i) for i in g[g['MolNames'][iMol][0]][:]))
#LookUpMolNames=np.array(LookUpMolNames)
#LookUpAbsMol=np.array(g['AbsMol']).flatten().astype(int)-1
#LookUpMolNames=LookUpMolNames[LookUpAbsMol]


#%%

#new = 'LookUpQuickRead_MoleculesAll_Res250K_ExoMol_CH4_2018724_0.5_1_1_v2.mat'
#old = 'LookUpQuickRead_MoleculesAll_HITEMP_Res250K_ExoMol_CH4_20150610_0.5_1_1.mat'
#
##new_file = h5py.File(new)
#old_file = h5py.File(old)
#
#old_file['nu']
##new_file['Wave_microns']
#
#
#
#LookUpMolNames = []
#for iMol in range(len(old_file['MolNames'])):
#    LookUpMolNames.append(''.join(chr(i) for i in old_file[old_file['MolNames'][iMol][0]][:]))
#LookUpMolNames=np.array(LookUpMolNames)
#LookUpAbsMol=np.array(old_file['AbsMol']).flatten().astype(int)-1
#LookUpMolNames=LookUpMolNames[LookUpAbsMol]
#
#AbsMolNames=np.array(['CH4','C2H2','O2','OH','H2O','CO','CO2','NH3','HCN','H2S','PH3','Na','K'])
#
##for iMol,MolName in enumerate(AbsMolNames):
##    ind=np.where(MolName==LookUpMolNames)[0][0]
##    print(MolName)
##    print(ind)
#
#
#ind_old = np.where('CO2'==LookUpMolNames)[0][0]
##ind_new = np.where('CO2'==new_file['LookUpMolNames'][:])[0][0]
##
#CO = h5py.File('/Users/pelletier/Research/GitHub/Opacity_Tables/CrossSectionLUT_20130205_15h07m_CO.mat')
##
##fig,ax = plt.subplots()
##ax.plot(1e4/CO['nu'][:].flatten(), CO['sigma_mol_i'][:][:,0,0,0] )
##ax.set_xlim([0.4,6])
##
#fig,ax = plt.subplots()
##ax.plot(new_file['Wave_microns'][:], new_file['sigma_mol'][:][0,0,:, 1 ] )
#ax.plot(1e4/old_file['nu'][:], old_file['sigma_mol'][:][0,0,:, 7 ] )
##
##
##
##fig,ax = plt.subplots()
#ax.plot(1e4/CO['nu'][:].flatten(), CO['sigma_mol_i'][:][:,0,0,0], color = 'k', label = 'Full' )
#ax.plot(Table['Wave_microns'], Table['sigma_mol'][:][0,0,:,0], label = 'Table' )


#%%

#resolution = 1

#CO = h5py.File('/Users/pelletier/Research/GitHub/Opacity_Tables/CrossSectionLUT_20130205_15h07m_CO.mat')




#wave = 1e4/np.array(CO['nu']).flatten()[::-1]
## find the indices corresponding to the desired wavenumber/wavelength range
#range_indices = np.arange(bisect(wave, Wave_range[0]), bisect(wave, Wave_range[-1]))
#range_indices = np.where((wave >= Wave_range[0]) & (wave <= Wave_range[-1]))[0]
## pick only 1 in every resolution elements
#coarse_indices = range_indices[0::int(resolution)]
## append the last point at the end to make sure range is included.  To avoid extrapolation.
#coarse_indices = np.append(coarse_indices, range_indices[-1])
#coarse_wave = wave[coarse_indices]
## now stacking all the cross sections
#
#css = CO['sigma_mol_i'][:][::-1,:,:,0][coarse_indices]
# inverse the direction of the cross-sections as is done with the wavelength
#css = css[coarse_indices]
# shift the order of the axes to be how scarlet expects them
#css = np.moveaxis(css, 0, -1)
#all_css[:,:,:,i] = css




#ax.plot(coarse_wave, css[:,0,0], label = '%s' % resolution )
#
##ax.set_xlim([0.4,6])
#
##ax.set_title('Resolution = %s' % resolution)
#ax.set_xlabel('Wavelength (microns)')
#ax.set_ylabel('Absorption Cross-Section')
#ax.legend()







#%% Comparison of code vs scarlet files





#a1 = np.transpose(np.genfromtxt('/Users/pelletier/Research/GitHub/Opacity_Tables/untitled folder/1H2-16O_0-30000_400K_0.010000.sigma.txt'))
#a2 = np.transpose(np.genfromtxt('/Users/pelletier/Research/GitHub/Opacity_Tables/untitled folder/1H2-16O_0-30000_2100K_0.010000.sigma.txt'))
#a3 = np.transpose(np.genfromtxt('/Users/pelletier/Research/GitHub/Opacity_Tables/untitled folder/1H2-16O_0-30000_4000K_0.010000.sigma.txt'))
#
#
##%%
#
#
#
#plt.plot(1e4/a1[0], a1[1], label='400K' )
#plt.plot(1e4/a2[0], a2[1], label='2100K' )
#plt.plot(1e4/a3[0], a3[1], label='4000K' )
#
#plt.title('H2O')
#
#plt.xlim([1,10])
#
#plt.xlabel('Wavelength (microns)')
#plt.ylabel('Cross-Section')
#
#plt.legend()







