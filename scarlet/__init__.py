# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import pkg_resources

from .atmosphere import atmosphere
from .modelcomparison import calcBayesFactors

from .plotOpacities import opacityPlotter


from .retrieval import retrieval
from .retrieval import fitparam

from scarlet import lcmodels

datapath = pkg_resources.resource_filename('scarlet', '../data/')



def loadAtm(filename):
    atm = atmosphere(name='empty')
    atm.load(filename)
    #atm.filebase = os.path.dirname(filename) + '/' + atm.filename + '_'
    
    atm.scarletpath = os.path.dirname(pkg_resources.resource_filename('scarlet', ''))
    atm.datapath = atm.scarletpath+'/data'

    return atm


def loadAtm_reinit(filename):

    atm = atmosphere(name='empty')
    atm.load(filename)
    #atm.filebase = os.path.dirname(filename) + '/' + atm.filename + '_'
    
    atm.scarletpath = os.path.dirname(pkg_resources.resource_filename('scarlet', ''))
    atm.datapath = atm.scarletpath+'/data'
    
    atm.reInitAtmosphereModel()
    
    return atm