# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:10:54 2013

@author: bbenneke
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import math
import os
import numpy as np

import pymultinest
#import threading   #for wait commands

import matplotlib.pyplot as plt

import plotMulti

from auxbenneke.utilities import *


import scipy.stats as stats



loadvar('filename.dat', locals())


#filename='Variables.dat'
#import pickle
#f = file(filename, 'rb')
#counter = pickle.load(f)
#a = pickle.load(f)
#f.close



#---Analyze the results-------------------------------------------------------------
a = pymultinest.Analyzer(n_params = n_params)
s = a.get_stats()

import json
json.dump(s, file('{:s}.json'.format(a.outputfiles_basename), 'w'), indent=2)


# keyboard()

print('')
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t{:.7e} +- {:.7e}".format( s['global evidence'], s['global evidence error'] ))


p = plotMulti.PlotMarginalModes(a)


#--Panel----------------------------------------------------------------------------------------
plt.figure(figsize=(4*n_params, 4*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
	plt.subplot(n_params, n_params, n_params * i + i + 1)
	p.plot_conditional(i, with_ellipses = True, with_points = False, grid_points=30)#, limits=np.array([[0.0,1.0],[0.0,1.0]]))
	plt.ylabel("Probability")
	plt.xlabel(parameters[i])
	for j in range(i):
		plt.subplot(n_params, n_params, n_params * j + i + 1)
		#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
		p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)#, limits=np.array([[0.0,1.0],[0.0,1.0]]))
		plt.xlabel(parameters[i])
		plt.ylabel(parameters[j])
        
plt.savefig("marginals_panel.pdf", bbox_inches='tight')
#plt.show()



#---Individual Marginal Plots--------------------------------------------------------------------- 
#for i in range(n_params):
#	# Marginal PDFs
#	outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
#	p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
#	plt.ylabel("Probability")
#	plt.xlabel(parameters[i])
#	plt.savefig(outfile, format='pdf', bbox_inches='tight')
#	plt.close()
#	openext(outfile)
#	
#	# Marginal CDFs      
#	outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
#	p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
#	plt.ylabel("Cumulative probability")
#	plt.xlabel(parameters[i])
#	plt.savefig(outfile, format='pdf', bbox_inches='tight')
#	plt.close()
#	openext(outfile)



#openext("marginals_multinest.pdf")



print("take a look at the pdf files in chains/" )
print("Done!")