import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')


import pickle

# Specify the path to your pickle file
pickle_file_path = 'data.pkl'

# Open the pickle file in read-binary mode
with open(pickle_file_path, 'rb') as f:
    # Load the data from the pickle file
    loaded_data = pickle.load(f)

A=loaded_data['A']
delflux=loaded_data['delflux']
levels=loaded_data['levels']

deltafluxsum=np.zeros(60)
for i in range(60):
    deltafluxsum[i] = np.trapz(x=atm.wave,y=delflux[i])
#    deltafluxsum[i]=np.sum(delflux[i])
#delta_levels=np.linalg.solve(A_regularized, delflux)

tolerance = 1e-6
A_regularized = A + tolerance * np.eye(A.shape[0])

x=np.linalg.solve(A_regularized, deltafluxsum)
pdb.set_trace()
