import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
#print(atm.p)
x=atm.calcNonGrayTpProf(atm.modelSetting,atm.params,False,False,False)
pdb.set_trace()
print(x)
