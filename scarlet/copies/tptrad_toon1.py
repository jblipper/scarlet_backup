import pickle

import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
from scipy.integrate import RK45
#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
pdb.set_trace()

#T=np.ones(60)*560
#T=np.array([ 1.77705462e+05, -3.54916625e+05, -1.49773140e+06, -3.75374431e+06,-8.01585268e+06, -1.55647184e+07, -2.58519628e+07, -1.79441072e+07,-1.99180266e+07, -2.10320078e+07, -2.22210297e+07, -2.35205869e+07,-2.49470340e+07, -2.64471817e+07, -2.86939881e+07, -3.10412549e+07,-3.34704627e+07, -3.57640188e+07, -3.81382279e+07, -4.13040850e+07,-4.34702936e+07, -4.32965084e+07, -4.15101782e+07, -3.62437233e+07,-2.91650108e+07, -2.14487072e+07, -1.43691073e+07, -8.89543618e+06,-5.15418976e+06, -2.84872796e+06, -1.49378615e+06, -7.44867920e+05,-3.64711696e+05, -1.69525010e+05, -7.21813088e+04, -2.89619026e+04,-9.17826942e+03, -2.44037612e+02,  3.47733547e+03,  5.01293760e+03,5.64129045e+03,  5.89084996e+03,  5.98535471e+03,  6.02135867e+03,6.03352028e+03,  6.03535983e+03,  6.03402545e+03,  6.03304804e+03,6.03215135e+03,  6.03172047e+03,  6.03155317e+03,  6.03145118e+03,6.03136859e+03,  6.03133279e+03,  6.03134648e+03,  6.03137525e+03,6.03141629e+03,  6.03152507e+03,  6.03190857e+03,  6.03289385e+03])
#T=np.array([702.9479696168794,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
#T=np.array([560,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
#T=T[::-1]
T=np.array([1074.4482181048131,1121.9420765688992,1176.3529281100662,1237.1687531116022,1301.4005724960277,1362.4086273923463,1402.6223753758416,1401.7351236157772,1352.7586679852475,1271.997702589589, 1184.0483476487666, 1103.1000736493618, 1035.9449571717614,986.797500540058,957.2243804914506, 942.4161131304311, 938.6587960382013,941.5663297590517,948.1386741924126,957.3290455687113,968.8335994367401,982.129081360941,995.9607058657162,1008.6423055105771,1021.3346448503885, 1035.752560542239,1050.6941447818976,1067.5775549337513,1087.098222219365,1110.8451698424633, 1139.6561523363555,1174.9212057099726, 1218.359082917966, 1273.6212332735063, 1341.459527076653,1417.8946811784303,1499.0066667346894, 1582.9874597665373,1664.89421085604,1730.6751835590262,1779.0398081179833, 1813.2218065923082,1833.2320207862224,1840.0728756309957,1841.0714980919167, 1841.3372270032817,1841.8367993611864,1842.928365853905,1845.3828214028129, 1851.0315438246794,1863.1992209831199,1887.536925372106,1934.6726191550067, 2022.1010786574093,2172.6376471906956,2401.6915165149267,2697.4015040771096, 3012.084204402088,3307.702111716429,3577.725788640204])
def tp_trad(T):
    atm.T=T    
    N=len(T)
    levels=T.copy()    
    extinctCoef=atm.opac['extinctCoef'].copy()
    scatCoef=atm.opac['scatCoef'].copy()
    extinctCoef = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:])
    scatCoef = 0.5*(scatCoef[:atm.nLay-1,:]+scatCoef[1:atm.nLay,:])
    toon= atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = True,thermal = True)
    flux0=toon[2]             
    
    pdb.set_trace()
    #------------------------------------------------------------------------#
    #---------------Calculate temperature perturbations----------------------#
    #------------------------------------------------------------------------#
     
    delflux=-1*flux0
    
    def fun(t, temps):
        atm.T=temps
        flux0_full=toon[2] 
        flux0=np.trapz(x=atm.wave,y=flux0_full)
        A=np.zeros((N-1,N),dtype=atm.numerical_precision)
        for level_i in range(N):
            print('*****')
            print(level_i)
            print('*****')
            deltaT=1
            levels_ptb=levels.copy()
            levels_ptb[level_i]=levels[level_i]+deltaT
            atm.T=levels_ptb
            flux_ptb0=atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = True,thermal = True)[2]
            flux_ptb=np.trapz(x=atm.wave,y=flux_ptb0)
            A_level_i=(flux_ptb-flux0)/deltaT
            for layer_i in range(N-1):
                A[layer_i][level_i]=A_level_i[layer_i]
        atm.T=levels
        xx=np.dot(A,temps)  
        pdb.set_trace()
        return xx  
        
    t_0=0
    t_b=10
    flux0_full=toon[2] 
    y0=np.trapz(x=atm.wave,y=flux0_full)
    pdb.set_trace() 
    solution=RK45(fun, t_0, y0, t_b)
    pdb.set_trace() 

    return [row[-1] for row in solution.y]               

x=tp_trad(T)
pdb.set_trace() 
