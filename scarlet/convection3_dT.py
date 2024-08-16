import pickle
from auxbenneke.constants import unitfac

import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
from scipy.integrate import RK45
from matplotlib import pyplot as plt
from toonew_funcs import get_fluxes_toon
from copy import deepcopy






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
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpNonGrayTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
pdb.set_trace()

#T=np.ones(60)*2000
T=np.array([ 801.04726038,  802.06800173,  805.98237113,  810.46787058,  818.47038155,  827.43164902,  839.73617398,  852.45641686,  867.41307381,  880.75136987,  893.33864209,  902.22252756,  910.80014821,  918.48811073,  928.69236472,  938.88333716,  951.69035626,  964.57268834,  979.87861396,  995.35824736, 1013.55341028, 1031.98709707, 1052.87973707, 1071.8743683,  1091.3655835, 1112.35532581, 1135.21706267, 1158.02967468, 1185.67665515, 1217.44437484, 1256.03165636, 1301.02118829, 1354.51825581, 1416.32259537, 1485.65744154, 1555.01349023, 1624.90291458, 1692.75385532, 1756.65506873, 1800.78171282, 1834.85726405, 1859.19141734, 1873.35973387, 1878.08347106, 1878.46252075, 1878.69783238, 1879.21135397, 1880.30695016, 1882.63383997, 1887.62244917, 1897.78171099, 1917.68511478, 1955.63669003, 2024.40178139, 2138.58925498, 2308.09571828, 2530.19339384, 2786.7255833,  3052.65723082, 3313.30311644])
T=np.array([ 795.00809122,  795.18674694,  796.99803796,  799.41937976,  804.69767702,  811.04595897,  820.58499912,  831.72011334,  845.39835334,  859.89207444,  874.99914839,  887.80780849,  898.43134577,  906.60032405,  915.41446608,  924.86730906,  936.41405059,  948.9855567,   963.46234245,  978.8925438,  996.48925532, 1015.22781704, 1036.03362651, 1056.27184005, 1075.36837544, 1096.04329258, 1119.10263593, 1141.87937785, 1168.54313598, 1200.13986227, 1238.07720349, 1283.5781127,  1337.74970719, 1402.33615525, 1475.36524597, 1550.27037012, 1624.79066,    1698.19054536, 1763.81920745, 1807.30132399, 1840.23716019, 1861.89899733, 1872.56504985, 1874.99022863, 1875.17961555, 1875.48484026, 1876.15068881, 1877.56803222, 1880.57386749, 1886.99875222, 1900.04623459, 1925.47027554, 1973.6203475,  2060.02279281, 2202.44481433, 2413.05016914, 2687.96443449, 3001.2260175,  3323.05366256, 3642.0439266 ])
#T=np.array([800.5913503, 801.61156179,  805.52803789,  810.0151056,   818.02208754, 826.98755989,  839.29916321,  852.02613816,  866.99188754,  880.33803998, 892.93380172,  901.82336893,  910.40684864,  918.09919834,  928.3099942, 938.50714901,  951.32280372,  964.21354661,  979.5301533,   995.02035119, 1013.22831453, 1031.67445686, 1052.5811021,  1071.58764479, 1091.0908855, 1112.09354873, 1134.96915658, 1157.79522741, 1185.45787976, 1217.2421723, 1255.8476895,  1300.85623921, 1354.37365776, 1416.19902431, 1485.55514694, 1554.93044716, 1624.83746748, 1692.70376727, 1756.6183794,  1800.75374456, 1834.83553592, 1859.17384072, 1873.34445768, 1878.06895131, 1878.44807211, 1878.68339998, 1879.19704599, 1880.29273299, 1882.62017482, 1887.60918166, 1897.77079955, 1917.67561862, 1955.63531759, 2024.40338315, 2138.61030184, 2308.11566269, 2530.23959546, 2786.75776089, 3052.71224738, 3313.3558681 ])
#T=np.zeros(60)
#T=np.array([ 45.04726038,  55.06800173,  60.98237113,  75.46787058,  80.47038155,  90.43164902,  100.73617398,  352.45641686,  607.41307381,  880.75136987,  893.33864209,  902.22252756,  910.80014821,  918.48811073,  928.69236472,  938.88333716,  951.69035626,  964.57268834,  979.87861396,  995.35824736, 1013.55341028, 1031.98709707, 1052.87973707, 1071.8743683,  1091.3655835, 1112.35532581, 1135.21706267, 1158.02967468, 1185.67665515, 1217.44437484, 1256.03165636, 1301.02118829, 1354.51825581, 1416.32259537, 1485.65744154, 1555.01349023, 1624.90291458, 1692.75385532, 1756.65506873, 1800.78171282, 1834.85726405, 1859.19141734, 1873.35973387, 1878.08347106, 1878.46252075, 1878.69783238, 1879.21135397, 1880.30695016, 1882.63383997, 1887.62244917, 1897.78171099, 1917.68511478, 1955.63669003, 2024.40178139, 2138.58925498, 2308.09571828, 2530.19339384, 2786.7255833,  3052.65723082, 3313.30311644])
#T=np.array([ 1.77705462e+05, -3.54916625e+05, -1.49773140e+06, -3.75374431e+06,-8.01585268e+06, -1.55647184e+07, -2.58519628e+07, -1.79441072e+07,-1.99180266e+07, -2.10320078e+07, -2.22210297e+07, -2.35205869e+07,-2.49470340e+07, -2.64471817e+07, -2.86939881e+07, -3.10412549e+07,-3.34704627e+07, -3.57640188e+07, -3.81382279e+07, -4.13040850e+07,-4.34702936e+07, -4.32965084e+07, -4.15101782e+07, -3.62437233e+07,-2.91650108e+07, -2.14487072e+07, -1.43691073e+07, -8.89543618e+06,-5.15418976e+06, -2.84872796e+06, -1.49378615e+06, -7.44867920e+05,-3.64711696e+05, -1.69525010e+05, -7.21813088e+04, -2.89619026e+04,-9.17826942e+03, -2.44037612e+02,  3.47733547e+03,  5.01293760e+03,5.64129045e+03,  5.89084996e+03,  5.98535471e+03,  6.02135867e+03,6.03352028e+03,  6.03535983e+03,  6.03402545e+03,  6.03304804e+03,6.03215135e+03,  6.03172047e+03,  6.03155317e+03,  6.03145118e+03,6.03136859e+03,  6.03133279e+03,  6.03134648e+03,  6.03137525e+03,6.03141629e+03,  6.03152507e+03,  6.03190857e+03,  6.03289385e+03])
#T=np.array([702.9479696168794,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
#T=np.array([560,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
#T=T[::-1]
#T=np.array([1074.4482181048131,1121.9420765688992,1176.3529281100662,1237.1687531116022,1301.4005724960277,1362.4086273923463,1402.6223753758416,1401.7351236157772,1352.7586679852475,1271.997702589589, 1184.0483476487666, 1103.1000736493618, 1035.9449571717614,986.797500540058,957.2243804914506, 942.4161131304311, 938.6587960382013,941.5663297590517,948.1386741924126,957.3290455687113,968.8335994367401,982.129081360941,995.9607058657162,1008.6423055105771,1021.3346448503885, 1035.752560542239,1050.6941447818976,1067.5775549337513,1087.098222219365,1110.8451698424633, 1139.6561523363555,1174.9212057099726, 1218.359082917966, 1273.6212332735063, 1341.459527076653,1417.8946811784303,1499.0066667346894, 1582.9874597665373,1664.89421085604,1730.6751835590262,1779.0398081179833, 1813.2218065923082,1833.2320207862224,1840.0728756309957,1841.0714980919167, 1841.3372270032817,1841.8367993611864,1842.928365853905,1845.3828214028129, 1851.0315438246794,1863.1992209831199,1887.536925372106,1934.6726191550067, 2022.1010786574093,2172.6376471906956,2401.6915165149267,2697.4015040771096, 3012.084204402088,3307.702111716429,3577.725788640204])

#Tint 500
#T=np.array([797.03515266,   798.05086823,   801.98753673,   806.48863085, 814.5364388,    823.53922801,   835.914488,     848.70106525, 863.74818354,   877.16414528,   889.83696187,   898.77954733, 907.42075723,   915.15935352,   925.43838296,   935.70169153,  948.60792001,   961.58879513,   977.01859178,   992.62275656, 1010.96783103,  1029.55144103,  1050.61477696,  1069.76743409, 1089.42737464,  1110.59808665,  1133.65917769,  1156.67591331,  1184.55713081,  1216.57726385,  1255.45005819,  1300.75282899,  1354.60754351,  1416.81304476,  1486.61237002,  1556.50490357,  1627.05967059,  1695.78916082,  1760.95928692,  1807.08848681,  1844.51506503,  1874.98792449,  1901.3945783,   1932.12689339,  1986.46448861,  2087.82896173,  2257.29911354,  2510.3590495,  2865.82100245,  3346.35005683,  3908.89755141,  4484.65794955, 5063.44568174,  5661.07198671,  6302.7142164,   7014.56694351, 7828.93909526,  8784.1954358,   9930.77025663, 11337.81165218])

Tsolu=np.array([1074.4482181048131,1121.9420765688992,1176.3529281100662,1237.1687531116022,1301.4005724960277,1362.4086273923463,1402.6223753758416,1401.7351236157772,1352.7586679852475,1271.997702589589, 1184.0483476487666, 1103.1000736493618, 1035.9449571717614,986.797500540058,957.2243804914506, 942.4161131304311, 938.6587960382013,941.5663297590517,948.1386741924126,957.3290455687113,968.8335994367401,982.129081360941,995.9607058657162,1008.6423055105771,1021.3346448503885, 1035.752560542239,1050.6941447818976,1067.5775549337513,1087.098222219365,1110.8451698424633, 1139.6561523363555,1174.9212057099726, 1218.359082917966, 1273.6212332735063, 1341.459527076653,1417.8946811784303,1499.0066667346894, 1582.9874597665373,1664.89421085604,1730.6751835590262,1779.0398081179833, 1813.2218065923082,1833.2320207862224,1840.0728756309957,1841.0714980919167, 1841.3372270032817,1841.8367993611864,1842.928365853905,1845.3828214028129, 1851.0315438246794,1863.1992209831199,1887.536925372106,1934.6726191550067, 2022.1010786574093,2172.6376471906956,2401.6915165149267,2697.4015040771096, 3012.084204402088,3307.702111716429,3577.725788640204])
Tsolu=np.array([ 801.04726038,  802.06800173,  805.98237113,  810.46787058,  818.47038155,  827.43164902,  839.73617398,  852.45641686,  867.41307381,  880.75136987,  893.33864209,  902.22252756,  910.80014821,  918.48811073,  928.69236472,  938.88333716,  951.69035626,  964.57268834,  979.87861396,  995.35824736, 1013.55341028, 1031.98709707, 1052.87973707, 1071.8743683,  1091.3655835, 1112.35532581, 1135.21706267, 1158.02967468, 1185.67665515, 1217.44437484, 1256.03165636, 1301.02118829, 1354.51825581, 1416.32259537, 1485.65744154, 1555.01349023, 1624.90291458, 1692.75385532, 1756.65506873, 1800.78171282, 1834.85726405, 1859.19141734, 1873.35973387, 1878.08347106, 1878.46252075, 1878.69783238, 1879.21135397, 1880.30695016, 1882.63383997, 1887.62244917, 1897.78171099, 1917.68511478, 1955.63669003, 2024.40178139, 2138.58925498, 2308.09571828, 2530.19339384, 2786.7255833,  3052.65723082, 3313.30311644])
Tsolu=np.array([ 795.00809122,  795.18674694,  796.99803796,  799.41937976,  804.69767702,  811.04595897,  820.58499912,  831.72011334,  845.39835334,  859.89207444,  874.99914839,  887.80780849,  898.43134577,  906.60032405,  915.41446608,  924.86730906,  936.41405059,  948.9855567,   963.46234245,  978.8925438,  996.48925532, 1015.22781704, 1036.03362651, 1056.27184005, 1075.36837544, 1096.04329258, 1119.10263593, 1141.87937785, 1168.54313598, 1200.13986227, 1238.07720349, 1283.5781127,  1337.74970719, 1402.33615525, 1475.36524597, 1550.27037012, 1624.79066,    1698.19054536, 1763.81920745, 1807.30132399, 1840.23716019, 1861.89899733, 1872.56504985, 1874.99022863, 1875.17961555, 1875.48484026, 1876.15068881, 1877.56803222, 1880.57386749, 1886.99875222, 1900.04623459, 1925.47027554, 1973.6203475,  2060.02279281, 2202.44481433, 2413.05016914, 2687.96443449, 3001.2260175,  3323.05366256, 3642.0439266 ])

def plot_fluxes(in_toon,firstplot=False):
    flux_up=np.trapz(x=atm.wave,y=in_toon[0])
    flux_down=np.trapz(x=atm.wave,y=in_toon[1])
    #flux=np.trapz(x=atm.wave,y=in_toon[2])
    print(flux_up)
    print(flux_down)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    y=atm.p#[:59]
    
    x1_fluxup=flux_up
    x1_fluxdown=flux_down
    ax1.plot(x1_fluxup,y,label='Up')
    ax1.plot(x1_fluxdown,y,label='Down')
    ax1.set_xlabel(r'$Flux (\frac{W}{m^{2}})$')
    ax1.set_yscale('log')
    ax1.invert_yaxis() 
    ax1.set_ylabel(r'Pressure (bar)')
    plt.ylabel('Pressure (bar)') 
    ax1.legend()
    
    
    x2=(flux_up-flux_down)/flux_up
    ax2.plot(x2,y)
    ax2.set_xlabel(r'$\frac{{Flux_{up} - Flux_{down}}}{{Flux_{up}}}$')
    ax2.set_yscale('log')
    ax2.invert_yaxis() 
    ax2.set_ylabel(r'Pressure (bar)')
    plt.ylabel('Pressure (bar)')
    
    plt.tight_layout()
    if firstplot==True:
        plt.savefig('plot0.png')
    else:
        plt.savefig('plot.png')
    
   

def plotTp(ax=None,axisOnly=False,figsize=[8,10],save=True,showTeq=True,marker='x',punit='bar',partialPresMol=None,forceLabelAx=False,**kwargs):
    ## show convection limit
    if ax is None:
        fig, ax = plt.subplots(1,figsize=figsize)
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure ['+punit+']')
        if axisOnly is False:
            if showTeq:
                plt.axvline(x=atm.Teq,color='r',linestyle=':',zorder=-10)
            if partialPresMol is None:
                ax.semilogy(atm.T,atm.p*unitfac(punit),marker=marker,**kwargs)  
                ax.set_ylabel('Pressure ['+punit+']')
            else:
                ax.semilogy(atm.T,atm.p*unitfac(punit)*atm.getMixRatio(partialPresMol),marker=marker,**kwargs)  
                ax.set_ylabel('Partial pressure of {} [{}]'.format(partialPresMol,punit))
        
        ax.semilogy(Tsolu,atm.p*unitfac(punit),marker=marker,**kwargs) 
        
        ax.set_ylim([1e-12,1e4])
        ax.invert_yaxis()
        ax.minorticks_on()
        if save:
            fig.savefig('Tp.pdf')            
        return fig,ax
    else:
        if partialPresMol is None:
            ax.semilogy(atm.T,atm.p*unitfac(punit),marker=marker,**kwargs)  
            ax.set_ylabel('Pressure ['+punit+']')
        else:
            ax.semilogy(atm.T,atm.p*unitfac(punit)*atm.qmol_lay[:,atm.getMolIndex(partialPresMol)],marker=marker,**kwargs)  
            ax.set_ylabel('Partial pressure of {} [{}]'.format(partialPresMol,punit))
                
    if forceLabelAx:
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure ['+punit+']')
        ax.set_ylim([1e-12,1e4])
        ax.invert_yaxis()
        ax.minorticks_on()



def plotTpChanges(TList,ax=None,save=True,close=True,loop=None,**kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1,sharex=True,sharey=True,figsize=[8,10])
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            plt.axvline(x=atm.Teq,color='black',linestyle='--')
            for temp in range(len(TList[0])):
                ax.semilogy( TList[:,temp],atm.p/1e5,**kwargs)  
            ax.semilogy( TList[:,temp],atm.p/1e5,marker='x',**kwargs)
            ax.set_ylim([1e-10,1e4])
            #ax.set_xlim([atm.TList.max*1.2])
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                if loop is None:
                    fig.savefig('_TpChange.pdf')
                else:
                    fig.savefig('_TpChange'+str(loop)+'.pdf')
            if close:
                plt.close(fig)
            return fig,ax
        else:
            for temp in range(len(TList[0])):
                ax.semilogy( TList[:,temp],atm.p/1e5,marker='x',label=temp,**kwargs)



def tp_trad(T,plot=True):
    atm.T=T    
    N=len(T)
    levels=T.copy()    
    TList =np.array([np.array([level]) for level in deepcopy(levels)])
    extinctCoef=atm.opac['extinctCoef'].copy()
    scatCoef=atm.opac['scatCoef'].copy()
    #extinctCoef = 0.5*(extinctCoef[:atm.nLay-1,:]+extinctCoef[1:atm.nLay,:])
    #scatCoef = 0.5*(scatCoef[:atm.nLay-1,:]+scatCoef[1:atm.nLay,:])          
    
    pdb.set_trace()
    #------------------------------------------------------------------------#
    #---------------Calculate temperature perturbations----------------------#
    #------------------------------------------------------------------------#
     
    deltas=np.ones(N)*11#/100
    count=0
    doplot=True
    firstplot=True
    conv_zones=np.zeros(N)
    while np.max(np.abs(deltas))>0.5:
        deltas,levels=take_step_tp_trad(TList,levels,N,extinctCoef,scatCoef,conv_zones,doplot,firstplot)
        TList = np.c_[TList,levels]
        firstplot=False
        #doplot=False
        count+=1
        print(count)  
        print(levels)
    #Convection
    convection=False
    if convection==True:
        levels=conv_corr(atm,levels,conv_zones,TList,extinctCoef,scatCoef)    
        atm.T=levels
        plotTp(forceLabelAx=True)
        TList = np.c_[TList,levels]
        plotTpChanges(TList)      
    
def take_step_tp_trad(TList,in_levels,N,extinctCoef,scatCoef,conv_zones,doplot=False,firstplot=False):
    rad_zones=1-conv_zones
    N_rad=np.sum(rad_zones==1)
    print(f'N_rad:{N_rad}')
    #N_rad=60
    #pdb.set_trace()
    plotTp(forceLabelAx=True)
    plotTpChanges(TList)
    atm.T=in_levels
    atm.z,atm.dz,atm.grav,atm.ntot,atm.nmol,atm.MuAve,atm.scaleHeight,atm.RpBase,atm.r=calcHydroEqui(in_levels)
    toon= get_fluxes_toon(atm,atm.IrradStar,extinctCoef,scatCoef)
    
    if doplot==True:
        plot_fluxes(toon,firstplot)
    #flux0_up=toon[0]
    #flux0_down=toon[1]
    flux0=toon[0]-toon[1] 
    check=np.trapz(x=atm.wave,y=flux0[0])-sigmaSB*100**4
    print(f'Net Flux at Top - σT_int^4: {check}')
    print(f'Net Flux at Top - σT_int^4 normalized: {check/(0.5*(np.trapz(x=atm.wave,y=toon[0][0])+np.trapz(x=atm.wave,y=toon[1][0])))}')    
        
        
    
    
    delflux=-1*flux0
    A=np.zeros((N_rad,N_rad),dtype=atm.numerical_precision)
    
    for level_i in range(N_rad):
        print('*****')
        print(level_i)
        print('*****')
        deltaT=0.001
        levels_ptb=in_levels.copy()
        levels_ptb[level_i]=in_levels[level_i]+deltaT
        if level_i==N_rad-1:
            print('zzz')
            for j in range(N_rad,N):
                levels_ptb[j]=in_levels[j]+deltaT
         
            print(levels_ptb-in_levels)
            
        atm.T=levels_ptb
        atm.z,atm.dz,atm.grav,atm.ntot,atm.nmol,atm.MuAve,atm.scaleHeight,atm.RpBase,atm.r=calcHydroEqui(levels_ptb)
        flux_ptb_all=get_fluxes_toon(atm,atm.IrradStar,extinctCoef,scatCoef)
        
        flux_ptb=flux_ptb_all[0]-flux_ptb_all[1]        
        #flux_ptb_up=flux_ptb_all[0]
        #flux_ptb_down=flux_ptb_all[1]
        A_level_i=(flux_ptb-flux0)/deltaT
        for layer_i in range(N_rad):
            A[layer_i][level_i]=np.trapz(x=atm.wave,y=A_level_i[layer_i])
        atm.T=in_levels         
    
    deltafluxsum=np.zeros(N_rad)
    for i in range(N_rad):
        deltafluxsum[i] = np.trapz(x=atm.wave,y=delflux[i])
    delta_levels_lin_rad = np.linalg.lstsq(A, (sigmaSB*100**4+deltafluxsum), rcond=None)[0]
    delta_levels_lin = np.concatenate((delta_levels_lin_rad, np.ones(atm.nLay-N_rad)*delta_levels_lin_rad[-1]))
    #pdb.set_trace()
    delta_levels_reduced=delta_levels_lin*0.10
    maxabs=np.max(np.abs(delta_levels_reduced))
    if maxabs>250:
        delta_levels_reduced=delta_levels_reduced*250/maxabs
    new_levels=in_levels+delta_levels_reduced#*rad_zones
    
    #for zone in np.where(conv_zones == 1)[0]:
    #    #pdb.set_trace()
    #    new_levels[zone]=new_levels[zone-1]+100
    
    atm.T=new_levels    
    
    return delta_levels_reduced,new_levels
   
    
    
     
   
    data = {'A': A, 'delflux': delflux, 'levels':levels, 'delfluxsum': deltafluxsum, 'delta_levels': delta_levels}
    with open('data.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    #delta_levels=np.linalg.solve(A,delflux)
    
    
    atm.T=levels
    #pdb.set_trace() 
    #tp_trad(levels,do_scat=True,negli_flux=0)
    return levels    
    
def conv_corr(atm,T,conv_zones,TList,extinctCoef,scatCoef):    
    z,dz,grav,ntot,nmol,MuAve,scaleHeight,RpBase,r=calcHydroEqui(T)
    plotTp(forceLabelAx=True)
    plotTpChanges(TList)
    print('convection')
    dT=np.diff(T)
    dTdz = dT / dz
    MuAveLay = 0.5*(MuAve[:-1]+MuAve[1:])
    Rconst = 8.3144598 /(MuAveLay/uAtom * 0.001)#J/mol*K
    heatCap = 7./2. * Rconst *2.7
    grav=0.5*(grav[:-1]+grav[1:])
    dTdzAdiabat = grav/heatCap *1.2 #1.2 ADDED
    slopeTooSteep = (np.where(dTdz > dTdzAdiabat+1e-10)[0])[::-1] 
    print(slopeTooSteep)
    print(dTdz-dTdzAdiabat)
    plotTp(forceLabelAx=True)
    plotTpChanges(TList)
    if len(slopeTooSteep>0):
        conv_lay = slopeTooSteep[0]
        print(f'conv_lay:{conv_lay}')
        T[conv_lay+1]=T[conv_lay]+dTdzAdiabat[conv_lay]*dz[conv_lay]
        #T[conv_lay]=5000
        #T[conv_lay+1]=3700
        #T[conv_lay+1]=T[conv_lay]+100
        conv_zones[conv_lay+1:conv_lay+2]=1
        deltas=np.ones(atm.nLay)*11
        print(f'conv:{conv_zones}')
        #while np.max(np.abs(deltas*(1-conv_zones)))>0.5:
        for ube in range(1):
            deltas,T=take_step_tp_trad(TList,T,atm.nLay,extinctCoef,scatCoef,conv_zones,doplot=False,firstplot=False)
            for zone in np.where(conv_zones == 1)[0]:
                #pdb.set_trace()
                T[zone]=T[zone-1]+dTdzAdiabat[zone-1]*dz[zone-1]
            TList = np.c_[TList,T]
            print(deltas)
        T=conv_corr(atm,T,conv_zones,TList,extinctCoef,scatCoef)       
    
    return T  

def calcHydroEqui(T):

        #Note: All values without "grid" in variable name are at cell centers
        
        #array of molar masses (atm.MolarMass) from readMoleculesProperties() above
        MuAve=uAtom*np.sum(atm.MolarMass[np.newaxis,:]*atm.qmol_lay,axis=1)  #[kg]
        
        #don't need to define p since already have atm.p
        ntot=atm.p/(kBoltz*T) # [m^-3]
        
        nmol=((ntot[:,np.newaxis]).dot(np.ones([1,atm.nMol], dtype=atm.numerical_precision)))*atm.qmol_lay #(iLev) in [m^-3]
        
        r               =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        grav            =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        scaleHeight     =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        dz              =np.zeros(atm.nLay, dtype=atm.numerical_precision)   # [m]
        
        r[atm.iLevRpRef]=deepcopy(atm.Rp)
        
        #Atmosphere below r[atm.iLevRpRef]=atm.Rp to higher pressures:
        for iLay in range(atm.iLevRpRef,atm.nLay-1):
                for repeat in range(1,5):
                    grav[iLay]        = G*atm.Mp / ( r[iLay] - 0.5*dz[iLay] )**2                                     # [m/s^2]
                    scaleHeight[iLay] = (kBoltz*T[iLay])/(MuAve[iLay]*grav[iLay])                 # [m]
                    dz[iLay]          = -scaleHeight[iLay]*np.log(atm.p[iLay]/atm.p[iLay+1]) # [m]
                r[iLay+1]     = r[iLay] - dz[iLay]
        #Atmosphere above r[atm.iLevRpRef]=atm.Rp to lower pressures:
        for iLay in range(atm.iLevRpRef,0,-1):
                for repeat in range(1,5):
                    grav[iLay-1]        = G*atm.Mp / ( r[iLay] + 0.5*dz[iLay-1] )**2                                     # [m/s^2]
                    scaleHeight[iLay-1] = (kBoltz*T[iLay-1])/(MuAve[iLay-1]*grav[iLay-1])         # [m]
                    dz[iLay-1]          = -scaleHeight[iLay-1]*np.log(atm.p[iLay-1]/atm.p[iLay]) # [m]
                r[iLay-1]     = r[iLay] + dz[iLay-1]
                
                   
  

        ## --> at this point it has determined r completely (everything else can now be derived from that)
        
        RpBase=r[-1]
        
        z=r-r[-1]         # [m]
        dz=np.zeros(atm.nLay-1, dtype=atm.numerical_precision)   # [m]
        dz=-np.diff(z)            # [m]   
        
        grav=G*atm.Mp /  r**2        # [m/s^2]
        scaleHeight=(kBoltz*T)/(MuAve*grav) # [m]
        
        #consistency check
#        scaleHeightLay = 0.5*(scaleHeight[:atm.nLay-1]+scaleHeight[1:atm.nLay])
#        dz2=-scaleHeightLay*np.log(atm.p[:-1]/atm.p[1:]) # [m]
#        print (dz2-dz)/dz*100

        return z,dz,grav,ntot,nmol,MuAve,scaleHeight,RpBase,r

x=tp_trad(T)
pdb.set_trace() 