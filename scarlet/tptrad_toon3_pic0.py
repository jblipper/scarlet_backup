import pickle
from auxbenneke.constants import unitfac

import pdb
import scarlet
from scarlet import radutils as rad
import numpy as np
from scipy.integrate import RK45
from matplotlib import pyplot as plt
from copy import deepcopy

from numba import jit, vectorize
from toon_pic import multiScatToon,CP,CM,slice_gt

#%% STARTING POINT OF MAIN CODE
if __name__ == "__main__":
    #atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240307_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity75_CtoO0.54_pQuench1e-99_TpNonGrayTint75.0f0.25A0.1_pCloud100000.0mbar.atm')
    atm = scarlet.loadAtm('/Users/justinlipper/Research/GitHub/scarlet_results/FwdRuns20240209_0.3_100.0_64_nLay60/HD_209458_b/HD_209458_b_Metallicity1_CtoO0.54_pQuench1e-99_TpTeqTint100.0f0.25A0.1_pCloud100000.0mbar.atm')
pdb.set_trace()

T=np.ones(60)*500
#T=np.array([ 1.77705462e+05, -3.54916625e+05, -1.49773140e+06, -3.75374431e+06,-8.01585268e+06, -1.55647184e+07, -2.58519628e+07, -1.79441072e+07,-1.99180266e+07, -2.10320078e+07, -2.22210297e+07, -2.35205869e+07,-2.49470340e+07, -2.64471817e+07, -2.86939881e+07, -3.10412549e+07,-3.34704627e+07, -3.57640188e+07, -3.81382279e+07, -4.13040850e+07,-4.34702936e+07, -4.32965084e+07, -4.15101782e+07, -3.62437233e+07,-2.91650108e+07, -2.14487072e+07, -1.43691073e+07, -8.89543618e+06,-5.15418976e+06, -2.84872796e+06, -1.49378615e+06, -7.44867920e+05,-3.64711696e+05, -1.69525010e+05, -7.21813088e+04, -2.89619026e+04,-9.17826942e+03, -2.44037612e+02,  3.47733547e+03,  5.01293760e+03,5.64129045e+03,  5.89084996e+03,  5.98535471e+03,  6.02135867e+03,6.03352028e+03,  6.03535983e+03,  6.03402545e+03,  6.03304804e+03,6.03215135e+03,  6.03172047e+03,  6.03155317e+03,  6.03145118e+03,6.03136859e+03,  6.03133279e+03,  6.03134648e+03,  6.03137525e+03,6.03141629e+03,  6.03152507e+03,  6.03190857e+03,  6.03289385e+03])
#T=np.array([702.9479696168794,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
#T=np.array([560,713.2786779124422,722.7045784133379,732.696157051642,743.753880183868,755.8014535091122,767.8598673647135,779.1186692058243,789.865361820865,801.6670726257925,814.742379966605,828.6750471762499,843.2483019054786,858.3760281728315,874.1125084156081,890.4942820168549,908.7858667827371,929.5094115598118,951.8382193253866,974.9589343354603,999.1837372000501,1024.729984410233,1052.7937360483536,1084.0202653679503,1119.885983709968,1162.417493130407,1213.538497151404,1274.004805599077,1346.5224290743968,1432.988337933702,1533.9008637812565,1647.9971468587405,1767.8961021566101,1881.6243060330044,1981.4794040434522,2074.074340634299,2170.0445849194275,2265.771777135666,2338.3642622765815,2369.0464699681697,2373.9567160921856,2374.226548940419,2374.412542119859,2374.7652316013637,2375.4391468653853,2376.7372971765253,2379.258447753211,2384.1343880619065,2393.585123482547,2412.0223042325765,2444.3811724995944,2495.2790526803033,2572.7978653135356,2688.3503570766975,2855.0270002514562,3081.9134842255653,3371.8247108003216,3719.3416035449873,4107.782387253428,4517.8515421764005])
#T=T[::-1]
#T=np.array([1074.4482181048131,1121.9420765688992,1176.3529281100662,1237.1687531116022,1301.4005724960277,1362.4086273923463,1402.6223753758416,1401.7351236157772,1352.7586679852475,1271.997702589589, 1184.0483476487666, 1103.1000736493618, 1035.9449571717614,986.797500540058,957.2243804914506, 942.4161131304311, 938.6587960382013,941.5663297590517,948.1386741924126,957.3290455687113,968.8335994367401,982.129081360941,995.9607058657162,1008.6423055105771,1021.3346448503885, 1035.752560542239,1050.6941447818976,1067.5775549337513,1087.098222219365,1110.8451698424633, 1139.6561523363555,1174.9212057099726, 1218.359082917966, 1273.6212332735063, 1341.459527076653,1417.8946811784303,1499.0066667346894, 1582.9874597665373,1664.89421085604,1730.6751835590262,1779.0398081179833, 1813.2218065923082,1833.2320207862224,1840.0728756309957,1841.0714980919167, 1841.3372270032817,1841.8367993611864,1842.928365853905,1845.3828214028129, 1851.0315438246794,1863.1992209831199,1887.536925372106,1934.6726191550067, 2022.1010786574093,2172.6376471906956,2401.6915165149267,2697.4015040771096, 3012.084204402088,3307.702111716429,3577.725788640204])

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
            #ax.set_xlim([self.TList.max*1.2])
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
    extinctCoef0=atm.opac['extinctCoef'].copy()
    scatCoef0=atm.opac['scatCoef'].copy()
    extinctCoef = 0.5*(extinctCoef0[:atm.nLay-1,:]+extinctCoef0[1:atm.nLay,:])
    scatCoef = 0.5*(scatCoef0[:atm.nLay-1,:]+scatCoef0[1:atm.nLay,:])          
    
    pdb.set_trace()
    #------------------------------------------------------------------------#
    #---------------Calculate temperature perturbations----------------------#
    #------------------------------------------------------------------------#
     
    deltas=np.ones(N)*11
    count=0
    doplot=True
    firstplot=True
    #while np.max(np.abs(deltas))>0.1:
    while True:    
        print(f'deltas: {deltas}, levels: {levels}')
        deltas,levels=take_step(TList,levels,N,extinctCoef0,scatCoef0,doplot,firstplot)
        TList = np.c_[TList,levels]
        firstplot=False
        #doplot=False
        count+=1
        print(count)
        if count>20*10**20:
            pdb.set_trace()
            count=0
            #doplot=True
            
    
def take_step(TList,in_levels,N,extinctCoef0,scatCoef0,doplot=False,firstplot=False):
    plotTp(forceLabelAx=True)
    plotTpChanges(TList)
    atm.T=in_levels
    #toon= atm.multiScatToon(atm.IrradStar,extinctCoef,scatCoef,refl = True,thermal = True,w=0)
    toon= multiScatToon(atm,atm.IrradStar,extinctCoef0,scatCoef0,asym = 0,u0 = 0.5773502691896258,refl = False,thermal = True,w=-1)
    
    for qw in range(59):
        toon[0][qw][:7900]=0
        toon[1][qw][:7900]=0
    
    if doplot==True:
        plot_fluxes(toon,firstplot)
    flux0_up=toon[0]
    flux0_down=toon[1]
    flux0=toon[0]-toon[1] 
    #flux0=toon[2]
    #flux0=np.zeros((len(in_levels)-1,len(atm.wave)))
    #for lev in range(1,len(flux0)):
    #    flux_in=flux0_up[lev-1]+flux0_down[lev]
    #    flux_out=flux0_down[lev-1]+flux0_up[lev]
    #    flux0[lev]=flux_in-flux_out
        
        ##flux0[lev]=toon[2][lev]-toon[2][lev-1]
        
        
        
    
    
    delflux=-1*flux0
    A=np.zeros((N-1,N),dtype=atm.numerical_precision)
    
    for level_i in range(N):
        print('*****')
        print(level_i)
        print('*****')
        deltaT=1
        levels_ptb=in_levels.copy()
        levels_ptb[level_i]=in_levels[level_i]+deltaT
        atm.T=levels_ptb
        flux_ptb_all= multiScatToon(atm,atm.IrradStar,extinctCoef0,scatCoef0,asym = 0,u0 = 0.5773502691896258,refl = False,thermal = True,w=-1)
        
        for qw in range(59):
            flux_ptb_all[0][qw][:7900]=0
            flux_ptb_all[1][qw][:7900]=0
        
        flux_ptb=flux_ptb_all[0]-flux_ptb_all[1]
        #flux_ptb=flux_ptb_all[2]
        #flux_ptb_up=flux_ptb_all[0]
        #flux_ptb_down=flux_ptb_all[1]
        #flux_ptb=np.zeros((len(in_levels)-1,len(atm.wave)))
        #for lev in range(1,len(flux_ptb)):
        #    flux_ptb_in=flux_ptb_up[lev-1]+flux_ptb_down[lev]
        #    flux_ptb_out=flux_ptb_down[lev-1]+flux_ptb_up[lev]
        #    flux_ptb[lev]=flux_ptb_in-flux_ptb_out
            
            ##flux_ptb[lev]=flux_ptb_all[2][lev]-flux_ptb_all[2][lev-1]
        A_level_i=(flux_ptb-flux0)/deltaT
        for layer_i in range(N-1):
            A[layer_i][level_i]=np.trapz(x=atm.wave,y=A_level_i[layer_i])
        atm.T=in_levels    
    pdb.set_trace()     
    
    deltafluxsum=np.zeros(N-1)
    for i in range(N-1):
        deltafluxsum[i] = np.trapz(x=atm.wave,y=delflux[i])
    delta_levels_lin = np.linalg.lstsq(A, deltafluxsum, rcond=None)[0]
    delta_levels_reduced=delta_levels_lin*0.10
    maxabs=np.max(np.abs(delta_levels_reduced))
    if maxabs>100:
        delta_levels_reduced=delta_levels_reduced*100/maxabs
    new_levels=in_levels+delta_levels_reduced
    
   ## for k in range(len(new_levels)-3):
   ##         if (new_levels[k]>new_levels[k+1] and new_levels[k+1]<new_levels[k+2] and new_levels[k+2]>new_levels[k+3]) or (new_levels[k]<new_levels[k+1] and new_levels[k+1]>new_levels[k+2] and new_levels[k+2]<new_levels[k+3]):
   ##             smooth=0.25*(new_levels[k]+new_levels[k+1]+new_levels[k+2]+new_levels[k+3])
   ##             #interval=levels[k+3]-levels[k]
   ##             new_levels[k]=smooth
   ##             new_levels[k+1]=smooth
   ##             new_levels[k+2]=smooth
   ##             new_levels[k+3]=smooth
    
    atm.T=new_levels    
    
    return delta_levels_reduced,new_levels
   
    
    
     
   
    data = {'A': A, 'delflux': delflux, 'levels':levels, 'delfluxsum': deltafluxsum, 'delta_levels': delta_levels}
    with open('data.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    #delta_levels=np.linalg.solve(A,delflux)
    
    
    atm.T=levels
    pdb.set_trace() 
    #tp_trad(levels,do_scat=True,negli_flux=0)
    return levels               

x=tp_trad(T)
pdb.set_trace() 
