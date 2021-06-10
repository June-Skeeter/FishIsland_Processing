import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta
import datetime
import pytz
from sklearn.utils import resample
from sklearn import metrics
from scipy.optimize import curve_fit
from scipy import stats



class Compile:
    def __init__(self,Flux_Paths,Met,Soil,Daytime,Taglu=None,NARR=None,Drop_Variables=None):
        self.Taglu = Taglu
        self.NARR = NARR
        self.Fluxes = ['H','LE','co2_flux','ch4_flux']
        Flux_10 = self.Format(pd.read_csv(Flux_Paths[0],delimiter = ',',skiprows = 0,parse_dates={'datetime':[1,2]},header = 1,na_values = -9999),v=1,drop = [0,1])
        Flux_10 = Flux_10.loc[Flux_10['file_records']==18000]
        Flux_1 = self.Format(pd.read_csv(Flux_Paths[1],delimiter = ',',skiprows = 0,parse_dates={'datetime':[1,2]},header = 1,na_values = -9999),v=1,drop = [0,1])
        Flux_1 = Flux_1.loc[Flux_1['file_records']==1800]
        Daytime = pd.read_csv(Daytime)
        Daytime=Daytime.set_index(pd.DatetimeIndex(pd.to_datetime(Daytime['Date']))).drop('Date',axis=1)


        # self.Format(pd.read_csv(Daytime,delimiter = ',',skiprows=0,parse_dates={'datetime':[0]},header=1),v=0,drop=[0])
        Flux_10['Hz']=10
        Flux_1['Hz'] = 1
        Flux = Flux_1.append(Flux_10)
        Met = self.Format(pd.read_csv(Met,delimiter = ',',skiprows = 1,parse_dates={'datetime':[0]},header = 0),v=2,drop = [0])
        Soil = self.Format(pd.read_csv(Soil,delimiter = ',',skiprows = 0,parse_dates={'datetime':[0]},header = 0),v=0,drop = [0])
        BL = self.Format(pd.read_csv('C:/FishIsland_2017/BL_Data/PBLH_GFS.csv'),v=0,drop=[0])
        self.RawData = pd.concat([Flux,Met,Soil,BL],axis = 1, join = 'outer')
        self.RawData = self.RawData.join(Daytime,how='inner')
        self.RawData['Daytime'] = np.ceil(self.RawData['Daytime'])

        if Drop_Variables != None:
            self.RawData = self.RawData.drop(Drop_Variables,axis=1)

        self.RawData['Date_Key'] = 0
        for var in self.Fluxes:
            self.RawData[var+'_drop'] = 0
        self.Mt = pytz.timezone('US/Mountain')
        self.RawData['UTC'] = self.RawData.index.tz_localize(self.Mt)#.tz_convert(pytz.utc)
        self.RawData = self.RawData.set_index(self.RawData['UTC'])
        self.RawData['Minute'] = self.RawData.index.hour*60+self.RawData.index.minute
        self.RawData['Day'] = np.floor(self.RawData['DOY'])
        self.uThresh = .1
        self.RawData['Radians'] = self.RawData['wind_dir']/180*np.pi
        self.RawData['North'] = self.RawData['wind_speed'] * np.cos(self.RawData['Radians'])
        self.RawData['East'] = self.RawData['wind_speed'] * np.sin(self.RawData['Radians']) 
        self.RawData['N']=self.RawData['North']/(self.RawData['North']**2+self.RawData['East']**2)**.5
        self.RawData['E']=self.RawData['East']/(self.RawData['North']**2+self.RawData['East']**2)**.5
        self.RawData['Wind_Direction'] = self.RawData['wind_dir']
        self.RawData.loc[self.RawData['wind_dir']>215,'Wind_Direction'] -= 360
        self.RawData.loc[self.RawData['flowrate_mean']<0.0002,'co2_flux'] = np.nan
        self.Data=self.RawData.copy()
        
    def Format(self,df,v,drop):
        df = df.iloc[v:]
        df = df.set_index(pd.DatetimeIndex(df.datetime))
        df.datetime = df.index
        df.datetime = df.datetime.apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,30*(dt.minute // 30)))
        df = df.set_index(pd.DatetimeIndex(df.datetime))
        # if drop
        df = df.drop(df.columns[drop],axis=1)
        df = df.astype(float)
        return(df)
    
    def Date_Drop(self,Dates,start):
        for Date in Dates:
            self.Data.loc[(self.Data.index>=Date[0])&(self.Data.index<=Date[1]),self.Fluxes]=np.nan
        # Others = self.Data.loc[((np.isnan(self.Data['PPFD_Avg'])==True)&(np.isnan(self.Data['file_records'])==True))].index.values
        # for O in Others:
        self.Data = self.Data.drop(self.Data.loc[self.Data.index.tz_localize(None)<start].index)
        # self.Data=self.RawData.copy()

    # def Date_Key(self,Date,key):
    #     self.Data.loc[(self.Data.index>Date[0])&(self.Data.index<Date[1]),'Date_Key'] = key
    #     # self.Data['biWeek'] = int(self.Data.index.week.values/2)
    #     self.Data['Month'] = self.Data.index.month
        
            
    def Wind_Bins(self,Bins):
        self.bins = np.arange(0,360.1,Bins)
        self.Data['Dir'] = pd.cut(self.Data['wind_dir'],bins=self.bins,labels = (self.bins[0:-1]+self.bins[1:])/2)
        
    def ustar_Bins(self,Bins,LightFilter = {'Var':'PPFD_Avg','Thresh':10},
               uFilter={'Var':'co2_flux','Plot':False},BootStraps={'Repetitions':100,'n_samples':10000}):
        def Rcalc(Grp,thrsh=0.95):
            Ratios=[]
            for G in pd.to_numeric(Grp.index).values:#Grp.index:
                m1 = Grp[uFilter['Var']][pd.to_numeric(Grp.index)==G].values[0]
                m2 = Grp[uFilter['Var']][pd.to_numeric(Grp.index)>G].mean()
                Ratios.append(m1/m2)
            Ratios = np.asanyarray(Ratios)
            Ratios[np.where(np.isnan(Ratios)==True)[0]]=1
            try:
                idx = pd.to_numeric(Grp.index).values
                uThresh = idx[np.where(Ratios>=.95)[0]][0]
            except:
                print('Could not find u* thersh, defaulting to 0.1')
                uThresh = 0.1
            return(uThresh)
        self.uFilterData = self.Data[self.Data[LightFilter['Var']]<=LightFilter['Thresh']].copy()
        self.bins = self.uFilterData['u*'].quantile(np.arange(0,Bins,1)/Bins).values
        # print(self.uFilterData)
        # print(self.Data)
        self.uFilterData['u*bin'] = pd.cut(self.uFilterData['u*'],bins=self.bins,labels = (self.bins[0:-1]+self.bins[1:])/2)

        Grp = self.uFilterData.groupby(['u*bin']).mean()
        GrpC = self.uFilterData.groupby(['u*bin']).size()
        GrpSE = self.uFilterData.groupby(['u*bin'])['co2_flux'].std()/(GrpC)**.5
        self.uThresh_SampSize = GrpC.sum()
        
        self.uThresh = Rcalc(Grp)
        self.BootStraps = {}
        for i in range(BootStraps['Repetitions']):
            Samp = resample(self.Data,replace=True,n_samples=BootStraps['n_samples'])
            Samp = Samp[Samp[LightFilter['Var']]<=LightFilter['Thresh']]
            bins = Samp['u*'].quantile(np.arange(0,Bins,1)/Bins).values
            Samp['u*bin'] = pd.cut(Samp['u*'],bins=bins,labels = (bins[0:-1]+bins[1:])/2)
            self.BootStraps[str(i)] = Samp
        Ge = []
        for i in self.BootStraps:
            G = self.BootStraps[i].groupby(['u*bin']).mean()
            Ge.append(Rcalc(G))
        Ge = np.asanyarray(Ge)
        self.Pct = {'5%':np.percentile(Ge,[5]),'50%':np.percentile(Ge,[50]),'95%':np.percentile(Ge,[95])}
        self.uThresh = Ge.mean()
        if uFilter['Plot'] == True:
            # plt.figure(figsize=(6,5))
            # plt.errorbar(Grp['u*'],Grp[uFilter['Var']],yerr=GrpSE,label = 'Mean +- 1SE')
            # plt.hist(Ge,bins=30,density=True)
            ymin, ymax = plt.ylim()
            def Vlines(var,c,l):
                plt.plot([var,var],[ymin,ymax],
                         color = c,label=l,linewidth=5)

                # plt.plot([var,var],[Grp[uFilter['Var']].min(),Grp[uFilter['Var']].max()],
                #          color = c,label=l,linewidth=5)
            Vlines(self.uThresh,c='red',l='Mean')
            Vlines(self.Pct['5%'],c='green',l='5%')
            Vlines(self.Pct['50%'],c='yellow',l='50%')
            Vlines(self.Pct['95%'],c='blue',l='95%')
            plt.legend()
            plt.title('u* Thershold & Bootstrapped 95% CI')
            plt.grid()
        
    def PPFD_Bins(self,Bins):
        self.bins = np.arange(0,self.Data['PPFD_Avg'].max()+1,Bins)
        self.Data['Photon_Flux'] = pd.cut(self.Data['PPFD_Avg'],bins=self.bins,labels = (self.bins[0:-1]+self.bins[1:])/2)

    def Rain_Check(self,thresh):
        # self.Data['Rain_diff'] = self.Data['Rain_mm_Tot'].diff()
        for var in self.Fluxes:
            if var!='ch4_flux':
                self.Data.loc[self.Data['Rain_mm_Tot']>thresh[0],[var,var+'_drop']]=[np.nan,1]
            else:
                self.Data.loc[self.Data['Rain_mm_Tot']>thresh[1],[var,var+'_drop']]=[np.nan,1]
        
    def Spike_Removal(self,z_thresh,AltData=None,var=None):
        def Remove(series):
            di1 = series.diff()
            di1[:-1] = di1[1:]
            di = di1.diff()
            MD = di.median()
            MAD = np.abs(di-MD).median()
            F1 = di<MD-(z_thresh*MAD/0.6745)
            F2 = di>MD+(z_thresh*MAD/0.6745)
            series.loc[F1==True]=np.nan
            series.loc[F2==True]=np.nan
            Droppers = series.index[np.isnan(series)==True]
            VAR = self.Data[var].copy()
            VAR.loc[VAR.index.isin(Droppers)] = np.nan
            dina = VAR.diff()
            dina[:-1] = dina[1:]
            dina2 = VAR.diff()
            NaMid = VAR.index[((np.isnan(dina)==True)&(np.isnan(dina2)==True))]
            VAR.loc[VAR.index.isin(NaMid)] = np.nan
            return(VAR)       
        
        if AltData == None and var == None:
            for var in self.Fluxes:
                self.Data[var+'_PrSpk'] = self.Data[var].copy()
                # Temp = self.Data.loc[self.Data.daytime<1,var]
                # Temp = Remove(Temp.dropna())
                # self.Data.loc[self.Data.daytime<1,var] = Temp

                # Temp = self.Data.loc[self.Data.daytime>0,var]
                # Temp = Remove(Temp.dropna())
                # self.Data.loc[self.Data.daytime>0,var] = Temp
                self.Data[var]=Remove(self.Data[var].dropna())
        elif AltData == None:
#             for var in self.Fluxes:
            self.Data[var+'_PrSpk'] = self.Data[var].copy()
            self.Data[var]=Remove(self.Data[var].dropna())
        else:
            Data[var+'_PrSpk'] = self.Data[var].copy()
            AltData[var]=Remove(self.AltData[var].dropna())
            return(AltData[0])
        
    def Wind_Filter(self,width,angle):
        for var in self.Fluxes:
            self.Data.loc[((self.Data['wind_dir']>angle-width)&(self.Data['wind_dir']<angle+width)),[var,var+'_drop']]=[np.nan,1]
        
    def StorageCorrection(self,Raw=True):
        self.Data['co2_raw'] = self.Data['co2_flux']+0.0
        self.Data['ch4_raw'] = self.Data['ch4_flux']+0.0
        self.Data['co2_flux'] = self.Data['co2_flux']+self.Data['co2_strg']
        self.Data['ch4_flux'] = self.Data['ch4_flux']+self.Data['ch4_strg']
        
    def Signal_Check(self,RSSI_thresh=10,NoSignal_Thresh=.01):
        self.Data['ch4_noSSFilter'] = self.Data['ch4_flux']
        self.Data.loc[self.Data['rssi_77_mean']<RSSI_thresh,['ch4_flux','ch4_flux_drop']] = [np.nan,1]
        self.Data.loc[self.Data['no_signal_LI-7700']/18000>NoSignal_Thresh,['ch4_flux','ch4_flux_drop']] = [np.nan,1]
    
    def QC_Check(self,thresh):
        for var in self.Fluxes:
            self.Data[var+'_PrQC'] = self.Data[var].copy()
            self.Data.loc[self.Data['qc_'+var]>=thresh,[var,var+'_drop']]=[np.nan,1]
            self.Data.loc[np.isnan(self.Data[var]) == True,[var+'_drop']]=1
            
    def Ustar_Drop(self,Override=None,Drop=.25):
        Temp = self.Data[['u*','wind_speed']].dropna()
        y = Temp['u*']#np.random.random(10)
        x = Temp['wind_speed']#np.random.random(10)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        self.Data.loc[self.Data['u*']>self.Data['wind_speed']*slope+intercept+Drop,'u*']=np.nan


        if Override != None:
            self.uThresh = Override
        for var in self.Fluxes:
            self.Data[var+'_Pru*'] = self.Data[var].copy()
            self.Data.loc[self.Data['u*']<self.uThresh,[var,var+'_drop']]=[np.nan,1]
            self.Data.loc[np.isnan(self.Data['u*'])==True,[var,var+'_drop']]=[np.nan,1]
         #    self.Data.loc[((self.Data['u*']/self.Data['wind_speed']>(self.Data['u*']/self.Data['wind_speed']).quantile(.99))&\
         # (self.Data['u*']>self.uThresh)),['u*',var,var+'_drop']]=[np.nan,np.nan,1]
        # self.StorageCorrection(Raw=False)
        
    def CustomVars(self,Hours=24):
        self.Data['Total_Rain_mm_Tot'] = self.Data['Rain_mm_Tot'].rolling(str(Hours)+'H').sum()
        self.Data['Total_PPFD'] = self.Data['PPFD_Avg'].rolling(str(Hours)+'H').sum()
        self.Data['Ratio'] = self.Data['u*']/self.Data['wind_speed']
        # self.Data['Delta_Table_1'] = self.Data['Table_1'].diff()
        self.Data['Delta_air_pressure'] = self.Data['air_pressure'].diff()
        self.Data['Delta_Table_1'] = self.Data['Table_1'].rolling(str(Hours)+'H').mean()-self.Data['Table_1']
        self.Data['Delta_VWC_1'] = self.Data['VWC_1'].rolling(str(Hours)+'H').mean()-self.Data['VWC_1']
        self.Data['Delta_VWC_2'] = self.Data['VWC_2'].rolling(str(Hours)+'H').mean()-self.Data['VWC_2']

        self.Data['fco2']=self.Data['co2_flux']
        self.Data['fch4']=self.Data['ch4_flux']
        self.Data['DOY'] = self.Data.index.dayofyear
        self.Data['ER'] = self.Data['co2_flux']
        self.Data.loc[self.Data['Daytime']>0,'ER']=np.nan
        # self.Data['Delta_air_pressure'] = self.Data['air_pressure'].rolling(str(Hours)+'H').mean()-self.Data['air_pressure']
        try:
            self.Data['Total_Rainfall_Tot'] = self.Data['Rainfall_Tot'].rolling(str(Hours)+'H').sum()
            self.Data['Delta_SoilMoist(1)'] = self.Data['SoilMoist(1)'].diff(Hours)
            self.Data['Delta_SoilMoist(2)'] = self.Data['SoilMoist(2)'].diff(Hours)
            self.Data['Delta_SoilMoist(3)'] = self.Data['SoilMoist(3)'].diff(Hours)
            self.Data['Delta_SoilMoist(4)'] = self.Data['SoilMoist(4)'].diff(Hours)
            self.Data['Delta_SoilMoist(5)'] = self.Data['SoilMoist(5)'].diff(Hours)
        except:
            pass

    def Soil_Data_Avg(self,ratios=[.8,.2]):
        self.Data['Ts 2.5cm'] = self.Data['Temp_2_5_1']*ratios[0]+self.Data['Temp_2_5_2']*ratios[1]
        self.Data['Ts 5cm'] = self.Data['Temp_5_1']*ratios[0]+self.Data['Temp_5_2']*ratios[1]
        self.Data['Ts 15cm'] = self.Data['Temp_15_1']*ratios[0]+self.Data['Temp_15_2']*ratios[1]

    def Merge(self):#,Vars,Aliases):
        # self.Data[Aliases]=self.Data[Vars]

        if self.Taglu is not None:
            self.dfTaglu = pd.read_csv(self.Taglu,parse_dates=['datetime'],index_col=['datetime'])

            for V in ['SoilMoist(1)','SoilMoist(2)','SoilMoist(3)','SoilMoist(4)','SoilMoist(5)','SoilMoist(6)']:
                self.dfTaglu[V]=pd.to_numeric(self.dfTaglu[V])

            UTC = self.dfTaglu.index+timedelta(hours=6)
            self.dfTaglu = self.dfTaglu.set_index(UTC)
            self.dfTaglu.index = self.dfTaglu.index.tz_localize(pytz.utc).tz_convert(self.Mt)
            # self.dfTaglu['MT'] = self.dfTaglu.index.tz_localize(self.Mt,ambiguous=True)
            #,nonexistent='shift_forward')#.tz_convert(pytz.utc)


            # self.dfTaglu=self.dfTaglu.reset_index()
            # self.dfTaglu = self.dfTaglu.set_index(self.dfTaglu['MT'])
            self.dfTagluI=self.dfTaglu.resample('30T').interpolate()
            self.dfTagluI['Rainfall_Tot']=self.dfTaglu['Rainfall_Tot'].resample('30T').asfreq().fillna(0)
            self.Data_DS = self.Data.resample('h').mean()
            self.Data_DS['Rain_mm_Tot'] = self.Data.resample('h').sum()['Rain_mm_Tot']
            self.AllData = pd.concat([self.Data_DS,self.dfTaglu],axis=1,join='outer')
            self.Data = pd.concat([self.Data,self.dfTagluI],axis=1,join='inner')

            if self.NARR is not None:
                self.dfNARR = pd.read_csv(self.NARR,parse_dates=[0],index_col=[0])
                self.dfNARR.index.name = 'datetime'
                self.dfNARR[['soilw_0','soilw_10','soilw_40','soill_0','soill_10','soill_40']]*=100
                self.dfNARRH=self.dfNARR.resample('h').interpolate()
                self.dfNARRH['apcp']=self.dfNARR['apcp'].resample('h').asfreq().fillna(0)
                self.dfNARR=self.dfNARR.resample('30T').interpolate()
                self.dfNARR['apcp']=self.dfNARR['apcp'].resample('30T').asfreq().fillna(0)
                self.dfNARR['MT'] = self.dfNARR.index.tz_localize(pytz.utc).tz_convert(self.Mt)
                self.dfNARRH['MT'] = self.dfNARRH.index.tz_localize(pytz.utc).tz_convert(self.Mt)
                self.dfNARR=self.dfNARR.set_index(pd.DatetimeIndex((self.dfNARR['MT'])))
                self.dfNARRH=self.dfNARRH.set_index(pd.DatetimeIndex((self.dfNARRH['MT'])))
                self.AllData = pd.concat([self.AllData,self.dfNARRH],axis=1,join='outer')
                self.Data = pd.concat([self.Data,self.dfNARR],axis=1,join='inner')

            # self.AllData.resample('3h').mean().to_csv('C:/Users/wesle/NetworkAnalysis/FishIsland/FullDataset.csv')
            # self.VWC_Calc()

    # def VWC_Calc(self):
    #     def Curve(p,a,b,c,d):
    #         return(a*p**3+b*p**2+c*p**1+d)

    #     for p,sm in zip(('1','2'),('1','4')):
    #         P = 'Period_'+p
    #         SM = 'SoilMoist('+sm+')'
    #         # print(self.Data.columns)

    #         Temp = self.Data.loc[((np.isnan(self.Data[P])==False)&(np.isnan(self.Data[SM])==False))]
    #         popt_r, pcov = curve_fit(Curve,Temp[P],Temp[SM])
    #         print('VWC!!!')

    #         print(metrics.r2_score(Temp[SM],Curve(Temp[P],*popt_r)))
    #         # print(popt_r)
    #         # plt.figure()
    #         # plt.plot(Curve(Data[P],*popt_r),c='g')
    #         # plt.plot(Data[SM],c='k')
    #         self.Data['VWC_'+p]=Curve(self.Data[P],*popt_r)

    #         self.AllData['VWC_'+p]=Curve(self.AllData[P],*popt_r)


        # self.Data[Aliases].to_csv(Root+'FilteredData' +str(datetime.datetime.now()).split(' ')[0]+'.csv')
#         self.Data=self.Data.drop(Aliases,axis=1)
