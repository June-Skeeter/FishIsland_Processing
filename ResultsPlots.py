import pandas as pd 
import numpy as np 
import datetime as dt
from matplotlib import pyplot as plt 
import Plot_Tricks as pt
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from string import ascii_uppercase

plt.rc('axes', axisbelow=True)

MajorLine = 5
MinorLine = 1

ScatterSize = 20

DarkGreen = (.05,.5,.1)
LightGreen = (.25,.85,.35)

DarkRed = (.5,.1,.05)
LightRed = (.85,.35,.25)

DarkBlue = (.05,.1,.5)
LightBlue = (.25,.35,.85)

Gold = (1,.843,0)

umolPPFD = 'PPFD $umol\ m^{-2} s^{-1}$'
umolCO2 = '$umol\ CO_{2} m^{-2} s^{-1}$'
nmolCH4 = '$nmol\ CH_{4} m^{-2} s^{-1}$'


gCO2 = '$g\ CO_{2} m^{-2} s^{-1}$'
mgCH4 = '$mg\ CH_{4} m^{-2} s^{-1}$'
gCO2eq = '$g\ CO_{2} eq. m^{-2} s^{-1}$'

class Results:
	def __init__(self,DataPath,FillPath_CH4,SummaryPath_CH4,FillPath_CO2,SummaryPath_CO2,MajorFont,MinorFont):
		self.MinorFont = MinorFont
		self.MajorFont = MajorFont
		self.Data = pd.read_csv(DataPath)
		self.Data = self.Data.set_index(pd.DatetimeIndex(self.Data.datetime))
		self.Data['Month'] = self.Data.index.month
		self.Data['air pressure']*=1e-2
		self.Daily = self.Data.resample('D').mean()
		self.Daily['Day'] = self.Daily.index.dayofyear
		self.Daily = self.Daily[(self.Daily['Day']<self.Daily['Day'].max())&(self.Daily['Day']>self.Daily['Day'].min())]
		self.Daily['NaN'] = np.nan

		self.Filled_CH4 = pd.read_csv(FillPath_CH4)
		self.Filled_CH4 = self.Filled_CH4.set_index(pd.DatetimeIndex(self.Data.datetime))
		self.Filled_CH4['air pressure']*=1e-2
		self.DailyFilled_CH4 = self.Filled_CH4.resample('D').mean()
		self.DailyFilled_CH4['Day'] = self.DailyFilled_CH4.index.dayofyear
		self.DailyFilled_CH4 = self.DailyFilled_CH4[(self.DailyFilled_CH4['Day']<self.DailyFilled_CH4['Day'].max())&(self.DailyFilled_CH4['Day']>self.DailyFilled_CH4['Day'].min())]

		self.Filled_CO2 = pd.read_csv(FillPath_CO2)
		self.Filled_CO2 = self.Filled_CO2.set_index(pd.DatetimeIndex(self.Data.datetime))
		self.Filled_CO2['air pressure']*=1e-2
		self.DailyFilled_CO2 = self.Filled_CO2.resample('D').mean()
		self.DailyFilled_CO2['Day'] = self.DailyFilled_CO2.index.dayofyear
		self.DailyFilled_CO2 = self.DailyFilled_CO2[(self.DailyFilled_CO2['Day']<self.DailyFilled_CO2['Day'].max())&(self.DailyFilled_CO2['Day']>self.DailyFilled_CO2['Day'].min())]
		
		self.Daily['Fco2'] = self.DailyFilled_CO2['Model: Wind Spd+Ta+PPFD+VWC+Active Layer']*1e-6* 44.0095 *3600*24
		self.Daily['Fch4'] = self.DailyFilled_CH4['Model: Wind Spd+air pressure+PPFD+Active Layer+Water Table']*1e-6* 16.04246 *3600*24
		self.Daily['CO2eq'] = self.Daily['Fco2']+self.Daily['Fch4']*28*1e-3


		replace = {'PPFD':'PPFD','air pressure':'P$_a$','Active Layer':'Thaw Dpth','Water Table':'Wtr Tbl','Ta':'T$_a$','Ts 15 cm':'T$_s$ 15cm','Ts 2.5 cm':'T$_s$ 2.5cm','VWC':'VWC','Wind Spd':'Wind Spd','Rain':'Rain'}
		self.Summary_CH4 = pd.read_csv(SummaryPath_CH4,index_col=0)
		self.Summary_CH4['NewVar']=''
		self.Summary_CH4['Model Name']=''
		for index,row in self.Summary_CH4.iterrows():
			varz = set(row['Models'].split(': ')[-1].split('+'))
			if index > 1:
				self.Summary_CH4['NewVar'].iloc[index-1]= replace[list(varz - varz2)[0]]
				self.Summary_CH4['Model Name'].iloc[index-1]=ascii_uppercase[index-1]
			else:
				self.Summary_CH4['NewVar'].iloc[index-1]=replace[list(varz)[0]]
				self.Summary_CH4['Model Name'].iloc[index-1]=ascii_uppercase[index-1]
			varz2=varz

		self.Summary_CO2 = pd.read_csv(SummaryPath_CO2,index_col=0)
		self.Summary_CO2['NewVar']=''
		self.Summary_CO2['Model Name']=''
		for index,row in self.Summary_CO2.iterrows():
			varz = set(row['Models'].split(': ')[-1].split('+'))
			if index > 1:
				self.Summary_CO2['NewVar'].iloc[index-1]= replace[list(varz - varz2)[0]]
				self.Summary_CO2['Model Name'].iloc[index-1]=ascii_uppercase[index-1]
			else:
				self.Summary_CO2['NewVar'].iloc[index-1]= replace[list(varz)[0]]
				self.Summary_CO2['Model Name'].iloc[index-1]=ascii_uppercase[index-1]
			varz2=varz

		self.Chamber = pd.read_csv('C:/FishIsland_2017/ChamberFluxes.csv').dropna()
		# plt.figure()
		self.Boxes = []
		self.Box_label = []
		for t in self.Chamber['Type'].unique():
			self.Box_label.append(t)
			self.Boxes.append(self.Chamber['Flux'].loc[self.Chamber['Type']==t])

	def Climate(self,ax):
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		rect1 = [0.05,.14,.42,.77]
		rect2 = [.55,.14,.42,.77]
		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)

		ax1.plot(self.Daily.index,self.Daily['Ta'],color = LightRed, label = "$T_a$",linewidth=MajorLine)
		ax1.plot(self.Daily.index,self.Daily['Ts 2.5 cm'],color=DarkRed, label = "$T_p$ 2.5 cm",linewidth=MajorLine)
		ax1.plot(self.Daily.index,self.Daily['Ts 15 cm'],color=DarkRed,linestyle=':', label = "$T_p$ 15 cm",linewidth=MajorLine)
		
		ax2.plot(self.Daily['Active Layer']*-1,color = DarkRed,label = 'Thaw Depth',linewidth=MajorLine)
		ax2.plot(self.Daily['Water Table'],color = DarkBlue,label = 'Water Table Depth',linewidth=MajorLine)
		
		self.allfmt(ax1)
		self.allfmt(ax2)

		ax1.set_title('Daily Air and Peat Temperatures', fontsize=self.MajorFont)
		ax1.set_ylabel('$\circ C$',fontsize=self.MinorFont)
		plt.sca(ax1)
		plt.xticks(fontsize=self.MinorFont,rotation=20)

		ax2.set_title('Daily Water Table and Thaw Depth', fontsize=self.MajorFont)
		ax2.set_ylabel('Depth m',fontsize=self.MinorFont)
		plt.sca(ax2)
		plt.xticks(fontsize=self.MinorFont,rotation=20)

	def Four_Plots(self,ax,D3 = False):
		size = [.37,.37]
		rect1 = [0.055,.575,size[0],size[1]]
		rect2 = [0.555,.575,size[0],size[1]]
		rect3 = [0.055,.075,size[0],size[1]]
		rect4 = [0.555,.075,size[0],size[1]]
		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)
		if D3 == True:
			ax3 = pt.add_subplot_axes(ax,rect3,Proj='3d')
		else:
			ax3 = pt.add_subplot_axes(ax,rect3)
		ax4 = pt.add_subplot_axes(ax,rect4)
		return(ax1,ax2,ax3,ax4)

	def allfmt(self,ax,legend=True,loc=1):
		if legend == True:
			ax.legend(fontsize=self.MinorFont,loc=loc)
		ax.grid()
		plt.sca(ax)
		plt.xticks(fontsize=self.MinorFont)
		plt.yticks(fontsize=self.MinorFont)

	def GPP_ER(self,ax):
		ax1,ax2,ax3,ax4 = self.Four_Plots(ax,D3=False)
		ax2.boxplot(self.Boxes)

		Colors = [LightBlue,LightGreen,LightRed,Gold]
		Labels = ['June','July','August','September']
		for i,mo in enumerate(self.Daily.Month.unique()):
			Subset = self.Data[self.Data.Month == mo].copy()
			ax1.plot(Subset['Ta'],Subset['ER'],color=Colors[i], label=Labels[i],linewidth = 5)
			if i == 3:
				Labels[3]+='*'
			Subset.sort_values(by = 'GPP',inplace=True)
			ax3.plot(Subset['PPFD'],Subset['GPP'],color = Colors[i],label=Labels[i],linewidth = 5)
			Subset.sort_values(by='ER',inplace=True)


		Score2 = self.Data.sort_values(by='fco2',inplace=False)
		ax4.plot(Score2['fco2'],Score2['fco2'],label='1:1',color='black',linewidth = 1)
		Score = self.Data[['fco2','NEE']].dropna()
		LR = stats.linregress(Score['fco2'],Score['NEE'])
		Line = LR[0]*Score['fco2']+LR[1]
		ax4.scatter(Score['fco2'],Score['NEE'],color = LightGreen,label=None,s=ScatterSize)
		ax4.plot(Score['fco2'],Line,label = '$r^2$: '+str(np.round(LR[2]**2,2)),linewidth=MajorLine,color=DarkGreen)

		self.allfmt(ax1,loc=1)
		self.allfmt(ax2,legend=False)
		self.allfmt(ax3)
		self.allfmt(ax4)

		ax1.set_title('ER vs. T$_a$',fontsize=self.MajorFont)
		ax1.legend(fontsize=self.MinorFont)
		ax1.set_ylabel(umolCO2,fontsize=self.MinorFont)
		ax1.set_xlabel("$\circ C$",fontsize=self.MinorFont)

		ax2.set_title('Flux Chamber ER',fontsize=self.MajorFont)
		ax2.set_ylabel(umolCO2,fontsize=self.MinorFont)
		ax2.set_xticklabels(self.Box_label,fontsize=self.MinorFont,rotation=15)

		ax3.set_title('GPP vs. PPFD',fontsize=self.MajorFont)
		ax3.set_ylabel(umolCO2,fontsize=self.MinorFont)
		ax3.set_xlabel(umolPPFD,fontsize=self.MinorFont)

		ax4.set_title('NEE: Modeled vs. Observed',fontsize=self.MajorFont)
		ax4.legend(fontsize=self.MinorFont)
		ax4.set_ylabel('Modeled '+umolCO2,fontsize=self.MinorFont)
		ax4.set_xlabel('Observed '+umolCO2,fontsize=self.MinorFont)

	def NN_Style(self,ax,ax1,ax2,ax3,ax4,Var,cbobj1,cbobj2):
		ax1.set_title('Relative Model Performace',fontsize=self.MajorFont)
		ax1.set_ylabel('Normalized MSE',fontsize = self.MinorFont)
		ax1.set_xticks(np.arange(1,11))
		ax1.set_xlabel('Model Size',fontsize=self.MinorFont)

		rect = [0.95,.65,.01,.25]
		cbax1 = pt.add_subplot_axes(ax,rect)
		self.fig.colorbar(cbobj1,cax=cbax1)

		rect = [0.45,.15,.01,.25]
		cbax2 = pt.add_subplot_axes(ax,rect)
		self.fig.colorbar(cbobj2,cax=cbax2)

		ax4.set_title('Best Model Performace',fontsize=self.MajorFont)

		self.allfmt(ax1)
		self.allfmt(ax2)
		self.allfmt(ax3,legend=False)
		self.allfmt(ax4)

		if Var == 'co2':
			ax1.set_ylim(0.05,0.3)
			ax2.set_title('$F_{CH_4}$ vs. PPFD',fontsize=self.MajorFont)
		else:
			ax1.set_ylim(0.26,0.84)
			ax2.set_title('$F_{CH_4}$ vs. PPFD',fontsize=self.MajorFont)

	def Normalize(self,a,b,Data,Name):
		c = np.abs(Data[[a,b]].min().values.min())+1
		Data[[a,b]]+=c
		Data[Name] = (Data[b] - Data[a])/(Data[b] + Data[a])
		return(Data[Name])


	def CO2(self,ax,fig):
		self.fig = fig
		ax1,ax2,ax3,ax4 = self.Four_Plots(ax)

		Best_Model = self.Summary_CO2.loc[self.Summary_CO2['MSE']==self.Summary_CO2['MSE'].min()]
		BM = Best_Model['Models'].values[0]
		BM = 'Model: Wind Spd+Ta+PPFD+VWC+Active Layer'

		ax1.bar(self.Summary_CO2.index,self.Summary_CO2['MSE'],color = DarkGreen)
		ax1.errorbar(self.Summary_CO2.index,self.Summary_CO2['MSE'],self.Summary_CO2['CI'],fmt = 'o',
			color='black',label = '95% CI')
		ax1.annotate('Most Parsimonious Model', xy=(5, .18), xytext=(5.5, .25),
            arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='center',fontsize=self.MinorFont-2)

		y = .055
		for index,row in self.Summary_CO2.iterrows():
			if index > 1:
				ax1.text(index,y,row['Model Name']+') '+lastModel+'+'+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			else:
				ax1.text(index,y,row['Model Name']+') '+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			lastModel = row['Model Name']


		a,b,Name='Model: PPFD','Model: Ta+PPFD+Active Layer','Normed_3_1'
		self.Filled_CO2[Name]=self.Normalize(a,b,self.Filled_CO2.copy(),Name)
		a,b,Name='Model: Ta+PPFD+Active Layer','Model: Wind Spd+Ta+PPFD+VWC+Active Layer','Normed_5_3'
		self.Filled_CO2[Name]=self.Normalize(a,b,self.Filled_CO2.copy(),Name)

		Subset = self.Filled_CO2[self.Filled_CO2[Name].between(self.Filled_CO2[Name].quantile(.05), self.Filled_CO2[Name].quantile(.95), inclusive=True)]
		cbobj1=ax2.scatter(Subset['Active Layer'],Subset['Ta'],c=Subset[Name],cmap = 'bwr',s=ScatterSize,
			vmin=-((Subset[Name]**2)**.5).max(),vmax=((Subset[Name]**2)**.5).max())
		cbobj2=ax3.scatter(Subset['Wind Spd'],Subset['VWC'],c=Subset[Name],cmap = 'bwr',s=ScatterSize,
			vmin=-((Subset[Name]**2)**.5).max(),vmax=((Subset[Name]**2)**.5).max())
		# cbobj1=ax2.scatter(self.Filled_CO2['Active Layer'],self.Filled_CO2['Ta'],c=self.Filled_CO2[Name],cmap = 'bwr',s=ScatterSize)
		# cbobj2=ax3.scatter(self.Filled_CO2['Wind Spd'],self.Filled_CO2['VWC'],c=self.Filled_CO2[Name],cmap = 'bwr',s=ScatterSize)

		Score = self.Filled_CO2[['fco2',BM]].dropna()
		LR = stats.linregress(Score['fco2'].values,Score[BM].values)
		Line = LR[0]*Score['fco2']+LR[1]
		ax4.plot(self.Filled_CO2['fco2'],self.Filled_CO2['fco2'],label='1:1',color='black',linewidth=MinorLine)
		ax4.scatter(self.Filled_CO2['fco2'],self.Filled_CO2[BM],label=None,color=LightGreen,s=ScatterSize)
		ax4.plot(Score['fco2'],Line,color = DarkGreen,label='$r^2$: '+str(np.round(LR[2]**2,2)),linewidth=MajorLine)

		self.NN_Style(ax,ax1,ax2,ax3,ax4,'co2',cbobj1,cbobj2)

	def CH4(self,ax,fig):
		self.fig = fig
		ax1,ax2,ax3,ax4 = self.Four_Plots(ax)

		Best_Model = self.Summary_CH4.loc[self.Summary_CH4['MSE']==self.Summary_CH4['MSE'].min()]
		BM = Best_Model['Models'].values[0]
		BM = 'Model: Wind Spd+air pressure+PPFD+Active Layer+Water Table'

		ax1.bar(self.Summary_CH4.index,self.Summary_CH4['MSE'],color = DarkRed)
		ax1.errorbar(self.Summary_CH4.index,self.Summary_CH4['MSE'],self.Summary_CH4['CI'],fmt = 'o',
			color='black',label = '95% CI')
		ax1.annotate("Most Parsimonious Model",
		 xy=(5, .59), xytext=(6.1, .68),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='center',fontsize=self.MinorFont)
		
		y = .27
		for index,row in self.Summary_CH4.iterrows():
			if index > 1:
				ax1.text(index,y,row['Model Name']+') '+lastModel+'+'+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			else:
				ax1.text(index,y,row['Model Name']+') '+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			lastModel = row['Model Name']


		a,b,Name='Model: PPFD','Model: Wind Spd+PPFD+Active Layer','Normed_3_1'
		self.Filled_CH4[Name]=self.Normalize(a,b,self.Filled_CH4.copy(),Name)
		a,b,Name='Model: Wind Spd+PPFD+Active Layer','Model: Wind Spd+air pressure+PPFD+Active Layer+Water Table','Normed_5_3'
		self.Filled_CH4[Name]=self.Normalize(a,b,self.Filled_CH4.copy(),Name)
		
		Subset = self.Filled_CH4[self.Filled_CH4[Name].between(self.Filled_CH4[Name].quantile(.05), self.Filled_CH4[Name].quantile(.95), inclusive=True)]
		cbobj1=ax2.scatter(Subset['Active Layer'],Subset['Wind Spd'],c=Subset[Name],cmap = 'bwr',s=ScatterSize,
			vmin=-((Subset[Name]**2)**.5).max(),vmax=((Subset[Name]**2)**.5).max())
		cbobj2=ax3.scatter(Subset['Water Table'],Subset['air pressure'],c=Subset[Name],cmap = 'bwr',s=ScatterSize,
			vmin=-((Subset[Name]**2)**.5).max(),vmax=((Subset[Name]**2)**.5).max())
		# cbobj1=ax2.scatter(self.Filled_CH4['Active Layer'],self.Filled_CH4['Wind Spd'],c=self.Filled_CH4[Name],cmap = 'bwr',s=ScatterSize)
		# cbobj2=ax3.scatter(self.Filled_CH4['Water Table'],self.Filled_CH4['air pressure'],c=self.Filled_CH4[Name],cmap = 'bwr',s=ScatterSize)

		Score = self.Filled_CH4[['fch4',BM]].dropna()
		LR = stats.linregress(Score['fch4'].values,Score[BM].values)
		Line = LR[0]*Score['fch4']+LR[1]
		ax4.plot(self.Filled_CH4['fch4'],self.Filled_CH4['fch4'],label='1:1',color='black',linewidth=MinorLine)
		ax4.scatter(self.Filled_CH4['fch4'],self.Filled_CH4[BM],label=None,color=LightRed,s=ScatterSize)
		ax4.plot(Score['fch4'],Line,color = DarkRed,label='$r^2$: '+str(np.round(LR[2]**2,2)),linewidth=MajorLine)

		self.NN_Style(ax,ax1,ax2,ax3,ax4,'ch4',cbobj1,cbobj2)

	def C_Balance(self,ax):

		rect1 = [.05,0.7,.9,0.25]
		rect2 = [.05,0.375,.9,0.25]
		rect3 = [.05,0.05,.9,0.25]
		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)
		ax3 = pt.add_subplot_axes(ax,rect3)

		co2Mean = self.Daily['Fco2'].mean()
		self.Daily['Fco2_mean'] = co2Mean
		ch4Mean = self.Daily['Fch4'].mean()
		self.Daily['Fch4_mean'] = ch4Mean
		co2eqMean = self.Daily['CO2eq'].mean()
		self.Daily['CO2eq_mean'] = co2eqMean


		co2Std = self.Daily['Fco2'].std()
		ch4Std = self.Daily['Fch4'].std()
		co2eqStd = self.Daily['CO2eq'].std()

		c = self.Daily['Fch4_mean'].count()


		co2CI = pt.round_sigfigs(co2Std/(c)**.5 *1.96,2)
		ch4CI = pt.round_sigfigs(ch4Std/(c)**.5 *1.96,2)
		co2eqCI = pt.round_sigfigs(co2eqStd/(c)**.5 *1.96,2)

		print(co2eqStd/(c)**.5 *1.96)

		co2Mean = pt.round_sigfigs(co2Mean,2)
		ch4Mean = pt.round_sigfigs(ch4Mean,2)
		co2eqMean = pt.round_sigfigs(co2eqMean,2)

		ax1.bar(self.Daily.index,self.Daily['Fco2'],color = DarkGreen,edgecolor='black')
		ax1.plot(self.Daily.index,self.Daily['Fco2_mean'],color='black',label='Mean :'+str(co2Mean) +gCO2+\
			' 95% CI $\pm$ '+str(co2CI),linewidth=MajorLine)

		ax2.bar(self.Daily.index,self.Daily['Fch4'],color = DarkRed,edgecolor='black')
		ax2.plot(self.Daily.index,self.Daily['Fch4_mean'],color='black',label='Mean :'+str(ch4Mean) +mgCH4+\
			' 95% CI $\pm$ '+str(ch4CI),linewidth=MajorLine)
		
		ax3.bar(self.Daily.index,self.Daily['CO2eq'],color = Gold,edgecolor='black')
		ax3.plot(self.Daily.index,self.Daily['CO2eq_mean'],color='black',label='Mean :'+str(co2eqMean) +gCO2eq+\
			' 95% CI $\pm$ '+str(co2eqCI),linewidth=MajorLine)



		for cax in [ax1,ax2,ax3]:
			cax.set_xlim(self.Daily.index[0]-pd.offsets.Day(1),self.Daily.index[-1]+pd.offsets.Day(1))


		self.allfmt(ax1,loc=4)
		self.allfmt(ax2,loc=1)
		self.allfmt(ax3,loc=4)

		ax1.set_xticklabels([])
		ax2.set_xticklabels([])

		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		ax1.set_title('NEE',fontsize=self.MajorFont,loc='left')
		ax2.set_title('NME',fontsize=self.MajorFont,loc='left')
		ax3.set_title('NEE + NMEx28',fontsize=self.MajorFont,loc='left')

		ax1.set_ylabel(gCO2,fontsize=self.MajorFont)
		ax2.set_ylabel(mgCH4,fontsize=self.MajorFont)
		ax3.set_ylabel(gCO2eq,fontsize=self.MajorFont)


