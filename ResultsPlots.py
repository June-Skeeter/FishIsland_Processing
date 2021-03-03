import pandas as pd 
import numpy as np 
import datetime as dt
from matplotlib import pyplot as plt 
import Plot_Tricks as pt
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from string import ascii_uppercase

import matplotlib as mpl
mpl.rcParams["mathtext.default"] = 'regular'

# mathtext.default rcParam

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

umolPPFD = 'PPFD $\mu mol\ m^{-2} s^{-1}$'
umolCO2 = '$\mu mol\ CO_{2} m^{-2} s^{-1}$'
nmolCH4 = '$nmol\ CH_{4} m^{-2} s^{-1}$'


gCO2 = '$g\ CO_{2} m^{-2} d^{-1}$'
mgCH4 = '$mg\ CH_{4} m^{-2} d^{-1}$'
gCO2eq = '$g\ CO_{2} eq. m^{-2} d^{-1}$'

class Results:
	def __init__(self,DataPath,FillPath_CH4,SummaryPath_CH4,FillPath_CO2,SummaryPath_CO2,MajorFont,MinorFont):
		self.MinorFont = MinorFont
		self.MajorFont = MajorFont
		self.Data = pd.read_csv(DataPath)
		self.Data = self.Data.set_index(pd.DatetimeIndex(self.Data.datetime))
		self.Data['Month'] = self.Data.index.month
		self.Data['air pressure']*=1e-2

		self.Wind_Groups = self.Data.groupby(['Dir']).size()

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


		replace = {'PPFD':'PPFD','air pressure':'P$_a$','Active Layer':'Thaw Dpth','Water Table':'Wtr Tbl',
		'Ta':'T$_a$','Ts 15 cm':'T$_s$ 15cm','Ts 2.5 cm':'T$_s$ 2.5cm','VWC':'SWC','Wind Spd':'Wind Spd','Rain':'Rain'}
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

		xscale = .78
		xpad = .075
		yscale = .79
		ypad = .07
		y = ypad
		x = np.linspace(0.01,.68,3)
		xs,ys = x[1]*xscale,yscale
		x,y=x+xpad,y+ypad


		rect1 = [x[0]-.0075,y,xs,ys]
		rect2 = [x[1],y,xs,ys]
		rect3 = [x[2]-xpad*.5,y,xs,ys*.93]
		
		# rect1 = [0.05,.14,.42,.77]
		# rect2 = [.55,.14,.42,.77]
		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)
		ax3 = pt.add_subplot_axes(ax,rect3,Proj='polar')

		ax1.plot(self.Daily.index,self.Daily['Ta'],color = LightRed, label = "$T_a$",linewidth=MajorLine)
		ax1.plot(self.Daily.index,self.Daily['Ts 2.5 cm'],color=DarkRed, label = "$T_p$ 2.5 cm",linewidth=MajorLine)
		ax1.plot(self.Daily.index,self.Daily['Ts 15 cm'],color=DarkRed,linestyle=':', label = "$T_p$ 15 cm",linewidth=MajorLine)
		
		ax2.plot(self.Daily['Active Layer']*-1,color = DarkRed,label = 'Thaw Depth',linewidth=MajorLine)
		ax2.plot(self.Daily['Water Table'],color = DarkBlue,label = 'Water Table Depth',linewidth=MajorLine)

		# print(self.Wind_Groups)
		ax3.bar(pd.to_numeric(self.Wind_Groups.index)*np.pi/180,self.Wind_Groups.values/self.Wind_Groups.values.sum(),width = 30*np.pi/180, edgecolor = 'black')


		ax3.set_theta_direction(-1)
		ax3.set_theta_offset(0)
		ax3.set_theta_zero_location('N')
		ax3.set_thetagrids([0,45,90,135,180,225,270,315],['N','NE','E','SE','S','SW','W','NW'],fontsize=self.MinorFont)
		ax3.set_title('Wind Frequency Distribution %',fontsize = self.MajorFont,y = 1.075)
		ax3.set_rlabel_position(215)
		ax3.set_yticks([.05,.1,.15,.2])
		ax3.set_yticklabels(['5%','10%','15%','20%'],fontsize = self.MinorFont)
		
		self.allfmt(ax1)
		self.allfmt(ax2,loc=5)

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
		rect1 = [0.055,.555,size[0],size[1]]
		rect2 = [0.555,.555,size[0],size[1]]
		rect3 = [0.055,.055,size[0],size[1]]
		rect4 = [0.555,.055,size[0],size[1]]
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
		# ax1,ax2,ax3,ax4 = self.Four_Plots(ax,D3=False)


		xscale = .8
		xpad = .055
		yscale = .79
		ypad = .06
		x = np.linspace(0,.5,2)
		y = np.linspace(0,.5,2)
		xs,ys = x[1]*xscale,y[1]*yscale
		x,y=x+xpad,y+ypad
		print(x,y)


		# rect1 = [x[0],y[2],xs,ys]
		# rect2 = [x[1],y[2],xs,ys]
		rect1 = [x[0],y[1],xs,ys]
		rect2 = [x[1],y[1],xs,ys]
		rect3 = [x[0],y[0],xs,ys]
		rect4 = [x[1],y[0],xs,ys]
		
		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)
		ax3 = pt.add_subplot_axes(ax,rect3)
		ax4 = pt.add_subplot_axes(ax,rect4)
		# ax5 = pt.add_subplot_axes(ax,rect5)
		# ax6 = pt.add_subplot_axes(ax,rect6)


		ax2.boxplot(self.Boxes)

		Colors = [LightBlue,LightGreen,LightRed,Gold]
		Labels = ['June','July','August','September']
		for i,mo in enumerate(self.Daily.Month.unique()):
			Subset = self.Data[self.Data.Month == mo].copy()
			ax1.plot(Subset['Ta'],Subset['ER'],color=Colors[i], label=Labels[i],linewidth = 5)
			# if i == 3:
			# 	Labels[3]+='*'
			Subset.sort_values(by = 'GPP',inplace=True)
			ax3.plot(Subset['PPFD'],Subset['GPP'],color = Colors[i],label=Labels[i],linewidth = 5)
			Subset.sort_values(by='ER',inplace=True)


		Score2 = self.Data.sort_values(by='fco2',inplace=False)
		ax4.plot(Score2['fco2'],Score2['fco2'],label='1:1',color='black',linewidth = 1)
		Score = self.Data[['fco2','NEE']].dropna()
		LR = stats.linregress(Score['fco2'],Score['NEE'])
		Line = LR[0]*Score['fco2']+LR[1]
		ax4.scatter(Score['fco2'],Score['NEE'],color = LightGreen,label=None,s=ScatterSize,edgecolor='black',linewidth = .5)
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
		# plt.sca(ax2)
		ax1.set_ylim(0,2.2)
		ax2.set_ylim(0,2.2)

		ax3.set_title('GPP vs. PPFD',fontsize=self.MajorFont)
		ax3.set_ylabel(umolCO2,fontsize=self.MinorFont)
		ax3.set_xlabel(umolPPFD,fontsize=self.MinorFont)

		ax4.set_title('NEE: Modeled vs. Observed',fontsize=self.MajorFont)
		ax4.legend(fontsize=self.MinorFont)
		ax4.set_ylabel('Modeled '+umolCO2,fontsize=self.MinorFont)
		ax4.set_xlabel('Observed '+umolCO2,fontsize=self.MinorFont)


	def NN_Style(self,ax,ax1,ax2,ax3,ax4,Var,cbobj1,cbobj2):
		ax1.set_title('Relative Model Performance',fontsize=self.MajorFont)
		ax1.set_ylabel('Normalized MSE',fontsize = self.MinorFont)
		ax1.set_xticks(np.arange(1,11))
		ax1.set_xlabel('Number of Factors',fontsize=self.MinorFont)

		ax2.set_title('C: Norm A',fontsize=self.MajorFont)
		rect = [0.93,.65,.01,.25]
		cbax1 = pt.add_subplot_axes(ax,rect)
		cb = self.fig.colorbar(cbobj1,cax=cbax1)
		cb.set_label('Normalized Difference',fontsize=self.MinorFont)

		ax3.set_title('E: Norm C',fontsize=self.MajorFont)
		rect = [0.43,.15,.01,.25]
		cbax2 = pt.add_subplot_axes(ax,rect)
		# cbax2.set_ylabel('Normalized Difference',fontsize=self.MinorFont)
		cb = self.fig.colorbar(cbobj2,cax=cbax2)
		cb.set_label('Normalized Difference',fontsize=self.MinorFont)

		ax4.set_title('Best Model Performance',fontsize=self.MajorFont)

		self.allfmt(ax1)
		self.allfmt(ax2,legend=False)
		self.allfmt(ax3,legend=False)
		self.allfmt(ax4,loc=4)

		if Var == 'co2':
			ax1.set_ylim(0.0,0.3)
			ax4.set_xlabel('Observed'+umolCO2,fontsize=self.MinorFont)
			ax4.set_ylabel('Modeled'+umolCO2,fontsize=self.MinorFont)
		else:
			ax1.set_ylim(0.0,0.84)
			ax4.set_xlabel('Observed'+nmolCH4,fontsize=self.MinorFont)
			ax4.set_ylabel('Modeled'+nmolCH4,fontsize=self.MinorFont)

	def BestML(self,ax):
		# ax1,ax2,ax3,ax4 = self.Four_Plots(ax)


		xscale = .78
		xpad = .075
		yscale = .78
		ypad = .075
		x = np.linspace(0,.66,3)
		y = np.linspace(0,.5,2)
		xs,ys = x[1]*xscale,y[1]*yscale
		x,y=x+xpad,y+ypad
		print(x,y)


		rect1 = [x[0],y[1],xs,ys]
		rect2 = [x[0],y[0],xs,ys]
		rect3 = [x[1],y[1],xs,ys]
		rect4 = [x[1],y[0],xs,ys]
		rect5 = [x[2],y[1],xs,ys]
		rect6 = [x[2],y[0],xs,ys]
		
		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)
		ax3 = pt.add_subplot_axes(ax,rect3)
		ax4 = pt.add_subplot_axes(ax,rect4)
		ax5 = pt.add_subplot_axes(ax,rect5)
		ax6 = pt.add_subplot_axes(ax,rect6)
		


		self.BMCO2 = 'Model: Wind Spd+Ta+PPFD+VWC+Active Layer'
		self.BMCH4 = 'Model: Wind Spd+air pressure+PPFD+Active Layer+Water Table'

		ax1.bar(self.Summary_CO2.index,self.Summary_CO2['MSE'],color = DarkGreen)
		ax1.errorbar(self.Summary_CO2.index,self.Summary_CO2['MSE'],self.Summary_CO2['CI'],fmt = 'o',color='black',label = '95% CI')
		ax1.annotate('Most Parsimonious Model', xy=(5, .18), xytext=(5.5, .25),
            arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='center',fontsize=self.MinorFont-2)
		y_co2 = .01
		for index,row in self.Summary_CO2.iterrows():
			if index > 1:
				ax1.text(index,y_co2,row['Model Name']+') '+lastModel+'+'+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			else:
				ax1.text(index,y_co2,row['Model Name']+') '+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			lastModel = row['Model Name']

		ax2.bar(self.Summary_CH4.index,self.Summary_CH4['MSE'],color = DarkRed)
		ax2.errorbar(self.Summary_CH4.index,self.Summary_CH4['MSE'],self.Summary_CH4['CI'],fmt = 'o',
			color='black',label = '95% CI')
		ax2.annotate("Most Parsimonious Model",
		 xy=(5, .59), xytext=(5.1, .73),arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='center',fontsize=self.MinorFont)
		y_ch4 = .01
		for index,row in self.Summary_CH4.iterrows():
			if index > 1:
				ax2.text(index,y_ch4,row['Model Name']+') '+lastModel+'+'+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			else:
				ax2.text(index,y_ch4,row['Model Name']+') '+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			lastModel = row['Model Name']

		Score = self.Filled_CO2[['fco2',self.BMCO2]].dropna()
		LR = stats.linregress(Score['fco2'].values,Score[self.BMCO2].values)
		Line = LR[0]*Score['fco2']+LR[1]
		ax3.plot(self.Filled_CO2['fco2'],self.Filled_CO2['fco2'],label='1:1',color='black',linewidth=MinorLine)
		ax3.scatter(self.Filled_CO2['fco2'],self.Filled_CO2[self.BMCO2],label=None,color=LightGreen,s=ScatterSize,edgecolor='black',linewidth = .5)
		ax3.plot(Score['fco2'],Line,color = DarkGreen,label='$r^2$: '+str(np.round(LR[2]**2,2)),linewidth=MajorLine)

		Score = self.Filled_CH4[['fch4',self.BMCH4]].dropna()
		LR = stats.linregress(Score['fch4'].values,Score[self.BMCH4].values)
		Line = LR[0]*Score['fch4']+LR[1]
		ax4.plot(self.Filled_CH4['fch4'],self.Filled_CH4['fch4'],label='1:1',color='black',linewidth=MinorLine)
		ax4.scatter(self.Filled_CH4['fch4'],self.Filled_CH4[self.BMCH4],label=None,color=LightRed,s=ScatterSize,edgecolor='black',linewidth = .5)
		ax4.plot(Score['fch4'],Line,color = DarkRed,label='$r^2$: '+str(np.round(LR[2]**2,2)),linewidth=MajorLine)


		ax1.set_ylim(0.0,0.3)
		ax3.set_xlabel('Observed '+umolCO2,fontsize=self.MinorFont)
		ax3.set_ylabel('Modeled '+umolCO2,fontsize=self.MinorFont)
		ax3.set_title('Best Model Performance: NEE',fontsize=self.MajorFont)

		ax2.set_ylim(0.0,0.84)
		ax4.set_xlabel('Observed '+nmolCH4,fontsize=self.MinorFont)
		ax4.set_ylabel('Modeled '+nmolCH4,fontsize=self.MinorFont)
		ax4.set_title('Best Model Performance: NME',fontsize=self.MajorFont)


		ax1.set_title('Relative Model Performance: NEE',fontsize=self.MajorFont)
		ax1.set_ylabel('Normalized MSE',fontsize = self.MinorFont)
		ax1.set_xticks(np.arange(1,11))
		ax1.set_xlabel('Number of Factors',fontsize=self.MinorFont)


		ax2.set_title('Relative Model Performance: NME',fontsize=self.MajorFont)
		ax2.set_ylabel('Normalized MSE',fontsize = self.MinorFont)
		ax2.set_xticks(np.arange(1,11))
		ax2.set_xlabel('Number of Factors',fontsize=self.MinorFont)

		self.allfmt(ax1)
		self.allfmt(ax2)
		self.allfmt(ax3,loc=2)
		self.allfmt(ax4,loc=2)

		self.Filled_CO2.sort_values(by='PPFD',inplace=True)
		score = self.Filled_CO2[['fco2','Model: PPFD']].dropna()
		lr = stats.linregress(score['fco2'].values,score['Model: PPFD'].values)

		ax5.scatter(self.Filled_CO2['PPFD'],self.Filled_CO2['fco2'],s=ScatterSize,color=LightGreen,edgecolor='black',linewidth=.5,label='Oservations')
		ax5.plot(self.Filled_CO2['PPFD'],self.Filled_CO2['Model: PPFD'],linewidth=5,color=DarkGreen,label = '$r^2$: '+str(np.round(lr[2]**2,2)))
		ax5.set_title('A) PPFD: NEE',fontsize=self.MajorFont)
		ax5.set_xlabel(umolPPFD,fontsize=self.MinorFont)
		ax5.set_ylabel(umolCO2,fontsize=self.MinorFont)
		ax5.legend(fontsize=self.MinorFont)
		ax5.grid()

		self.Filled_CH4.sort_values(by='PPFD',inplace=True)
		score = self.Filled_CH4[['fch4','Model: PPFD']].dropna()
		lr = stats.linregress(score['fch4'].values,score['Model: PPFD'].values)

		ax6.scatter(self.Filled_CH4['PPFD'],self.Filled_CH4['fch4'],s=ScatterSize,color=LightRed,edgecolor='black',linewidth=.5,label='Oservations')
		ax6.plot(self.Filled_CH4['PPFD'],self.Filled_CH4['Model: PPFD'],linewidth=5,color=DarkRed,label = '$r^2$: '+str(np.round(lr[2]**2,2)))
		ax6.set_title('A) PPFD: NME',fontsize=self.MajorFont)
		ax6.set_xlabel(umolPPFD,fontsize=self.MinorFont)
		ax6.set_ylabel(nmolCH4,fontsize=self.MinorFont)
		ax6.legend(fontsize=self.MinorFont)
		ax6.grid()

	def Get_Name(self,Name):
		co2 = self.Summary_CO2[self.Summary_CO2['Model Name'] ==Name]['Models'].values[0]
		ch4 = self.Summary_CH4[self.Summary_CH4['Model Name'] ==Name]['Models'].values[0]
		return(co2,ch4)

	def Factors(self,ax):
		xscale = .76
		xpad = .075
		yscale = .78
		ypad = .05
		x = np.linspace(0,.5,2)
		y = np.linspace(0,.5,2)
		xs,ys = x[1]*xscale,y[1]*yscale
		x,y=x+xpad,y+ypad
		print(x,y)


		rect1 = [x[0],y[1],xs,ys]
		rect2 = [x[1],y[1],xs,ys]
		# rect3 = [x[2],y[1],xs,ys]
		# rect4 = [x[3],y[1],xs,ys]
		rect5 = [x[0],y[0],xs,ys]
		rect6 = [x[1],y[0],xs,ys]
		# rect7 = [x[2],y[0],xs,ys]
		# rect8 = [x[3],y[0],xs,ys]

		ax1 = pt.add_subplot_axes(ax,rect1)
		ax2 = pt.add_subplot_axes(ax,rect2)
		ax3 = pt.add_subplot_axes(ax,rect5)
		ax4 = pt.add_subplot_axes(ax,rect6)
		# ax5 = pt.add_subplot_axes(ax,rect5)
		# ax6 = pt.add_subplot_axes(ax,rect6)
		# ax7 = pt.add_subplot_axes(ax,rect7)
		# ax8 = pt.add_subplot_axes(ax,rect8)

		CO2=self.Filled_CO2[np.isnan(self.Filled_CO2['fco2'])==False].copy()
		CH4=self.Filled_CH4[np.isnan(self.Filled_CH4['fch4'])==False].copy()

		Aco2,Ach4 = self.Get_Name('A')

		Bco2,Bch4 = self.Get_Name('B')
		CO2['A_B']=CO2[Bco2]-CO2[Aco2]
		CH4['A_B']=CH4[Bch4]-CH4[Ach4]


		ax1.scatter(CO2['Active Layer']*-1,CO2['A_B'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		# ax5.scatter(CH4['Wind Spd'],CH4['A_B'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		ax1.set_ylabel('Difference '+umolCO2,fontsize=self.MinorFont)
		# ax5.set_ylabel('Difference '+nmolCH4,fontsize=self.MinorFont)

		ax1.set_xlabel('Thaw Depth m',fontsize=self.MinorFont)
		# ax5.set_xlabel('Wind Speed m s${-1}$',fontsize=self.MinorFont)

		ax1.set_title('B-A: NEE',fontsize=self.MajorFont)
		# ax5.set_title('B-A: NME',fontsize=self.MinorFont)

		Cco2,Cch4 = self.Get_Name('C')
		CO2['B_C']=CO2[Cco2]-CO2[Bco2]
		CH4['B_C']=CH4[Cch4]-CH4[Bch4]
		ax2.scatter(CO2['Ta'],CO2['B_C'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		# ax6.scatter(CH4['Active Layer'],CH4['B_C'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		ax2.set_xlabel('$T_a\ ^{\circ}C$',fontsize=self.MinorFont)
		# ax6.set_xlabel('Thaw Depth m',fontsize=self.MinorFont)

		ax2.set_title('C-B: NEE',fontsize=self.MajorFont)
		# ax6.set_title('C-B: NME',fontsize=self.MinorFont)

		Dco2,Dch4 = self.Get_Name('D')
		CO2['C_D']=CO2[Dco2]-CO2[Cco2]
		CH4['C_D']=CH4[Dch4]-CH4[Cch4]
		ax3.scatter(CO2['Wind Spd'],CO2['C_D'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		# ax7.scatter(CH4['air pressure'],CH4['C_D'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		ax3.set_xlabel('Wind Speed m s$^{-1}$',fontsize=self.MinorFont)
		# ax7.set_xlabel('$P_a\ kPa$',fontsize=self.MinorFont)

		ax3.set_title('D-C: NEE',fontsize=self.MajorFont)
		ax3.set_ylabel('Difference '+umolCO2,fontsize=self.MinorFont)
		# ax7.set_title('D-C: NME',fontsize=self.MinorFont)

		Eco2,Ech4 = self.Get_Name('E')
		CO2['D_E']=CO2[Eco2]-CO2[Dco2]
		CH4['D_E']=CH4[Ech4]-CH4[Dch4]
		ax4.scatter(CO2['VWC'],CO2['D_E'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		# ax8.scatter(CH4['Water Table'],CH4['D_E'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		ax4.set_xlabel('SWC %',fontsize=self.MinorFont)
		# ax8.set_xlabel('Water Table Depth m',fontsize=self.MinorFont)

		ax4.set_title('E-D: NEE',fontsize=self.MajorFont)
		# ax8.set_title('E-D: NME',fontsize=self.MinorFont)

		# ax1.grid()
		# ax2.grid()
		# ax3.grid()
		# ax4.grid()


		score = CO2[['fco2',Aco2]].dropna()
		lra = stats.linregress(score['fco2'].values,score[Aco2].values)
		score = CO2[['fco2',Bco2]].dropna()
		lrb = stats.linregress(score['fco2'].values,score[Bco2].values)
		score = CO2[['fco2',Cco2]].dropna()
		lrc = stats.linregress(score['fco2'].values,score[Cco2].values)
		score = CO2[['fco2',Dco2]].dropna()
		lrd = stats.linregress(score['fco2'].values,score[Dco2].values)
		score = CO2[['fco2',Eco2]].dropna()
		lre = stats.linregress(score['fco2'].values,score[Eco2].values)
		print(lra[2]**2,lrb[2]**2,lrc[2]**2,lrd[2]**2,lre[2]**2)
		# ax5.grid()
		# ax6.grid()
		# ax7.grid()
		# ax8.grid()


		self.allfmt(ax1,legend=False)
		self.allfmt(ax2,legend=False)
		self.allfmt(ax3,legend=False)
		self.allfmt(ax4,legend=False)

		# Cco2,Cch4 = self.Get_Name('E')
		# CO2['C_D']=CO2[Bco2]-CO2[Aco2]
		# CH4['C_D']=CH4[Bch4]-CH4[Ach4]
		# ax5.scatter(CO2['Wind Spd'],CO2['C_D'])
		# ax7.scatter(CH4['air pressure'],CH4['C_D'])

	def Factors2(self,ax):
		xscale = .82
		xpad = .075
		yscale = .78
		ypad = .05
		x = np.linspace(0,.5,2)
		y = np.linspace(0,.5,2)
		xs,ys = x[1]*xscale,y[1]*yscale
		x,y=x+xpad,y+ypad
		print(x,y)


		rect1 = [x[0],y[1],xs,ys]
		rect2 = [x[1],y[1],xs,ys]
		# rect3 = [x[2],y[1],xs,ys]
		# rect4 = [x[3],y[1],xs,ys]
		rect5 = [x[0],y[0],xs,ys]
		rect6 = [x[1],y[0],xs,ys]
		# rect7 = [x[2],y[0],xs,ys]
		# rect8 = [x[3],y[0],xs,ys]

		# ax1 = pt.add_subplot_axes(ax,rect1)
		# ax2 = pt.add_subplot_axes(ax,rect2)
		# ax3 = pt.add_subplot_axes(ax,rect5)
		# ax4 = pt.add_subplot_axes(ax,rect6)
		ax5 = pt.add_subplot_axes(ax,rect1)
		ax6 = pt.add_subplot_axes(ax,rect2)
		ax7 = pt.add_subplot_axes(ax,rect5)
		ax8 = pt.add_subplot_axes(ax,rect6)

		CO2=self.Filled_CO2[np.isnan(self.Filled_CO2['fco2'])==False].copy()
		CH4=self.Filled_CH4[np.isnan(self.Filled_CH4['fch4'])==False].copy()

		Aco2,Ach4 = self.Get_Name('A')

		Bco2,Bch4 = self.Get_Name('B')
		CO2['A_B']=CO2[Bco2]-CO2[Aco2]
		CH4['A_B']=CH4[Bch4]-CH4[Ach4]
		# ax1.scatter(CO2['Active Layer']*-1,CO2['A_B'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		ax5.scatter(CH4['Wind Spd'],CH4['A_B'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		# ax1.set_ylabel('Difference '+umolCO2,fontsize=self.MinorFont)
		ax5.set_ylabel('Difference '+nmolCH4,fontsize=self.MinorFont)

		# ax1.set_xlabel('Thaw Depth m',fontsize=self.MinorFont)
		ax5.set_xlabel('Wind Speed m s${-1}$',fontsize=self.MinorFont)

		# ax1.set_title('B-A: NEE',fontsize=self.MinorFont)
		ax5.set_title('B-A: NME',fontsize=self.MajorFont)

		Cco2,Cch4 = self.Get_Name('C')
		CO2['B_C']=CO2[Cco2]-CO2[Bco2]
		CH4['B_C']=CH4[Cch4]-CH4[Bch4]
		# ax2.scatter(CO2['Ta'],CO2['B_C'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		ax6.scatter(CH4['Active Layer'],CH4['B_C'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		# ax2.set_xlabel('$T_a\ ^{\circ}C$',fontsize=self.MinorFont)
		ax6.set_xlabel('Thaw Depth m',fontsize=self.MinorFont)

		# ax2.set_title('C-B: NEE',fontsize=self.MinorFont)
		ax6.set_title('C-B: NME',fontsize=self.MajorFont)

		Dco2,Dch4 = self.Get_Name('D')
		CO2['C_D']=CO2[Dco2]-CO2[Cco2]
		CH4['C_D']=CH4[Dch4]-CH4[Cch4]
		# ax3.scatter(CO2['Wind Spd'],CO2['C_D'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		ax7.scatter(CH4['air pressure'],CH4['C_D'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		# ax3.set_xlabel('Wind Speed m s$^{-1}$',fontsize=self.MinorFont)
		ax7.set_xlabel('$P_a\ kPa$',fontsize=self.MinorFont)

		# ax3.set_title('D-C: NEE',fontsize=self.MinorFont)
		# ax3.set_ylabel('Difference '+umolCO2,fontsize=self.MinorFont)
		ax7.set_title('D-C: NME',fontsize=self.MajorFont)
		ax7.set_ylabel('Difference '+umolCO2,fontsize=self.MinorFont)

		Eco2,Ech4 = self.Get_Name('E')
		CO2['D_E']=CO2[Eco2]-CO2[Dco2]
		CH4['D_E']=CH4[Ech4]-CH4[Dch4]
		# ax4.scatter(CO2['VWC'],CO2['D_E'],color=LightGreen,edgecolor='black',linewidth=.5,s=ScatterSize)
		ax8.scatter(CH4['Water Table'],CH4['D_E'],color=LightRed,edgecolor='black',linewidth=.5,s=ScatterSize)

		# ax4.set_xlabel('SWC %',fontsize=self.MinorFont)
		ax8.set_xlabel('Water Table Depth m',fontsize=self.MinorFont)

		# ax4.set_title('E-D: NEE',fontsize=self.MinorFont)
		ax8.set_title('E-D: NME',fontsize=self.MajorFont)



		score = CH4[['fch4',Ach4]].dropna()
		lra = stats.linregress(score['fch4'].values,score[Ach4].values)
		score = CH4[['fch4',Bch4]].dropna()
		lrb = stats.linregress(score['fch4'].values,score[Bch4].values)
		score = CH4[['fch4',Cch4]].dropna()
		lrc = stats.linregress(score['fch4'].values,score[Cch4].values)
		score = CH4[['fch4',Dch4]].dropna()
		lrd = stats.linregress(score['fch4'].values,score[Dch4].values)
		score = CH4[['fch4',Ech4]].dropna()
		lre = stats.linregress(score['fch4'].values,score[Ech4].values)
		print(lra[2]**2,lrb[2]**2,lrc[2]**2,lrd[2]**2,lre[2]**2)


		# ax1.grid()
		# ax2.grid()
		# ax3.grid()
		# ax4.grid()

		self.allfmt(ax5,legend=False)
		self.allfmt(ax6,legend=False)
		self.allfmt(ax7,legend=False)
		self.allfmt(ax8,legend=False)

		# ax5.grid()
		# ax6.grid()
		# ax7.grid()
		# ax8.grid()



	def CO2(self,ax,fig):
		# self.fig = fig
		# ax1,ax2,ax3,ax4 = self.Four_Plots(ax)

		# Best_Model = self.Summary_CO2.loc[self.Summary_CO2['MSE']==self.Summary_CO2['MSE'].min()]
		# BM = Best_Model['Models'].values[0]
		# BM = 'Model: Wind Spd+Ta+PPFD+VWC+Active Layer'

		# ax1.bar(self.Summary_CO2.index,self.Summary_CO2['MSE'],color = DarkGreen)
		# ax1.errorbar(self.Summary_CO2.index,self.Summary_CO2['MSE'],self.Summary_CO2['CI'],fmt = 'o',
		# 	color='black',label = '95% CI')
		# ax1.annotate('Most Parsimonious Model', xy=(5, .18), xytext=(5.5, .25),
  #           arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='center',fontsize=self.MinorFont-2)

		# y = .055
		# for index,row in self.Summary_CO2.iterrows():
		# 	if index > 1:
		# 		ax1.text(index,y,row['Model Name']+') '+lastModel+'+'+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
		# 	else:
		# 		ax1.text(index,y,row['Model Name']+') '+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
		# 	lastModel = row['Model Name']


		a,b,Name='Model: PPFD+Active Layer','Model: Ta+PPFD+Active Layer','Normed_3_2'
		self.Filled_CO2[Name]=self.Normalize(a,b,self.Filled_CO2.copy(),Name)
		a,b,Name='Model: Ta+PPFD+Active Layer','Model: Wind Spd+Ta+PPFD+VWC+Active Layer','Normed_4_3'
		self.Filled_CO2[Name]=self.Normalize(a,b,self.Filled_CO2.copy(),Name)


		Subset = self.Filled_CO2[np.isnan(self.Filled_CO2['fco2'])==False]
		# Subset = Subset[Subset[Name].between(Subset[Name].quantile(.05), Subset[Name].quantile(.95), inclusive=True)]
		cbobj1=ax2.scatter(Subset['Ta']*-1,Subset['Normed_3_2'],c=Subset[Name],cmap = 'bwr',edgecolor='black',linewidth = .5,s=ScatterSize,
			vmin=-((Subset[Name]**2)**.5).max(),vmax=((Subset['Normed_3_2']**2)**.5).max())
		cbobj2=ax3.scatter(Subset['Wind Spd'],Subset['Normed_4_3'],c=Subset['Normed_4_3'],cmap = 'bwr',edgecolor='black',linewidth = .5,s=ScatterSize,
			vmin=-((Subset['Normed_4_3']**2)**.5).max(),vmax=((Subset['Normed_4_3']**2)**.5).max())

		ax2.set_xlabel('Thaw Depth m',fontsize=self.MinorFont)
		ax2.set_ylabel('$T_a \circ C$',fontsize=self.MinorFont)
		ax3.set_xlabel('Wind Speed $m s^{-1}$',fontsize=self.MinorFont)
		ax3.set_ylabel('Soil Water Content',fontsize=self.MinorFont)
		# cbobj1=ax2.scatter(self.Filled_CO2['Active Layer'],self.Filled_CO2['Ta'],c=self.Filled_CO2[Name],cmap = 'bwr',s=ScatterSize)
		# cbobj2=ax3.scatter(self.Filled_CO2['Wind Spd'],self.Filled_CO2['VWC'],c=self.Filled_CO2[Name],cmap = 'bwr',s=ScatterSize)

		# Score = self.Filled_CO2[['fco2',BM]].dropna()
		# LR = stats.linregress(Score['fco2'].values,Score[BM].values)
		# Line = LR[0]*Score['fco2']+LR[1]
		# ax4.plot(self.Filled_CO2['fco2'],self.Filled_CO2['fco2'],label='1:1',color='black',linewidth=MinorLine)
		# ax4.scatter(self.Filled_CO2['fco2'],self.Filled_CO2[BM],label=None,color=LightGreen,s=ScatterSize,edgecolor='black',linewidth = .5)
		# ax4.plot(Score['fco2'],Line,color = DarkGreen,label='$r^2$: '+str(np.round(LR[2]**2,2)),linewidth=MajorLine)

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
		
		y = .25
		for index,row in self.Summary_CH4.iterrows():
			if index > 1:
				ax1.text(index,y,row['Model Name']+') '+lastModel+'+'+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			else:
				ax1.text(index,y,row['Model Name']+') '+row['NewVar'],rotation=90,color='white',fontsize=self.MinorFont,horizontalalignment='center',verticalalignment='bottom')
			lastModel = row['Model Name']


		a,b,Name='Model: PPFD','Model: Wind Spd+PPFD','Normed_2_1'
		self.Filled_CH4[Name]=self.Normalize(a,b,self.Filled_CH4.copy(),Name)
		a,b,Name='Model: Wind Spd+air pressure+PPFD+Active Layer','Model: Wind Spd+air pressure+PPFD+Active Layer+Water Table','Normed_5_4'
		self.Filled_CH4[Name]=self.Normalize(a,b,self.Filled_CH4.copy(),Name)
		
		Subset = self.Filled_CH4[np.isnan(self.Filled_CH4['fch4'])==False]
		# Subset = Subset[Subset[Name].between(Subset[Name].quantile(.05), Subset[Name].quantile(.95), inclusive=True)]
		cbobj1=ax2.scatter(Subset['Wind Spd'],Subset['Normed_2_1'],c=Subset['Normed_2_1'],cmap = 'bwr',linewidths=.5,edgecolor='black',s=ScatterSize,
			vmin=-((Subset['Normed_2_1']**2)**.5).max(),vmax=((Subset['Normed_2_1']**2)**.5).max())
		cbobj2=ax3.scatter(Subset['Water Table'],Subset['Normed_5_4'],c=Subset['Normed_5_4'],cmap = 'bwr',linewidths=.5,edgecolor='black',s=ScatterSize,
			vmin=-((Subset['Normed_5_4']**2)**.5).max(),vmax=((Subset['Normed_5_4']**2)**.5).max())
		# cbobj1=ax2.scatter(self.Filled_CH4['Active Layer'],self.Filled_CH4['Wind Spd'],c=self.Filled_CH4[Name],cmap = 'bwr',s=ScatterSize)
		# cbobj2=ax3.scatter(self.Filled_CH4['Water Table'],self.Filled_CH4['air pressure'],c=self.Filled_CH4[Name],cmap = 'bwr',s=ScatterSize)
		ax2.set_xlabel('Thaw Depth m',fontsize=self.MinorFont)
		ax2.set_ylabel('Wind Speed $m s^{-1}$',fontsize=self.MinorFont)
		ax3.set_xlabel('Water Table Depth m',fontsize=self.MinorFont)
		ax3.set_ylabel('Air Pressure kPa',fontsize=self.MinorFont)

		Score = self.Filled_CH4[['fch4',BM]].dropna()
		LR = stats.linregress(Score['fch4'].values,Score[BM].values)
		Line = LR[0]*Score['fch4']+LR[1]
		ax4.plot(self.Filled_CH4['fch4'],self.Filled_CH4['fch4'],label='1:1',color='black',linewidth=MinorLine)
		ax4.scatter(self.Filled_CH4['fch4'],self.Filled_CH4[BM],label=None,color=LightRed,s=ScatterSize,edgecolor='black',linewidth = .5)
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

		print(co2eqMean-co2eqStd/(c)**.5 *1.96)
		print(co2eqMean+co2eqStd/(c)**.5 *1.96)

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

		# ax1.plot(self.Daily.index,self.Daily['CO2eq_mean']*0+co2Mean+co2CI,color='black',
		# 	linewidth=MajorLine,linestyle=':')
		# ax1.plot(self.Daily.index,self.Daily['CO2eq_mean']*0+co2Mean-co2CI,color='black',
		# 	linewidth=MajorLine,linestyle=':',label = None)

		ax2.bar(self.Daily.index,self.Daily['Fch4'],color = DarkRed,edgecolor='black')
		ax2.plot(self.Daily.index,self.Daily['Fch4_mean'],color='black',label='Mean :'+str(ch4Mean) +mgCH4+\
			' 95% CI $\pm$ '+str(ch4CI),linewidth=MajorLine)

		# ax2.plot(self.Daily.index,self.Daily['CO2eq_mean']*0+ch4Mean+ch4CI,color='black',
		# 	linewidth=MajorLine,linestyle=':')
		# ax2.plot(self.Daily.index,self.Daily['CO2eq_mean']*0+ch4Mean-ch4CI,color='black',
		# 	linewidth=MajorLine,linestyle=':',label = None)
		
		ax3.bar(self.Daily.index,self.Daily['CO2eq'],color = Gold,edgecolor='black')
		ax3.plot(self.Daily.index,self.Daily['CO2eq_mean'],color='black',label='Mean :'+str(co2eqMean) +gCO2eq+\
			' 95% CI $\pm$ '+str(co2eqCI),linewidth=MajorLine)

		# ax3.plot(self.Daily.index,self.Daily['CO2eq_mean']*0+co2eqMean+co2eqCI,color='black',
		# 	linewidth=MajorLine,linestyle=':')
		# ax3.plot(self.Daily.index,self.Daily['CO2eq_mean']*0+co2eqMean-co2eqCI,color='black',
		# 	linewidth=MajorLine,linestyle=':')



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
		ax3.set_title('NEE + 28*NME',fontsize=self.MajorFont,loc='left')

		ax1.set_ylabel(gCO2,fontsize=self.MinorFont)
		ax2.set_ylabel(mgCH4,fontsize=self.MinorFont)
		ax3.set_ylabel(gCO2eq,fontsize=self.MinorFont)



