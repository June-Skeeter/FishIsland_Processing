
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon, MultiPolygon, shape

import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.plot import show

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from datetime import datetime

from functools import partial
from multiprocessing import Pool

import ProgressBar as prb

from Klujn_2015_FootprinModel.calc_footprint_FFP_climatology_SkeeterEdits import FFP_climatology

class Calculate(object):
	"""docstring for Calculate"""
	def __init__(self,out_dir,Data,Domain,XY,params=None,Classes=None,nx=1000,dx=1,rs=[.50,.75,.90],ax=None,OtherClass = 'Other'):
		super(Calculate, self).__init__()
		self.rs=rs
		self.ax=ax
		self.Classes=Classes
		self.OtherClass = OtherClass
		self.out_dir=out_dir
		self.Runs = Data.shape[0]
		self.Data = Data
		self.raster_params=params
		self.raster_params['dx']=dx
		with rasterio.open(Domain,'r',**params) as self.Domain:
			# self.raster_params = self.Domain.profile
			# del self.raster_params['transform']    ### Transfrorms will become irelivant in rio 1.0 - gets rid of future warning
			self.Image = self.Domain.read(1)
		self.fp_params={'dx':dx,'nx':nx,'rs':rs}
		self.Prog = prb.ProgressBar(self.Runs)
		if self.Classes is not None:
			self.Intersections = self.Data[['datetime']].copy()
			for name in self.Classes['Class']:
				self.Intersections[name]=0.0
			self.Intersections[self.OtherClass] = 0.0
		self.run()

	def run(self):
		for i in range(self.Runs):
			self.i=i
			Name = str(self.Data['datetime'].iloc[i]).replace(' ','_').replace('-','').replace(':','')
			FP = FFP_climatology(zm=[self.Data['Zm'].iloc[i]],z0=[self.Data['Zo'].iloc[i]],h=[self.Data['PBLH'].iloc[i]],ol=[self.Data['L'].iloc[i]],
				sigmav=[self.Data['v_var'].iloc[i]],ustar=[self.Data['u*'].iloc[i]],wind_dir=[self.Data['wind_dir'].iloc[i]],**self.fp_params,)
			self.fpf = np.flipud(FP['fclim_2d'])*self.fp_params['dx']**2
			self.fpf /= self.fpf.sum()    ## Normalize by the domain!
			if self.Classes is not None:
				self.intersect()
			if i == 0:
				self.Sum = self.fpf
			else:
				self.Sum+= self.fpf
			if i >1:
				self.Prog.Update(i)
			with rasterio.open(self.out_dir+'30min/'+str(Name)+'.tif','w',**self.raster_params) as out:
				out.write(self.fpf,1)
				# out.write(self.fpf*0,2)
				# out.write(self.fpf*0,3)
				# out.write(self.fpf,4)
		self.Sum/=i+1
		# print(nx,dx)

		# with rasterio.open(self.out_dir+'Climatology.tif','w',**self.raster_params) as out:
		# 	out.write(self.Sum,1)


		Contours(self.out_dir,Sum = self.Sum,raster_params=self.raster_params,ax=self.ax,r=self.rs)

	def intersect(self):
		Sum = 0
		for code in self.Classes['Id']:
			# print(code)
			Template = self.Image*0.0
			Template[self.Image == code] = 1.0
			Template*= self.fpf
			Contribution = Template.sum()
			# print()
			Name = self.Classes['Class'].loc[self.Classes['Id'] == code].values[0]
			# print(self.Intersections.iloc[self.i][Name])
			# print(Contribution)
			self.Intersections.loc[self.Intersections.index==self.i,Name] = Contribution
			
			# print(self.Intersections.iloc[self.i][Name])
			Sum+=Contribution
			self.Intersections.loc[self.Intersections.index==self.i,self.OtherClass] = 1.0 - Sum
		# print(self.Intersections)

class Contours(object):
	"""docstring for ClassName"""
	def __init__(self,RasterPath,Sum=None,raster_params=None,ax=None,Jobs=None,r=[.25,.50,.70,.80,.90],PlotStyle=None):
		super(Contours, self).__init__()
		self.RasterPath=RasterPath
		print(raster_params)
		self.raster_params=raster_params
		self.r = r
		self.ax=ax
		self.PlotStyle=PlotStyle
		if Sum is not None:
			self.Sum = Sum
			self.job = 'Climatology_'+str(int(self.raster_params['dx']))+'m'
			self.Write_Contour()
		elif Jobs is not None:
			self.Jobs = Jobs
			self.Summarize()

	def Summarize(self):		
		for job in self.Jobs:
			self.job = job
			nj = 0		
			print(self.job+':')
			self.Prog = prb.ProgressBar(self.Jobs[job].shape[0])
			for date in self.Jobs[job]:
				self.Prog.Update(nj)
				Name = str(date).replace(' ','_').replace('-','').replace(':','')
				my_file = Path("/path/to/file")
				try:
					with rasterio.open(self.RasterPath+'30min/'+Name+'.tif','r') as FP:
						self.raster_params = FP.profile
						# del self.raster_params['transform']    ### Transfrorms will become irelivant in rio 1.0 - gets rid of future warning
						Image = FP.read(1)
						if nj == 0:
							self.Sum = Image
						else:
							self.Sum += Image
						nj+=1
				except:
					pass
			self.Sum/=nj
			self.Write_Contour()

	def Write_Contour(self):
		with rasterio.open(self.RasterPath+self.job+'.tiff','w',**self.raster_params) as out:
			out.write(self.Sum,1)
			transform=out.transform

		# print(self.RasterPath+self.job+'.tiff')

		# with rasterio.open('C:\\FishIsland_2017\\Footprints/Climatology.tiff','r',**self.raster_params) as Im:
		#     show(Im.read([1]),transform=Im.profile['transform'],ax=self.ax)


		Copy = self.Sum.copy()

		FlatCopy = np.sort(Copy.ravel())[::-1]
		Cumsum = np.sort(Copy.ravel())[::-1].cumsum()
		# print(self.raster_params)
		dx = self.raster_params['transform'][0]
		print(dx)
		d = {}
		d['contour'] = []
		geometry = list()
		for r in self.r:
			pct = FlatCopy[np.where(Cumsum < r)]
			Mask = self.Sum.copy()
			Mask[Mask>=pct[-1]] = 1
			Mask[Mask<pct[-1]] = np.nan
			# print(Mask.shape)
			multipart = 'No'
			# print(transform)
			for shp, val in features.shapes(Mask.astype('int16'), transform=transform):
				# print(shp)
				if val == 1:
					# print(shp)
					Poly = shape(shp)
					# print(Poly)
					Poly = Poly.buffer(dx, join_style=1).buffer(-dx, join_style=1)
					Poly = Poly.buffer(-dx, join_style=1).buffer(dx, join_style=1)
					if Poly.is_empty == False:
						if multipart == 'No':
							geometry.append(Poly)
							d['contour'].append(r)
						else:
							Multi = []
							for part in geometry[-1]:
								Multi.append(part)
							Multi.append(Poly)
							d['contour'].append(r)
							geometry[-1]=MulitPolygon(Multi)
						# multipart = 'Yes' ## Was a typo .. but it worked with and doesn't with out ... not sure why!
		df = pd.DataFrame(data=d)

		geo_df = gpd.GeoDataFrame(df,crs={'init': 'EPSG:32608'},geometry = geometry)
		geo_df['area'] =  geo_df.area 
		print(geo_df) 
		geo_df.to_file(self.RasterPath+'Contours/'+self.job+'.shp', driver = 'ESRI Shapefile')
		if self.ax is not None:
			if self.PlotStyle == 'Comparrison':
				c = 'k'# np.random.rand(3,)
				geo_df.plot(facecolor='None',edgecolor=c,ax=self.ax,linewidth=4)
				plt.plot(np.nan,np.nan,color = c,label = self.job)
			else:
				for r in self.r:
					c = 'k'# np.random.rand(3,)
					geo_df.loc[geo_df['contour'] == r].plot(facecolor='None',edgecolor=c,ax=self.ax,linewidth=4)
					plt.plot(np.nan,np.nan,color = c,label = str(r)+' %')


		else:
			geo_df.plot(facecolor='None',edgecolor=np.random.rand(3,),label = self.job)


		