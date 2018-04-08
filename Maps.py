from shapely.geometry import Point
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import show
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import box
import Plot_Tricks as PT
from Plot_Tricks import CurvedText as CT

from matplotlib.colors import LinearSegmentedColormap

GisDbase = 'C:\\Users\\wesle\\Documents\\GIS DataBase/'
Provinces = gpd.read_file(GisDbase+'Provinces.shp')
States = gpd.read_file(GisDbase+'States.shp')
Graticule = gpd.read_file(GisDbase+'ne_10m_graticules_1.shp')

NWT = gpd.read_file(GisDbase+'NTW_Yukon.shp')
Delta = gpd.read_file(GisDbase+'Delta.shp')
TreeLine = gpd.read_file(GisDbase+'NorthernAtlas\\shapefiles\\LCC_NAD83/treeline_l.shp')

FiRoot = 'C:/FishIsland_2017/'


Graticule = Graticule[Graticule['display'].isin(['80 N','70 N','60 N','50 N','40 N','150 W','140 W',
                                                '130 W','120 W','110 W','100 W','90 W'
                                                 ,'80 W','70 W','60 W','50 W','40 W'])] 

LCC = {'proj':'lcc' ,'lat_1':33 ,'lat_2':45, 
                        'lat_0':39 ,'lon_0':-96 ,'x_0':0 ,'y_0':0, 'ellps':'GRS80', 'datum':'NAD83', 'units':'m'}

Provinces=Provinces.to_crs(LCC)
Graticule=Graticule.to_crs(LCC)
States=States.to_crs(LCC)
NWT = NWT.to_crs(LCC)
Delta = Delta.to_crs(LCC)
TreeLine = TreeLine.to_crs(LCC)



def Points():
    X = [-134.881134,-133.719369]
    Y = [69.37229919,68.359180]
    Name = [' Fish\nIsland',' Inuvik']
    d = {'X':X,'Y':Y,'Site':Name}
    df = pd.DataFrame(data=d)
    geometry = [Point(xy) for xy in zip(df.X, df.Y)]

    crs = {'init': 'epsg:4326'}
    df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry) 
    df = df.to_crs(LCC)
    for i,row in df.iterrows():
#         print(df.geometry.x,df.geometry.y)
        df['X'].iloc[i] = row.geometry.x
        df['Y'].iloc[i] = row.geometry.y#[coords[0] for coords in df['coords']]
    return(df)

def Intro_Map(ax,Major_font,Minor_font,clipped = True):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    Water = (.5,.75,.95)
    Land = (.9,.85,.55)
    Green = (.2,.88,.3)
    Red = (1,.25,.1)
    Grey = (0.35,0.35,0.35)
    LightGrey = (0.5,0.5,0.5,0.5)

    X1span,Y1span = 0.5,1
    rect = [0.5,0.0,X1span,Y1span]
    ax1 = PT.add_subplot_axes(ax,rect)
    ar1 = X1span/Y1span
    ## Inset Map
    rect = [0.0,0.0,0.5,1]
    ax2 = PT.add_subplot_axes(ax,rect)
    Provinces.plot(ax=ax2,color = Land,edgecolor=Grey,linewidth=.5)
    ar,xl,yl = PT.Get_Aspect_Ratio(ax2)
    Graticule.plot(ax=ax2,color = LightGrey)
    States.plot(ax = ax2,facecolor = Land,edgecolor=Grey,linewidth=.5)
    ax2.set_facecolor(Water)
    ax2.text(-6.8e5,3.71e6,'70 N',rotation = -5,fontsize=Minor_font)
    # ax2.text(-3e5,2.43e6,'60 N',rotation = 0,fontsize=Minor_font)
    ax2.text(-1.35e6,4.535e6,'130 W',rotation = 75,fontsize=Minor_font)
    # ax2.text(-3e5,4.535e6,'90 W',rotation = 90,fontsize=Minor_font)

    Xc = -1.33e6
    Yc = 3.65e6
    Ydist = 2.9e6
    Xdist = Ydist*ar
    X1 = Xc-Xdist/2
    X2 = Xc+Xdist/2
    Y1 = Yc-Ydist/2
    Y2 = Yc+Ydist/2

    bbox = box(X1-1e3,Y1-1e3,X2+1e3,Y2+1e3)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs={'init': 'epsg:32608'})
    geo.to_crs(LCC)
    if clipped == False:
        NWTNew = gpd.overlay(NWT, geo, how='intersection')
        DeltaNew = gpd.overlay(Delta, geo, how='intersection')
        NWTNew = gpd.overlay(NWTNew,Delta,how = 'difference')
        NWTNew.to_file('ClippedNWT.shp')
    else:
        NWTNew = gpd.read_file('ClippedNWT.shp')


    ax2.set_ylim(Y1,Y2)
    ax2.set_xlim(X1,X2)

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    # Delta.plot(ax=ax1,color = Water,)

    NWTNew.plot(ax=ax1,color = Land,edgecolor=Grey,linewidth=1)
    spots = Points()
    # spots.plot(color=Red,ax=ax1,marker = '*',markersize = 400,label = 'Study Site')
    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'blue'), (1, 'green')])
    # spots.plot(column='Site',cmap=cmap,ax=ax1,marker = '*',markersize = 400,legend = True)
    ax1.scatter(spots.iloc[0]['X'],spots.iloc[0]['Y'],label=spots.iloc[0]['Site'],s=120,marker='*',color='red')
    ax1.scatter(spots.iloc[1]['X'],spots.iloc[1]['Y'],label=spots.iloc[1]['Site'],s=120,marker='*',color='blue')

    ax1.set_facecolor(Water)
    ## Get Natural aspect ratio
    ar,xl,yl = PT.Get_Aspect_Ratio(ax1)
    TreeLine.plot(ax=ax1,color = Green,linewidth=4,label='Tree\nLine')

    Xc = -1.775e6
    Yc = 3.875e6
    Ydist = 3.7e5
    Xdist = Ydist*ar1

    X1 = Xc-Xdist/2
    X2 = Xc+Xdist/2
    Y1 = Yc-Ydist/2
    Y2 = Yc+Ydist/2
    ax1.set_ylim(Y1,Y2)
    ax1.set_xlim(X1,X2)
    ax1.legend(loc=3,fontsize=Minor_font)
    
    def BBox(Y1,Y2,X1,X2,ax):
        L1y = np.linspace(Y1,Y2)
        L1x = np.linspace(X1,X1)
        ax.plot(L1x,L1y,color=Red,linewidth=2)
        L2y = np.linspace(Y2,Y2)
        L2x = np.linspace(X1,X2)
        ax.plot(L2x,L2y,color=Red,linewidth=2)
        L3y = np.linspace(Y1,Y2)
        L3x = np.linspace(X2,X2)
        ax.plot(L3x,L3y,color=Red,linewidth=2)
        L4y = np.linspace(Y1,Y1)
        L4x = np.linspace(X1,X2)
        ax.plot(L4x,L4y,color=Red,linewidth=2)
    BBox(Y1,Y2,X1,X2,ax2)
#     ax2.plot()

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

class Phenology_Change:
    def __init__(self,ax,Major_font,Minor_font,bounds,pos):
        xs,ys,x1,y1,x2,y2,x3,y3,x4,y4 = pos
        self.bounds=bounds
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs={'init': 'epsg:32608'})
        self.coords = self.getFeatures(geo)
        self.Major_font = Major_font
        self.Minor_font = Minor_font
        
        rect = [x1,y1,xs,ys]
        self.ax1 = PT.add_subplot_axes(ax,rect)
        self.ReadPlot('C:/FishIsland_2017/6_23_GeoReffed_to_8_21_Alt.tif',self.ax1,'June 23rd')
        
        rect = [x2,y2,xs,ys]
        self.ax2 = PT.add_subplot_axes(ax,rect)
        self.ReadPlot('C:/FishIsland_2017/7_10_GeoReffed_to_8_21_Alt.tif',self.ax2,'July 10th')
        
        rect = [x3,y3,xs,ys]
        self.ax3 = PT.add_subplot_axes(ax,rect)
        self.ReadPlot('C:/FishIsland_2017/8_21_Alt.tif',self.ax3,'August 21st')
        
        rect = [x4,y4,xs,ys]
        self.ax4 = PT.add_subplot_axes(ax,rect)
        self.ReadPlot('C:/FishIsland_2017/9_13_GeoReffed_to_8_21_Alt.tif',self.ax4,'September 13th')

        # self.Scale(ax3)
    
    def getFeatures(self,gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    def ReadPlot(self,Path,ax,Name=None,Scale=False):
        
        with rio.open(Path) as src:
            out_meta = src.meta.copy()
            # epsg_code = int(src.crs.data['init'][5:])
            out_img, out_transform = mask(raster=src, shapes=self.coords, crop=True)
            out_meta.update({"driver": "GTiff",
                             "height": out_img.shape[1],
                             "width": out_img.shape[2],
                             "transform": out_transform,
                             "count":4,
                             # "dtype":rio.float32,
                             "crs": src.crs})
            # print(out_meta)
            
            with rio.open(str(Name)+'.tif', "w", **out_meta) as dest:
                # print(out_img.shape)
                R = out_img[0,:,:]
                G = out_img[1,:,:]
                B = out_img[2,:,:]
                img2=(G-R)/(G+R)
                # print(img2.mean())
                dest.write(out_img)
                # ax.imshow(img2)
                # ax.colorbar()
                show(dest,ax=ax)

        ax.set_title(Name,fontsize=self.Major_font,y=1.025)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def Scale(self,ax,Units,Posx,Posy,Posx2,Posy2,size):
        ax.add_patch(Rectangle((self.bounds[0]+Posx,self.bounds[1]+Posy),Units,2.5,
                           edgecolor='black',facecolor='white',lw=2))
        ax.text(self.bounds[0]+Posx+Units+1,self.bounds[1]+Posy,
            '10 m',fontsize=self.Minor_font,rotation=0)

        bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=2)
        t = ax.text(self.bounds[0]+Posx2,self.bounds[1]+Posy2, "Nor", ha="center", va="center", rotation=90,
        size=size,color='None',
        bbox=bbox_props)
        ax.text(self.bounds[0]+Posx2,self.bounds[1]+Posy2, "N", ha="center", va="center", rotation=0,
        size=self.Minor_font)
            


    # F = 1e5
    # linex = np.linspace(0,F,F)+X1+6.5e4
    # liney = (F*F-np.linspace(-F,0,F)**2)**.5+Y1+1.0e4

    # # linex = np.linspace(5e4,1e5)+X1
    # # liney = np.linspace(0,10)**2+Y1
    # # ax1.plot(linex,liney,color ='red')
    # text = CT(
    #         x = linex,#curve[0],
    #         y = liney,#curve[1],
    #         text=" Mackenzie Delta",#text,#'this this is a very, very long text',
    #         va = 'bottom',
    #         axes = ax1, ##calls ax.add_artist in __init__
    #         fontsize=Major_font
    #     )