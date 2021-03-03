from shapely.geometry import Point
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.plot import show
from matplotlib import pyplot as plt
import Plot_Tricks as pt
from Plot_Tricks import CurvedText as CT

GisDbase = 'C:\\Users\\wesle\\Documents\\GIS DataBase/'
Provinces = gpd.read_file(GisDbase+'Provinces.shp')
States = gpd.read_file(GisDbase+'States.shp')
Graticule = gpd.read_file(GisDbase+'ne_10m_graticules_1.shp')

NWT = gpd.read_file(GisDbase+'NTW_Yukon.shp')
Delta = gpd.read_file(GisDbase+'Delta.shp')


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

def Points():
    X = [-134.881134]#,-133.719369]
    Y = [69.37229919]#,68.359180]
    Name = [' Study Site']#,' Inuvik']
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

def Intro_Map(ax1,Major_font,Minor_font):

    Water = (.5,.75,.95)
    Land = (.9,.85,.55)
    Red = (1,.25,.1)
    Grey = (0.5,0.5,0.5)

    ## Inset Map
    rect = [0.55,0.0,0.45,0.45]
    ax2 = pt.add_subplot_axes(ax1,rect)
    Provinces.plot(ax=ax2,color = Land,edgecolor=Grey,linewidth=1)
    ar,xl,yl = pt.Get_Aspect_Ratio(ax2)
    Graticule.plot(ax=ax2,color = Grey)
    States.plot(ax = ax2,facecolor = Land,edgecolor=Grey,linewidth=1)
    ax2.set_facecolor(Water)
    ax2.text(0,3.7e6,'70 N',rotation = 0,fontsize=Minor_font)
    ax2.text(0,2.43e6,'60 N',rotation = 0,fontsize=Minor_font)
    ax2.text(-1.74e6,4.475e6,'130 W',rotation = 70,fontsize=Minor_font)
    ax2.text(1e5,4.3e6,'90 W',rotation = 90,fontsize=Minor_font)

    Xc = -1.0e6
    Yc = 3.0e6
    Ydist = 3.5e6
    Xdist = Ydist*ar
    
    X1 = Xc-Xdist/2
    X2 = Xc+Xdist/2
    Y1 = Yc-Ydist/2
    Y2 = Yc+Ydist/2

    ax2.set_ylim(Y1,Y2)
    ax2.set_xlim(X1,X2)

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    NWT.plot(ax=ax1,color = Land)
    Delta.plot(ax=ax1,color=Water)

    spots = Points()
    spots.plot(color=Red,ax=ax1,marker = '*',markersize = 120)
    for idx, row in spots.iterrows():
        ax1.annotate(s=row['Site'], xy=(row['X'],row['Y']),fontsize = Major_font,clip_on=True)


    ## Main Map

    ax1.set_facecolor(Water)
    ## Get Natural aspect ratio
    ar,xl,yl = pt.Get_Aspect_Ratio(ax1)

    Xc = -1.77e6
    Yc = 3.92e6
    Ydist = 1.95e5
    Xdist = Ydist*ar

    X1 = Xc-Xdist/2
    X2 = Xc+Xdist/2
    Y1 = Yc-Ydist/2
    Y2 = Yc+Ydist/2

    # ax1.text(Xc-3.5e4,Yc,'Mackenzie Delta',fontsize=Major_font,rotation=20)
    # linex = np.linspace(X1,Xc+2.5e4)+2.5e4

    # liney = (1e4-np.linspace(-1e4,0)**2)**2 +Y1

    F = 1e5
    linex = np.linspace(0,F,F)+X1+5e4
    liney = (F*F-np.linspace(-F,0,F)**2)**.5+Y1+1.0e4

    # linex = np.linspace(5e4,1e5)+X1
    # liney = np.linspace(0,10)**2+Y1
    # ax1.plot(linex,liney,color ='red')
    text = CT(
            x = linex,#curve[0],
            y = liney,#curve[1],
            text=" Mackenzie Delta",#text,#'this this is a very, very long text',
            va = 'bottom',
            axes = ax1, ##calls ax.add_artist in __init__
            fontsize=Major_font
        )


    ax1.set_ylim(Y1,Y2)
    ax1.set_xlim(X1,X2)
    
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