#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:58:11 2022

@author: nikolaibeckjensen
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from geopy import distance
from shapely.geometry import Point, LineString
from pyproj import Geod




path = r'/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/external/road_markings etc/afmaerkning_total.shp'
markings_data = gpd.read_file(path)



# isolate and save as pickles
pedestrian = markings_data.loc[markings_data['type'] == 'Fodgængerfelt']
pedestrian = pedestrian.loc[pedestrian.geometry != None]
pedestrian['centroid'] = pedestrian.apply(lambda x: (x['geometry'].centroid.y,x['geometry'].centroid.x) , axis=1)
pedestrian.to_pickle('/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/external/pedestrian')



bumps = markings_data.loc[markings_data['type'].isin(['Prefabrikerede vejbump','Vejbump'])]
bumps = bumps.loc[bumps.geometry != None]
bumps['centroid'] = bumps.apply(lambda x: (x['geometry'].centroid.y,x['geometry'].centroid.x) , axis=1)
bumps.to_pickle('/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/external/speedbumps')
