#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:44:46 2022

@author: nikolaibeckjensen
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from geopy import distance
from shapely.geometry import Point, LineString
from pyproj import Geod

data_path = r'/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/interim/map_matched_data_GM_7448.pickle'
df = pd.read_pickle(data_path)
df = df.loc[df.street_name=='Tagensvej']



def avg_dist(points, centroid):
    """Measures the average distance from a set of points to a given centroid
    Points: list of coordinate tuples
    Centroid: tuple with lat long coordinates
    """
    dist = [distance.geodesic(point,centroid).km for point in points]
    
    return np.mean(dist)



def get_speed_limit(df):
    """
    Parameters
    ----------
    df : DataFrame
        Dataframe containing at least 'lon_map', 'lat_map' and 'street_name' where street_name is limited to a single section
    Returns
    -------
    The speed limit for the given street_name in df 
    """
    
    
    # Load external data, 
    path = r'/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/external/speed_limits/trafikhastigheder.shp'
    speed_data = gpd.read_file(path)
    speed_data = speed_data.loc[speed_data.geometry != None]
    speed_data = speed_data.loc[speed_data.hastigheds != 1000]
    
    speed_data['centroid'] = speed_data.apply(lambda x: x['geometry'].centroid.coords[0], axis=1)
    
    # save street name of the given df 
    assert len(df.street_name.unique()) == 1
    street_name = df.street_name.unique()[0]
    print(street_name)
    
    # Connect coordinate set for each row in df to a route
    df['points'] = df.apply(lambda x: (x['lon_map'], x['lat_map']), axis=1)
    if len(df) == 1:
        print("Not enough coordinates")
        return 0    
    gps_route = LineString([Point(point) for point in df.points])

    
    
    # Check for direct match on street segment 
    matches = speed_data.loc[speed_data.vejnavn == street_name]
    if len(matches) == 1:
        return matches.hastigheds.item()
    # if multiple select closest
    elif len(matches) > 1:
        matches['dist'] = matches.apply(lambda x: avg_dist(df['points'].to_list(), x['centroid']), axis=1)
        closest_match = matches.loc[matches.dist == matches.dist.min()]
        return closest_match.hastigheds.item()
    # else check if any of the linestrings intersect with current gpsroute
    else:
        speed_data['intersect'] = speed_data.apply(lambda x: x['geometry'].intersects(gps_route),axis=1)
        intersections = speed_data.loc[speed_data.intersect == True]
        
        if len(intersections) > 1:
            result = intersections.hastigheds.mean()
        elif len(intersections) == 1:
            result = intersections.hastigheds.item()
        else:
            result = 0
        return result


speed_dict = {}
for streetname in df.street_name.unique():
    data = df.loc[df.street_name==streetname]
    speed_dict[streetname] = get_speed_limit(data)




def pedestrian_helper_centroid(lat, lon, pedestrian, margin=10):
    pedestrian['count'] = pedestrian.apply(lambda x: x['id'] if distance.geodesic(x['centroid'], (lat, lon)).m <= margin else '', axis=1)
    return pedestrian['count'].unique()[-1]

def count_pedestrian_crossings(df):
    path = r'/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/external/pedestrian'
    pedestrian = pd.read_pickle(path)
    df['crosswalk'] = df.apply(lambda x: pedestrian_helper_centroid(x['lat_map'], x['lon_map'], pedestrian), axis=1)
    return df.crosswalk.nunique() - 1

def bump_helper_centroid(lat, lon, bumps, margin=10):
    bumps['count'] = bumps.apply(lambda x: x['id'] if distance.geodesic(x['centroid'], (lat, lon)).m <= margin else '', axis=1)
    return bumps['count'].unique()[-1]

def count_speed_bumps(df):
    path = r'/Users/nikolaibeckjensen/Desktop/Danmarks Tekniske Universitet/LiRA Project - Dokumenter/data/external/speedbumps'
    bumps = pd.read_pickle(path)
    df['bumps'] = df.apply(lambda x: bump_helper_centroid(x['lat_map'], x['lon_map'], bumps), axis=1)
    return df.bumps.nunique() - 1






    
    # ---- Approach 2: Search closest ------ #