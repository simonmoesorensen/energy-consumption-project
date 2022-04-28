#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:50:37 2022

@author: nikolaibeckjensen

NOTE: For the file reference in the script to work, the script requires for the
user to be in the preprocessing folder.
"""

import numpy as np
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform, nearest_points
from geopy import distance


#test
#df = pd.read_pickle(r'/Users/annabramslow/Library/CloudStorage/OneDrive-Deltebiblioteker–DanmarksTekniskeUniversitet/LiRA Project - Dokumenter/data/interim/map_matched_data_GM_7448.pickle')
#data = df.loc[df.street_name == "Torvegade"]


## ---------------------------------------------------------------------------

def traffic_lights_helper(lat,lon,df_traffic):
    df_traffic['count'] = df_traffic.apply(lambda x: x['id'] if lat >= x['lat_min'] and lat <= x['lat_max'] and lon >= x['lon_min'] and lon <= x['lon_max'] else '', axis=1)
    return df_traffic['count'].unique()[-1]


def count_traffic_lights(data, lat_margin=0.0004, lon_margin=0.0008):
    """
    Function computing the amount of traffic lights passed for a given series of
    coordinate pairs. 
    
    Parameters
    ----------
    data : DataFrame
        DataFrame containing a 'lat' and 'long' column.
        
    lat_margin : float
        Margin in the N/S-direction for traffic light detection (0.0004 is approximately 50m)
    
    long_margin : float
        Margin in the E/S-direction for traffic light detection (0.0008 is approximately 50m)
        Returns
   
    Returns
    -------
    Number of traffic lights passed

    """
    
    # load traffic light data 
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/traffic_lights/trafiksignal_oversigt.shp'
    shp_filepath = os.path.join(user_path, folder_path)
    df_traffic = gpd.read_file(shp_filepath)
    df_traffic['lat'] = df_traffic.geometry.y
    df_traffic['lon'] = df_traffic.geometry.x
    df_traffic['lat_min'] = df_traffic.lat - lat_margin
    df_traffic['lon_min'] = df_traffic.lon - lon_margin
    df_traffic['lat_max'] = df_traffic.lat + lat_margin
    df_traffic['lon_max'] = df_traffic.lon + lon_margin
    
    #Calculate number of unique traffic lights in segment
    data['traffic_light'] = data.apply(lambda x: traffic_lights_helper(x['lat'],x['lon'],df_traffic), axis=1)
    
    unique_traffic = np.delete(data['traffic_light'].unique(), np.where(data['traffic_light'].unique()==""))
   
    return len(unique_traffic)

## ---------------------------------------------------------------------------

def road_signs_helper(lat,lon,df_signs):
    df_signs['count'] = df_signs.apply(lambda x: x['id'] if lat >= x['lat_min'] and lat <= x['lat_max'] and lon >= x['lon_min'] and lon <= x['lon_max'] else '', axis=1)
    return df_signs['count'].unique()[-1]

def count_road_signs(data, lat_margin=0.0002, lon_margin=0.0004):
    """
    Function computing the amount of road signs passed for a given series of
    coordinate pairs. 
    
    Parameters
    ----------
    data : DataFrame
        DataFrame containing a 'lat' and 'long' column.
        
    lat_margin : float
        Margin in the N/S-direction for traffic light detection (0.0002 is approximately 25m)
    
    long_margin : float
        Margin in the E/S-direction for traffic light detection (0.0004 is approximately 25m)
        Returns
   
    Returns
    -------
    Number of road signs passed

    """
    
    # load road sign data and filter away any permanently or temporarily removed signs
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/road_signs/skilte_total.shp'
    shp_filepath = os.path.join(user_path, folder_path)
    df_signs = gpd.read_file(shp_filepath)
    df_signs = df_signs[df_signs['status']=="Opstillet"]
    df_signs['lat'] = df_signs.geometry.y
    df_signs['lon'] = df_signs.geometry.x
    df_signs['lat_min'] = df_signs.lat - lat_margin
    df_signs['lon_min'] = df_signs.lon - lon_margin
    df_signs['lat_max'] = df_signs.lat + lat_margin
    df_signs['lon_max'] = df_signs.lon + lon_margin
    
    #Calculate number of unique road signs in segment
    data['road_sign'] = data.apply(lambda x: road_signs_helper(x['lat'],x['lon'],df_signs), axis=1)
    
    unique_sign = np.delete(data['road_sign'].unique(), np.where(data['road_sign'].unique()==""))
   
    return len(unique_sign)

## ---------------------------------------------------------------------------

def avg_dist(points, centroid):
    """Measures the average distance from a set of points to a given centroid
    Points: list of coordinate tuples
    Centroid: tuple with lat long coordinates
    """
    dist = [distance.geodesic(point,centroid).km for point in points]
    
    return np.mean(dist)

## ---------------------------------------------------------------------------
## TODO slet none rows
def oneway(data):
    """
    
    Parameters
    ----------
    data : DataFrame
        DataFrame containing a 'lat' and 'long' column.

    Returns
    -------
    bool
        Returns TRUE if segment is a oneway stree and FALSE if it is not.

    """
    
    # load road sign data
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/oneway_streets/ensrettede_gader.shp'
    shp_filepath = os.path.join(user_path, folder_path)
    df_oneway = gpd.read_file(shp_filepath)
    df_oneway = df_oneway.loc[df_oneway.geometry != None]
    df_oneway = df_oneway.loc[df_oneway.vejnavn != "H.C. Andersens Boulevard"]
    
    # save street name of the given df 
    assert len(data.street_name.unique()) == 1
    street_name = data.street_name.unique()[0]
    
    """# Connect coordinate set for each row in df to a route
    data['points'] = data.apply(lambda x: (x['lon_map'], x['lat_map']), axis=1)
    if len(data) == 1:
        print("Not enough coordinates")
        return False    
    gps_route = LineString([Point(point) for point in data.points])"""
    
    # Check for direct match on street segment 
    matches = df_oneway.loc[df_oneway.vejnavn == street_name]
    if len(matches) >= 1:
        return True
    else:
        return False
        
    """
    # else check if any of the linestrings intersect with current gpsroute
    else:
        #df_oneway['intersect'] = df_oneway.apply(lambda x: x['geometry'].overlaps(gps_route),axis=1)
        df_oneway['intersect'] = df_oneway.apply(lambda x: 1 if x['geometry'].intersects(gps_route) else 0,axis=1)
        df_oneway['string'] = df_oneway.apply(lambda x: x['geometry'].intersection(gps_route).wkt if x['intersect'] == 1 else '',axis=1)
        #intersections = df_oneway.loc[df_oneway.intersect == True]
        intersections = df_oneway.string.sum()
        
        if len(intersections) >= 1:
            return True
        else:
            return False"""
    
## ---------------------------------------------------------------------------

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
    
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/speed_limits/trafikhastigheder.shp'
    shp_filepath = os.path.join(user_path, folder_path)
    speed_data = gpd.read_file(shp_filepath)
    speed_data = speed_data.loc[speed_data.geometry != None]
    speed_data = speed_data.loc[speed_data.hastigheds != 1000]
    
    speed_data['centroid'] = speed_data.apply(lambda x: x['geometry'].centroid.coords[0], axis=1)
    
    # save street name of the given df 
    assert len(df.street_name.unique()) == 1
    street_name = df.street_name.unique()[0]
    
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

## ---------------------------------------------------------------------------

def pedestrian_helper_centroid(lat, lon, pedestrian, margin=10):
    pedestrian['count'] = pedestrian.apply(lambda x: x['id'] if distance.geodesic(x['centroid'], (lat, lon)).m <= margin else '', axis=1)
    return pedestrian['count'].unique()[-1]

def count_pedestrian_crossings(df):
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/pedestrian'
    path = os.path.join(user_path, folder_path)
    pedestrian = pd.read_pickle(path)
    df['crosswalk'] = df.apply(lambda x: pedestrian_helper_centroid(x['lat_map'], x['lon_map'], pedestrian), axis=1)
    return df.crosswalk.nunique() - 1

## ---------------------------------------------------------------------------

def bump_helper_centroid(lat, lon, bumps, margin=10):
    bumps['count'] = bumps.apply(lambda x: x['id'] if distance.geodesic(x['centroid'], (lat, lon)).m <= margin else '', axis=1)
    return bumps['count'].unique()[-1]

def count_speed_bumps(df):
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/speedbumps'
    path = os.path.join(user_path, folder_path)
    bumps = pd.read_pickle(path)
    df['bumps'] = df.apply(lambda x: bump_helper_centroid(x['lat_map'], x['lon_map'], bumps), axis=1)
    return df.bumps.nunique() - 1

## ---------------------------------------------------------------------------

def bike_facilities(data):
    
    #load data on bike facilities
    user_path = os.path.split(os.getcwd())[0][:-3]
    folder_path = r'data/external/bike_data/cykeldata_kk.shp'
    shp_filepath = os.path.join(user_path, folder_path)
    df_bike = gpd.read_file(shp_filepath)
    df_bike = df_bike.loc[df_bike.geometry != None]
    
    #filter away any non-existing bike lanes
    df_bike = df_bike[df_bike['status']=='Eksisterende']
    df_bike = df_bike[df_bike['kategori']!='Cykelmulighed']
    
    # Connect coordinate set for each row in df to a route
    data['points'] = data.apply(lambda x: (x['lon_map'], x['lat_map']), axis=1)
    if len(data) == 1:
        print("Not enough coordinates")
        return []    
    gps_route = LineString([Point(point) for point in data.points])
    
    #do intersection and return category of bike lane
    
    p = df_bike.apply(lambda x: nearest_points(x.geometry,gps_route),axis=1) #nearest_points between trip and all bike lanes
    
    df_bike['dist'] = p.apply(lambda x: distance.geodesic((x[0].y,x[0].x),(x[1].y,x[1].x)).m) #calculate distance
    
    bike_options = df_bike.apply(lambda x: x['kategori'] if x['dist'] <= 10 else '', axis=1)
 
    unique_bike = np.delete(bike_options.unique(), np.where(bike_options.unique()==""))
    
    return unique_bike
    

