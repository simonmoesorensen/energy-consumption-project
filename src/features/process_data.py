#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:13:26 2022

@author: annabramslow
"""
import numpy as np
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform, nearest_points
from geopy import distance
from scipy.integrate import simpson

from road_features import count_traffic_lights, count_road_signs, oneway, get_speed_limit, count_pedestrian_crossings, count_speed_bumps, bike_facilities

# pd.set_option('mode.chained_assignment', None)

# user_path = os.path.split(os.getcwd())[0][:-3]
# filename = '13743_0_'
# sensor_filepath = os.path.join(user_path, 'data','interpolated',filename+'sensor.pickle')
# gps_filepath = os.path.join(user_path, 'data','interpolated_gps',filename+'gps.pickle')

# sensor_data = pd.read_pickle(sensor_filepath)
# gps = pd.read_pickle(gps_filepath)



def process_trip(sensor_data, gps, task_id):
    """ Assuming mapmatched sensor data """
    
    sensor_data['kW'] = sensor_data['obd.trac_cons.value'] - 160
    sensor_data['W'] = sensor_data['kW']*1000
    sensor_data['s'] = sensor_data['Time'].apply(lambda t: t.hour*60*60 + t.minute*60 + t.second + t.microsecond * 10**(-6))
    
    #Add unique id for each passed road (give same id if street is same as prior)
    # sensor_data["street_id"] = sensor_data.apply(lambda x: )
    # gps["street_id"] = gps.apply(lambda x:)
    
    df_final = pd.DataFrame()
    # run through all segments
    for street in gps.street_name.unique():
        print(f"Processing street {street} in trip {task_id}")
        
        road_features = {}
        road_features['task_id'] = task_id
        road_features['street_name'] = street
        
        #go through all road feature 
        df_street = gps.loc[gps.street_name == street]
        road_features['traffic_lights'] = count_traffic_lights(df_street)
        road_features['road_signs'] = count_road_signs(df_street)
        road_features['one_way'] = oneway(df_street)
        road_features['speed_limit'] = get_speed_limit(df_street)
        road_features['ped_walks'] = count_pedestrian_crossings(df_street)
        road_features['speed_bumps'] = count_speed_bumps(df_street)
        road_features['bike_lane'] = min(len(bike_facilities(df_street)),1)
        road_features['start_hour'] = df_street['TS_or_Distance'].min().hour
        
        
        # compute energy
        df_energy = sensor_data[sensor_data['T']=='obd.trac_cons']
        df_energy = df_energy.loc[df_energy.street_name_start == street]
        df_energy = df_energy.drop_duplicates(subset='s')
        ws = simpson(df_energy.W, df_energy.s)
        road_features['kWh'] = ws / 3600000
        
        # compute distance and duration
        df_dist = sensor_data[sensor_data['T']=='obd.odo']
        df_dist = df_dist.loc[df_dist.street_name_start == street]
        road_features['distance'] = df_dist['obd.odo.value'].max()-df_dist['obd.odo.value'].min()
        road_features['duration'] = df_dist['s'].max()-df_dist['s'].min()
        
        # compute average speed
        df_speed = sensor_data[sensor_data['T']=='obd.spd']
        df_speed = df_speed.loc[df_speed.street_name_start == street]
        road_features['avg_speed'] = df_speed['obd.spd.value'].mean()
        
        # compute window wiper activity
        df_wiper = sensor_data[sensor_data['T']=='obd.ww_f_stat']
        df_wiper = df_wiper.loc[df_wiper.street_name_start == street]
        road_features['ww_active_pct'] = df_wiper['obd.ww_f_stat.value'].sum()/len(df_wiper)
        
        # compute longitudinal acceleration
        acc = sensor_data.loc[(sensor_data['T'] == 'acc.xyz') & (sensor_data["street_name_start"] == street) ]['acc.xyz.x'].dropna()
        road_features['avg_acc'] = acc.mean()
        road_features['neg_acc_percent'] = np.sum(acc.lt(0)) / len(acc)
        road_features['pos_acc_percent'] = np.sum(acc.gt(0)) / len(acc)
        road_features['min_acc'] = acc.min()
        road_features['max_acc'] = acc.max()
        
        # Append extracted features to df
        df_final = pd.concat((df_final, pd.DataFrame(road_features, index=[0])))
        
    return df_final
