#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:00:27 2022

@author: annabramslow

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
from scipy.integrate import simpson

from road_features import count_traffic_lights, count_road_signs, oneway, get_speed_limit, count_pedestrian_crossings, \
    count_speed_bumps, bike_facilities
from process_data import process_trip


def load_data(task_id=[]):
    # create user specific file paths
    user_path = os.path.split(os.getcwd())[0][:-3]
    sensor_filepath = os.path.join(user_path, 'data', 'interpolated')
    gps_filepath = os.path.join(user_path, 'data', 'interpolated_gps')

    # load filenames, tripsids and passes of sensor data
    sensor_folder_content = os.listdir(sensor_filepath)
    df_sensor_data = pd.DataFrame({"Filename": sensor_folder_content})
    df_sensor_data[['task_id', 'pass', 'sensor/gps']] = df_sensor_data.apply(lambda x: x['Filename'].split("_"), axis=1,
                                                                             result_type="expand")
    df_sensor_data = df_sensor_data.sort_values(['task_id', 'pass'])

    # load filenames, tripsids and passes of gps data
    gps_folder_content = os.listdir(gps_filepath)
    df_gps_data = pd.DataFrame({"Filename": gps_folder_content})
    df_gps_data[['task_id', 'pass', 'sensor/gps']] = df_gps_data.apply(lambda x: x['Filename'].split("_"), axis=1,
                                                                       result_type="expand")
    df_gps_data = df_gps_data.sort_values(['task_id', 'pass'])

    # if no specific task ids are given, then run through all in folder
    if len(task_id) == 0:
        task_id = df_sensor_data.task_id.unique()

    # if len(task_id) > 0:
    # run through specified task ids and proces data + print to csv

    for task in task_id:

        temp_sensor = df_sensor_data.loc[df_sensor_data["task_id"] == str(task)]
        temp_gps = df_gps_data.loc[df_gps_data["task_id"] == str(task)]

        for i in range(len(temp_sensor)):
            sensor_filepath = os.path.join(user_path, 'data', 'interpolated', temp_sensor["Filename"].iloc[i])
            gps_filepath = os.path.join(user_path, 'data', 'interpolated_gps', temp_gps["Filename"].iloc[i])

            sensor_data = pd.read_pickle(sensor_filepath)
            gps = pd.read_pickle(gps_filepath)
            data = process_trip(sensor_data, gps, task)

            print_path = os.path.join(user_path, 'data', 'processed', f'Trip_{task}_{i}.csv')
            data.to_csv(print_path)

            # else:
    # #load all data in folder and proces it + print to csv
    #     for i in len(df_sensor_data):

    #         sensor_filepath = os.path.join(user_path, 'data','interpolated',df_sensor_data["Filename"].iloc[i])
    #         gps_filepath = os.path.join(user_path, 'data','interpolated_gps',df_gps_data["Filename"].iloc[i])

    #         sensor_data = pd.read_pickle(sensor_filepath)
    #         gps = pd.read_pickle(gps_filepath)
    #         data = process_trip(sensor_data, gps, task) 

    #         print_path = os.path.join(user_path, 'data','processed',f'Trip_{task}_{i}.csv')
    #         data.to_csv(print_path)
