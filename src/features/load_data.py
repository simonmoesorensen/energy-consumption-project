#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:00:27 2022

@author: annabramslow

NOTE: For the file reference in the script to work, the script requires for the
user to be in the preprocessing folder.
"""
import argparse

import numpy as np
import pandas as pd
import os
import geopandas as gpd
from tqdm import tqdm

from road_features import count_traffic_lights, count_road_signs, oneway, get_speed_limit, count_pedestrian_crossings, \
    count_speed_bumps, bike_facilities
from process_data import process_trip


def load_data(data_path, trip, task_id=None):
    print(f'Processing road data for {trip}')
    # create user specific file paths
    if task_id is None:
        task_id = []

    out_dir = os.path.join(data_path, 'processed', trip)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sensor_dir_path = os.path.join(data_path, 'interim', trip, 'interpolated', 'sensor')
    gps_dir_path = os.path.join(data_path, 'interim', trip, 'interpolated', 'gps')

    # load filenames, tripsids and passes of sensor data
    sensor_folder_content = os.listdir(sensor_dir_path)
    df_sensor_data = pd.DataFrame({"Filename": sensor_folder_content})
    df_sensor_data['task_id'] = df_sensor_data.apply(lambda x: x['Filename'].split("_")[2], axis=1)
    df_sensor_data = df_sensor_data.sort_values(['task_id'])

    # load filenames, tripsids and passes of gps data
    gps_folder_content = os.listdir(gps_dir_path)
    df_gps_data = pd.DataFrame({"Filename": gps_folder_content})
    df_gps_data['task_id'] = df_gps_data.apply(lambda x: x['Filename'].split("_")[2], axis=1)
    df_gps_data = df_gps_data.sort_values(['task_id'])

    # if no specific task ids are given, then run through all in folder
    if len(task_id) == 0:
        task_id = df_sensor_data.task_id.unique()

    # Load road data
    road_data = {
        'traffic': gpd.read_file('data/external/traffic_lights/trafiksignal_oversigt.shp'),
        'signs': gpd.read_file('data/external/road_signs/skilte_total.shp'),
        'oneway': gpd.read_file('data/external/oneway_streets/ensrettede_gader.shp'),
        'speed_limit': gpd.read_file('data/external/speed_limits/trafikhastigheder.shp'),
        'pedestrian': pd.read_pickle('data/external/pedestrian'),
        'speed_bumps': pd.read_pickle('data/external/speedbumps'),
        'bikes': gpd.read_file('data/external/bike_data/cykeldata_kk.shp'),
    }

    # run through specified task ids and proces data + print to pickle
    for task in task_id:
        print(f'Processing {task=} in {trip=}')
        temp_sensor = df_sensor_data.loc[df_sensor_data["task_id"] == str(task)]
        temp_gps = df_gps_data.loc[df_gps_data["task_id"] == str(task)]

        for i in range(len(temp_sensor)):
            sensor_filepath = os.path.join(sensor_dir_path, temp_sensor["Filename"].iloc[i])
            gps_filepath = os.path.join(gps_dir_path, temp_gps["Filename"].iloc[i])

            sensor_data = pd.read_pickle(sensor_filepath)
            gps = pd.read_pickle(gps_filepath)
            data = process_trip(sensor_data, road_data, gps, task)
            data.to_pickle(os.path.join(out_dir, f'trip_{task}_{i}.pickle'))


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description="Please provide command line arguments.")

    parser.add_argument("--route", type=str, help="Process this trip.")
    args = parser.parse_args()

    # Assuming run directory in root
    load_data('./data', trip=args.route)
    # load_data('./data', trip='CPH1_VH')
    # load_data('./data', trip='CPH6_HH')
    # load_data('./data', trip='CPH6_VH')
    # load_data('./data', trip='M3_HH')
    # load_data('./data', trip='M3_VH')
