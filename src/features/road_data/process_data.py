#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:13:26 2022

@author: annabramslow
"""
import numpy as np
import pandas as pd
import os
from scipy.integrate import simpson
from tqdm import tqdm

from road_features import count_traffic_lights, count_road_signs, oneway, get_speed_limit, count_pedestrian_crossings, count_speed_bumps, \
    bike_facilities, traffic_lights_helper

# pd.set_option('mode.chained_assignment', None)

# user_path = os.path.split(os.getcwd())[0][:-3]
# filename = '13743_0_'
# sensor_filepath = os.path.join(user_path, 'data','interpolated',filename+'sensor.pickle')
# gps_filepath = os.path.join(user_path, 'data','interpolated_gps',filename+'gps.pickle')

# sensor_data = pd.read_pickle(sensor_filepath)
# gps = pd.read_pickle(gps_filepath)
import warnings

warnings.filterwarnings('ignore')


def process_trip(sensor_data, road_data, gps, task_id):
    """ Assuming mapmatched sensor data """

    sensor_data['kW'] = sensor_data['obd.trac_cons.value'] - 160
    sensor_data['W'] = sensor_data['kW'] * 1000
    sensor_data['s'] = sensor_data['Time'].apply(lambda t: t.hour * 60 * 60 + t.minute * 60 + t.second + t.microsecond * 10 ** (-6))

    # Add unique id for each passed road (give same id if street is same as prior)
    # sensor_data["street_id"] = sensor_data.apply(lambda x: )
    # gps["street_id"] = gps.apply(lambda x:)

    # df_final = pd.DataFrame()

    # run through all segments
    for street in tqdm(gps.street_name.unique()):
        # go through all road feature
        df_street = gps.loc[gps.street_name == street]
        start_street = sensor_data.street_name_start == street

        # def get_street_features(row):
        #     return (traffic_lights_helper(row['lat'], row['lon'], df_traffic) +
        #             )

        # df_traffic = prepare_traffic_lights(road_data)

        # Calculate number of unique traffic lights in segment
        # df_traffic[street_columns] = df_street.apply(get_street_features, axis=1, result_type='expand')

        # unique_traffic = np.delete(data['traffic_light'].unique(), np.where(data['traffic_light'].unique() == ""))

        sensor_data.loc[start_street, 'traffic_lights'] = count_traffic_lights(df_street, road_data)
        sensor_data.loc[start_street, 'road_signs'] = count_road_signs(df_street, road_data)
        sensor_data.loc[start_street, 'one_way'] = oneway(df_street, road_data)
        sensor_data.loc[start_street, 'speed_limit'] = get_speed_limit(df_street, road_data)
        sensor_data.loc[start_street, 'ped_walks'] = count_pedestrian_crossings(df_street, road_data)
        sensor_data.loc[start_street, 'speed_bumps'] = count_speed_bumps(df_street, road_data)
        sensor_data.loc[start_street, 'bike_lane'] = min(len(bike_facilities(df_street, road_data)), 1)
        sensor_data.loc[start_street, 'start_hour'] = df_street['TS_or_Distance'].min().hour

        # # compute energy
        # df_energy = sensor_data[sensor_data['T']=='obd.trac_cons']
        # df_energy = df_energy.loc[df_energy.street_name_start == street]
        # df_energy = df_energy.drop_duplicates(subset='s')
        # ws = simpson(df_energy.W, df_energy.s)
        # df_street['kWh'] = ws / 3600000
        #
        # # compute distance and duration
        # df_dist = sensor_data[sensor_data['T']=='obd.odo']
        # df_dist = df_dist.loc[df_dist.street_name_start == street]
        # df_street['distance'] = df_dist['obd.odo.value'].max()-df_dist['obd.odo.value'].min()
        # df_street['duration'] = df_dist['s'].max()-df_dist['s'].min()
        #
        # # compute average speed
        # df_speed = sensor_data[sensor_data['T']=='obd.spd']
        # df_speed = df_speed.loc[df_speed.street_name_start == street]
        # df_street['avg_speed'] = df_speed['obd.spd.value'].mean()
        #
        # # compute window wiper activity
        # df_wiper = sensor_data[sensor_data['T']=='obd.ww_f_stat']
        # df_wiper = df_wiper.loc[df_wiper.street_name_start == street]
        # df_street['ww_active_pct'] = df_wiper['obd.ww_f_stat.value'].sum()/len(df_wiper)
        #
        # # compute longitudinal acceleration
        # acc = sensor_data.loc[(sensor_data['T'] == 'acc.xyz') & (sensor_data["street_name_start"] == street) ]['acc.xyz.x'].dropna()
        # df_street['avg_acc'] = acc.mean()
        # df_street['neg_acc_percent'] = np.sum(acc.lt(0)) / len(acc)
        # df_street['pos_acc_percent'] = np.sum(acc.gt(0)) / len(acc)
        # df_street['min_acc'] = acc.min()
        # df_street['max_acc'] = acc.max()

    return sensor_data
