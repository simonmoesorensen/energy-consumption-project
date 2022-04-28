"""
@author: Milena Bajic (DTU Compute)
e-mail: lenka.bajic@gmail.com
"""

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from math import radians, cos, sin, asin, sqrt, ceil, floor
import matplotlib.pyplot as plt
from utils.plotting import *
import osrm
import requests
import json
import os, sys


def haversine_distance(lat1, lat2, lon1, lon2, in_meters=True):
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    # calculate the result
    res = c * r
    if in_meters:
        return 1000 * res
    else:
        return res


def compute_kpi_aran(data):
    kpi = (data['DI'] + ((data['p79.RutDepthLeft'] + data['p79.RutDepthRight']) / 2) ** 0.5) * (
                (data['p79.IRI5'] + data['p79.IRI21']) / 2) ** 0.2

    return kpi


def latlon_to_cartesian(lats, lons):
    R = 6371000  # radium of the Earth in meters
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    x = R * np.cos(lats) * np.cos(lons)
    y = R * np.cos(lats) * np.sin(lons)
    z = R * np.sin(lats)
    return x, y, z


def get_segment(i_start, i_end, df, var):
    x = df[int(i_start): int(i_end) + 1][var].values
    x = x[~np.isnan(x)]
    return x


def clean(df):
    for var in ['acc.xyz.x', 'acc.xyz.y', 'acc.xyz.z', 'obd.spd_veh']:
        mask = (df[var].apply(lambda row: len(row)) > 0)
        df = df[mask]
        df.reset_index(inplace=True, drop=True)
    return df


def do_kNN_matching_irisegments_only(GM_data, iri, GM_task_id, DRD_id, max_distance=1, gps_points_to_plot=-1,
                                     out_dir='results'):
    out_filename = '{0}/matched_trip{1}_{2}_filter10_maxdistance{3}.pkl'.format(out_dir, GM_task_id, 'GM', max_distance)
    try:
        matched_data = pd.read_pickle(laser_out_filename)
        dist_idx = None
    except:
        # Convert latlon to eucleadian
        DRD_x, DRD_y, DRD_z = latlon_to_cartesian(iri.lat.values, iri.lon.values)
        GM_x, GM_y, GM_z = latlon_to_cartesian(GM_data.lat.values, GM_data.lon.values)

        # Match drd iri to the closest gm point
        print('Matching DRD to GM')
        latlon = np.vstack([GM_x, GM_y, GM_z]).T
        kd_tree = KDTree(latlon)
        findNearestRegion = kd_tree.query

        # Find the closest neigbour (the one with the minimum eucledian distance)
        dist_idx = [findNearestRegion(coords) for coords in zip(DRD_x, DRD_y, DRD_z)]  # dist_idx: (distance, DRD_index)

        # Set a dataframe to store the matches
        matched_data = pd.DataFrame()
        for col in GM_data.columns:
            new_col = col if col.startswith('GM') else 'GM_' + col
            matched_data[new_col] = pd.Series(dtype=GM_data.dtypes[col])
        for col in iri.columns:
            matched_data[col] = pd.Series(dtype=iri.dtypes[col])
        matched_data['Distance'] = pd.Series(dtype=np.float32)

        print('Filtering matches')
        index = 0
        # Filter on maximum distance and save the matching results
        for DRD_index, (distance, GM_index) in enumerate(dist_idx):
            # print(DRD_index, GM_index, distance)

            # Save only matched with distance smaller than this
            if distance > max_distance:
                continue

            # GM match
            GM_match = GM_data.iloc[GM_index]

            # DRD match
            DRD_match = iri.iloc[DRD_index]

            # Append the matched data
            for col_name in GM_data.columns:
                new_col_name = col_name if col_name.startswith('GM') else 'GM_' + col_name
                matched_data.at[index, new_col_name] = GM_match[col_name]

            for col_name in iri.columns:
                matched_data.at[index, col_name] = DRD_match[col_name]

            matched_data.at[index, 'Distance'] = distance

            index = index + 1
        # matched_data.drop(['DRD_DRDMeasurementId','DRD_message', 'Distance','DRD_T'],
        #                  axis=1,inplace=True)
        # matched_data = matched_data.reindex(columns=['GM_TS_or_Distance','GM_Date', 'GM_Time','GM_lat', 'GM_lon','GM_Acceleration_x', 'GM_Acceleration_y', 'GM_Acceleration_z','GM_Acceleration_full','DRD_TS_or_Distance', 'DRD_lat', 'DRD_lon','DRD_IRI5', 'DRD_IRI21'])

    # Min and max distance in matched data
    segments = []  # all segments
    min_d = int(matched_data.TS_or_Distance.min())
    max_d = int(matched_data.TS_or_Distance.max())
    for d in range(min_d, max_d, 10):
        print('Segment: {0}-{1}'.format(d, d + 10))
        seg0 = matched_data[matched_data.TS_or_Distance == d]
        seg1 = matched_data[matched_data.TS_or_Distance == d + 10]
        if seg0.empty or seg1.empty:
            continue

        # seg0.drop(['DRD_IRI5', 'DRD_IRI21','GM_Acceleration_x','GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full'],axis=1,inplace=True)
        seg0.reset_index(inplace=True)
        # seg1.drop(['GM_Acceleration_x','GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full'],axis=1,inplace=True)
        seg1.reset_index(inplace=True)

        # Column names
        for col_name in seg0.columns:
            seg0.rename(columns={col_name: col_name + '_start'}, inplace=True)
        for col_name in seg1.columns:
            if 'IRI' in col_name:
                continue
            seg1.rename(columns={col_name: col_name + '_end'}, inplace=True)

        for col_name in seg1.columns:
            seg0[col_name] = seg1[col_name]
        seg0.drop(['index_start', 'index_end'], axis=1, inplace=True)

        segments.append(seg0)

    # Prepare
    matched_iri_segments = pd.concat(segments)
    matched_iri_segments.reset_index(inplace=True, drop=True)

    # Compute
    matched_iri_segments['GM_TS_or_Distance_start'] = matched_iri_segments.apply(
        lambda row: datetime.datetime.combine(row.GM_Date_start, row.GM_Time_start), axis=1)
    matched_iri_segments['GM_TS_or_Distance_end'] = matched_iri_segments.apply(
        lambda row: datetime.datetime.combine(row.GM_Date_end, row.GM_Time_end), axis=1)

    # Save
    # matched_iri_segments.to_pickle('{0}/GMtrip-{1}_DRD-{2}_maxdistance-{3}_irisegments_startend.pkl'.format(out_dir, GM_task_id, DRD_id, max_distance))
    # matched_iri_segments.to_excel('{0}/GMtrip-{1}_DRD-{2}__maxdistance-{3}_irisegments_startend.xlsx'.format(out_dir, GM_task_id, DRD_id, max_distance))
    return matched_iri_segments


def do_kNN_matching(GM_data, DRD_data, iri, GM_task_id, var_name, max_distance=5, gps_points_to_plot=-1,
                    out_dir='results'):
    laser_out_filename = '{0}/matched_trip{1}_{2}_filter10_maxdistance{3}_lasers.pkl'.format(out_dir, GM_task_id,
                                                                                             var_name, max_distance)

    try:
        matched_data = pd.read_pickle(laser_out_filename)
        dist_idx = None
    except:
        if 'Date' not in GM_data.columns:
            GM_data['Date'] = GM_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).date())
            GM_data['Time'] = GM_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).time())

        # Convert latlon to eucleadian
        DRD_x, DRD_y, DRD_z = latlon_to_cartesian(DRD_data.DRD_lat.values, DRD_data.DRD_lon.values)
        GM_x, GM_y, GM_z = latlon_to_cartesian(GM_data.lat.values, GM_data.lon.values)

        # Do matching
        print('Matching GM to DRD')
        latlon = np.vstack([DRD_x, DRD_y, DRD_z]).T
        kd_tree = KDTree(latlon)
        findNearestRegion = kd_tree.query

        # Find the closest neigbour (the one with the minimum eucledian distance)
        dist_idx = [findNearestRegion(coords) for coords in zip(GM_x, GM_y, GM_z)]  # dist_idx: (distance, DRD_index)

        # Set a dataframe to store the matches
        matched_data = pd.DataFrame()
        for col in GM_data.columns:
            matched_data[col] = pd.Series(dtype=GM_data.dtypes[col])
        for col in DRD_data.columns:
            matched_data[col] = pd.Series(dtype=DRD_data.dtypes[col])
        matched_data['Distance'] = pd.Series(dtype=np.float32)
        index = 0

        print('Filtering matches')
        # Filter on maximum distance and save the matching results
        for GM_index, (distance, DRD_index) in enumerate(dist_idx):
            # print(DRD_index, GM_index, distance)

            # Save only matched with distance smaller than this
            if distance > max_distance:
                continue

            # GM match
            GM_match = GM_data.iloc[GM_index]

            # DRD match
            DRD_match = DRD_data.iloc[DRD_index]

            # Append the matched data
            for col_name in GM_data.columns:
                matched_data.at[index, col_name] = GM_match[col_name]

            for col_name in DRD_data.columns:
                matched_data.at[index, col_name] = DRD_match[col_name]

            matched_data.at[index, 'Distance'] = distance

            index = index + 1

        # Drop what is not needed
        matched_data.dropna(inplace=True)
        matched_data.drop(
            ['GPS_TS_or_Distance_start', 'GPS_lat_start', 'GPS_lon_start', 'GPS_dt', 'GPS_dlat', 'GPS_dlon', 'dtx'],
            axis=1, inplace=True)
        matched_data.sort_values(by='DRD_TS_or_Distance', inplace=True)
        matched_data.reset_index(inplace=True, drop=True)

        # Save match to lasers
        matched_data.to_pickle(laser_out_filename)

    # Filter iri to make it faster
    mind = matched_data.DRD_TS_or_Distance.min()
    maxd = matched_data.DRD_TS_or_Distance.max()
    iri = iri[(iri.DRD_TS_or_Distance > mind) & (iri.DRD_TS_or_Distance < maxd + 10)]
    matched_data.reset_index(inplace=True, drop=True)

    # Find iri
    iri_matched = df_to_iri_seq(matched_data, iri, var_name, out_dir)
    iri_matched.reset_index(inplace=True, drop=True)
    iri_matched['DRD_IRI_mean'] = iri_matched.apply(lambda row: (row.DRD_IRI5 + row.DRD_IRI21) / 2, axis=1)
    iri_matched.to_pickle(
        '{0}/matched_trip{1}_{2}_filter10_maxdistance{3}_iri.pkl'.format(out_dir, GM_task_id, var_name, max_distance))

    # Plot
    # plot_geolocation_2((matched_data.lon.values[0:gps_points_to_plot:50], matched_data.lat.values[0:gps_points_to_plot:50]),
    #                   (matched_data.DRD_lon.values[0:gps_points_to_plot:50], matched_data.DRD_lat.values[0:gps_points_to_plot:50]), name='_matched_trip{0}_{1}'.format(GM_task_id, var_name))

    return matched_data, dist_idx, iri_matched


def do_kNN_matching_iri(GM_data, iri, GM_task_id, DRD_id, gps_points_to_plot=-1, out_dir='results'):
    out_filename = '{0}/matched_trip{1}_{2}.pkl'.format(out_dir, GM_task_id, DRD_id)

    # Convert latlon to eucleadian: DRD
    iri_suffix = '_map_start'
    DRD_start_x, DRD_start_y, DRD_start_z = latlon_to_cartesian(iri['lat' + iri_suffix].values,
                                                                iri['lat' + iri_suffix].values)
    iri_suffix = '_map_end'
    DRD_end_x, DRD_end_y, DRD_end_z = latlon_to_cartesian(iri['lat' + iri_suffix].values,
                                                          iri['lat' + iri_suffix].values)

    # Convert latlon to eucleadian: GM
    GM_x, GM_y, GM_z = latlon_to_cartesian(GM_data['lat'].values, GM_data['lat'].values)

    # Match drd iri to the closest gm point
    print('Matching DRD to GM')
    latlon = np.vstack([GM_x, GM_y, GM_z]).T
    findNearestRegion = KDTree(latlon).query

    # Find the closest neigbour (the one with the minimum eucledian distance)
    dist_idx_start = [findNearestRegion(coords) for coords in
                      zip(DRD_start_x, DRD_start_y, DRD_start_z)]  # dist_idx: (distance, DRD_index)
    dist_idx_end = [findNearestRegion(coords) for coords in
                    zip(DRD_end_x, DRD_end_y, DRD_end_z)]  # dist_idx: (distance, DRD_index)

    # Remove unmatched 
    dist_idx_start = [(x, y) for (x, y) in dist_idx_start if (x != np.inf and y != np.inf)]
    dist_idx_end = [(x, y) for (x, y) in dist_idx_end if (x != np.inf and y != np.inf)]

    # Set a dataframe to store the matches
    iri['GM_lat_map_start'] = pd.Series(dtype=np.float32)  # match to the GPS start map
    iri['GM_lon_map_start'] = pd.Series(dtype=np.float32)  # match to the GPS start map
    iri['DRD_index_start'] = pd.Series(dtype=np.int64)  # match to the GPS start map
    iri['GM_index_start'] = pd.Series(dtype=np.int64)  # match to the GPS start map
    iri['Distance_start'] = pd.Series(dtype=np.float32)

    iri['GM_lat_map_end'] = pd.Series(dtype=np.float32)  # match to the GPS end
    iri['GM_lon_map_end'] = pd.Series(dtype=np.float32)  # match to the GPS end
    iri['DRD_index_end'] = pd.Series(dtype=np.int64)  # match to the GPS start map
    iri['GM_index_end'] = pd.Series(dtype=np.int64)  # match to the GPS start map
    iri['Distance_end'] = pd.Series(dtype=np.float32)

    for DRD_index, (distance, GM_index) in enumerate(dist_idx_start):
        # GM match
        GM_match = GM_data.iloc[GM_index]

        # DRD match
        DRD_match = iri.iloc[DRD_index]

        iri.at[DRD_index, 'GM_lat_map_start'] = GM_match.lat
        iri.at[DRD_index, 'GM_lon_map_start'] = GM_match.lon
        iri.at[DRD_index, 'DRD_index_start'] = DRD_index
        iri.at[DRD_index, 'GM_index_start'] = GM_index
        iri.at[DRD_index, 'Distance_start'] = distance

    for DRD_index, (distance, GM_index) in enumerate(dist_idx_end):
        # GM match
        GM_match = GM_data.iloc[GM_index]

        # DRD match
        DRD_match = iri.iloc[DRD_index]

        iri.at[DRD_index, 'GM_lat_map_end'] = GM_match.lat
        iri.at[DRD_index, 'GM_lon_map_end'] = GM_match.lon
        iri.at[DRD_index, 'DRD_index_end'] = DRD_index
        iri.at[DRD_index, 'GM_index_end'] = GM_index
        iri.at[DRD_index, 'Distance_end'] = distance

    return iri


def map_match(gps, host, lat_name='lat', lon_name='lon', r=50):
    # Map match a chunk of gps data
    lat = gps[lat_name]
    lon = gps[lon_name]
    c = list(zip(lon, lat))
    radiuses = [r] * len(lat)
    # print(lat)
    # print(lon)

    timeout = 100

    try:
        client = osrm.Client(host=host, profile='car', timeout=timeout)
        match_response = client.match(coordinates=c, radiuses=radiuses)

        t = match_response['tracepoints']
        wp = [i['location'] if i != None else (np.NaN, np.NaN) for i in match_response['tracepoints']]
        gps['lon_map'] = list(zip(*wp))[0]
        gps['lat_map'] = list(zip(*wp))[1]

        gps['way_id'] = [i['waypoint_index'] if i != None else (np.NaN, np.NaN) for i in match_response['tracepoints']]
        gps['street_name'] = [i['name'] if i != None else (np.NaN, np.NaN) for i in match_response['tracepoints']]
        # gps['hint'] = [i['hint'] if i!=None else (np.NaN,np.NaN) for i in match_response['tracepoints'] ]
        return gps
    except Exception as e:
        print('Error in mapmatching')
        print(e)
        return None


def map_match_gps_data(gps_data, host, is_GM=False, lat_name='lat', lon_name='lon', out_dir='.', out_file_suff='',
                       preload=False):
    out_filename = '{0}/{1}.pickle'.format(out_dir, out_file_suff)

    if os.path.exists(out_filename) and preload:
        print('Loading map matched data from file')
        gps_result = pd.read_pickle(out_filename)
        return gps_result

    else:
        # gps_data = data[[lat_name, lon_name]]

        # Create GPS chunks
        n = ceil(gps_data.shape[0] / 100)
        gps_chunks = np.array_split(gps_data, n)

        gps_chunks_mm = []
        # Map Match GPS chunks
        for i, gps_chunk in enumerate(gps_chunks):
            # if i%100==0:
            print(' === Chunk: {0}/{1} ===='.format(i, n))
            res = map_match(gps_chunk, host, lat_name, lon_name)
            if res is not None:
                gps_chunks_mm.append(res)

        if len(gps_chunks_mm) == 0:
            print('No chunks found')
            return None

        gps_result = pd.concat(gps_chunks_mm)

        # Clean non-matched points
        rem_ind = gps_result[gps_result['lat_map'].isna()].index
        gps_result.drop(index=rem_ind, inplace=True)
        rem_ind = gps_result[gps_result['lon_map'].isna()].index
        gps_result.drop(index=rem_ind, inplace=True)

        # Extract Message column if GM
        if is_GM:
            pass
            # Set Message to nan if from GPS 
            # data['Message'].mask( out_df['T']=='track.pos',inplace=True)

            # Extract Message for all sensors
            # print('- Extracting message')
            # data = pd.concat([data.drop(['Message'], axis=1),  data['Message'].apply(pd.Series)], axis=1)

            # Merge with initial dataset
            # gps_result = pd.concat([data, gps_result[['lat_map', 'lon_map']]],axis=1)

        # Save results
        gps_result.to_pickle(out_filename)
        # out_df.to_csv(out_filename.replace('.pickle', '.csv'))
        print('Saved map matching output to: {0}'.format(out_filename))

        return gps_result


def route_between_2GPS_points(gps):
    # Return nodes between gps points. Input gps is a dataframe.

    # Map matched GPS coordinates
    lat = gps.lat_map
    lon = gps.lon_map
    c = list(zip(lon, lat))

    # Call the Route service
    timeout = 20
    client = osrm.Client(host='http://liradbdev.compute.dtu.dk:5000', profile='car', timeout=timeout)
    match_response = client.route(coordinates=c, overview=osrm.overview.full, annotations=True)

    # Distance between the gps points
    distance_between_2GPS = match_response['routes'][0]['distance']

    # Compute time and speed
    dt = (gps.TS_or_Distance.iloc[1] - gps.TS_or_Distance.iloc[0]).total_seconds()
    speed = (distance_between_2GPS / dt) * 3.6

    gps['GPS_distance'] = [0, distance_between_2GPS]
    gps['GPS_dt'] = [0, dt]
    gps['GPS_speed'] = (0, speed)

    # Nodes
    geom = match_response['routes'][0]['geometry']
    nodes = pd.DataFrame(geom['coordinates'], columns=['lon', 'lat'])
    d = match_response['routes'][0]['legs'][0]['annotation']['distance']
    d.insert(0, 0)  # The distance from the first node to itself is 0
    nodes['distance'] = d

    return nodes, distance_between_2GPS, speed


def route_distance(lon1, lat1, lon2, lat2):
    # Routing distance is not very accurate

    c = [(lon1, lat1), (lon2, lat2)]

    # Call the Route service
    timeout = 20
    client = osrm.Client(host='http://liradbdev.compute.dtu.dk:5000', profile='car', timeout=timeout)
    response = client.route(coordinates=c, overview=osrm.overview.full, annotations=True)

    # Distance between the gps points
    d = response['routes'][0]['distance']
    # gps['distance'] = [0, distance_between_2GPS]

    return d


def interpolate_trip(all_sensor_data, out_dir='.', add_sensors=[], file_suff='GM', recreate=False):
    # Interpolate
    # lx = l1+(dl*dtx/dt)
    # gps_data['lat_start'] + (gps_data['dlat']*/gps_data['dt'])
    all_sensor_data = all_sensor_data.loc[all_sensor_data['T'].isin(np.append(add_sensors, ['track.pos', 'acc.xyz']))]

    print('Interpolating')
    out_filename_sensor = '{0}/sensor/{1}.csv'.format(out_dir, file_suff)
    out_filename_gps = '{0}/gps/{1}.csv'.format(out_dir, file_suff)

    # if os.path.exists(out_filename_sensor) and not recreate:
    #     print('Loading interpolated data from file: {0}'.format(out_filename))
    #     all_sensor_data = pd.read_pickle(out_filename)
    #     gps_out_filename = out_filename.replace('.pickle','_GPSonly.pickle')
    #     gps =  pd.read_pickle(gps_out_filename)
    #     return all_sensor_data, gps

    # Create segment dataframe
    gps = all_sensor_data[all_sensor_data['T'] == 'track.pos']
    gps = gps[['TS_or_Distance', 'T', 'lon', 'lat', 'lon_map', 'lat_map', 'way_id', 'street_name']]
    t1 = gps[0:-1]
    t1.reset_index(inplace=True)  # the last el. not included
    t2 = gps[1:]
    t2.reset_index(inplace=True)
    gps_data_segments = pd.merge(t1.drop(['lat', 'lon'], axis=1), t2.drop(['lat', 'lon'], axis=1), left_index=True,
                                 right_index=True, suffixes=('_start', '_end'))

    # Filter
    # gps_data_segments = gps_data_segments[gps_data_segments['dt']<2] #consider only these segmengps_data_segments for interpolation
    # gps_data_segments.reset_index(inplace=True)

    # Get time diff. between 2 meas.
    gps_data_segments['dt'] = (gps_data_segments['TS_or_Distance_end'] - gps_data_segments['TS_or_Distance_start'])
    gps_data_segments['dt'] = gps_data_segments['dt'].apply(lambda row: row.total_seconds())

    # Get lat and long dif in gps_data_segments 
    gps_data_segments['dlat'] = gps_data_segments['lat_map_end'] - gps_data_segments['lat_map_start']
    gps_data_segments['dlon'] = gps_data_segments['lon_map_end'] - gps_data_segments['lon_map_start']

    # Find a matching segment
    all_sensor_data['gps_segment'] = all_sensor_data.TS_or_Distance.apply(lambda t: gps_data_segments[
        (t > gps_data_segments.TS_or_Distance_start) & (t < gps_data_segments.TS_or_Distance_end)])
    all_sensor_data['gps_segment'] = all_sensor_data['gps_segment'].apply(lambda df: np.NaN if df.empty else df)
    all_sensor_data.dropna(subset=['gps_segment'], inplace=True)
    all_sensor_data.reset_index(inplace=True, drop=True)

    # Extract
    all_sensor_data['GPS_TS_or_Distance_start'] = all_sensor_data['gps_segment'].apply(
        lambda df: df['TS_or_Distance_start'].iloc[0])
    all_sensor_data['GPS_lat_map_start'] = all_sensor_data['gps_segment'].apply(lambda df: df['lat_map_start'].iloc[0])
    all_sensor_data['GPS_lon_map_start'] = all_sensor_data['gps_segment'].apply(lambda df: df['lon_map_start'].iloc[0])
    all_sensor_data['GPS_dt'] = all_sensor_data['gps_segment'].apply(lambda df: df['dt'].iloc[0])
    all_sensor_data['GPS_dlat'] = all_sensor_data['gps_segment'].apply(lambda df: df['dlat'].iloc[0])
    all_sensor_data['GPS_dlon'] = all_sensor_data['gps_segment'].apply(lambda df: df['dlon'].iloc[0])
    all_sensor_data['way_id'] = all_sensor_data['gps_segment'].apply(lambda df: df['way_id_start'].iloc[0])
    all_sensor_data['street_name_start'] = all_sensor_data['gps_segment'].apply(
        lambda df: df['street_name_start'].iloc[0])
    all_sensor_data['street_name_end'] = all_sensor_data['gps_segment'].apply(lambda df: df['street_name_end'].iloc[0])
    # all_sensor_data['hint'] =  all_sensor_data['gps_segment'].apply(lambda df: df['hint_start'].iloc[0])
    all_sensor_data.drop(columns=['gps_segment'], inplace=True, axis=1)

    # Filter?
    all_sensor_data = all_sensor_data[all_sensor_data['GPS_dt'] < 5]

    # Compute dtx 
    all_sensor_data['dtx'] = (all_sensor_data['TS_or_Distance'] - all_sensor_data['GPS_TS_or_Distance_start'])
    all_sensor_data['dtx'] = all_sensor_data['dtx'].apply(lambda row: row.total_seconds())

    # Compute lat and lon
    all_sensor_data['lat_int'] = all_sensor_data['GPS_lat_map_start'] + (
                all_sensor_data['GPS_dlat'] * all_sensor_data['dtx']) / all_sensor_data['GPS_dt']
    all_sensor_data['lon_int'] = all_sensor_data['GPS_lon_map_start'] + (
                all_sensor_data['GPS_dlon'] * all_sensor_data['dtx']) / all_sensor_data['GPS_dt']

    # var_data.drop(['GPS_TS_or_Distance_start', 'GPS_lat_start','GPS_lon_start', 'GPS_dt', 'GPS_dlat', 'GPS_dlon', 'dtx'],axis=1,inplace=True)

    # Select columns
    all_sensor_data.drop(
        ['lat', 'lon', 'way_id', 'GPS_dlat', 'GPS_dlon', 'dtx', 'street_name', 'GPS_TS_or_Distance_start',
         'GPS_lat_map_start', 'GPS_lon_map_start', 'lat_map', 'lon_map'], axis=1, inplace=True)
    all_sensor_data.reset_index(inplace=True, drop=True)

    # Extract Message
    # print('- Extracting message')
    all_sensor_data = pd.concat(
        [all_sensor_data.drop(['Message'], axis=1), all_sensor_data['Message'].apply(pd.Series)], axis=1)
    # GM_map_matched_data.drop([0,'@vid'], axis=1,inplace=True)

    # Save results
    all_sensor_data.to_csv(out_filename_sensor)
    all_sensor_data.to_pickle(out_filename_sensor.replace('.csv', '.pickle'))
    print('Saved interpolation output to: {0}'.format(out_filename_sensor))

    gps.to_csv(out_filename_gps)
    gps.to_pickle(out_filename_gps.replace('.csv', '.pickle'))
    print('Saved GPS recorded data to: {0}'.format(out_filename_gps))

    return all_sensor_data, gps
