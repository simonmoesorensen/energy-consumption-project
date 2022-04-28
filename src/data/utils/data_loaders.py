"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2  # pip install psycopg2==2.7.7 or pip install psycopg2==2.7.7
from json import loads
import sys, os, glob
from datetime import datetime
import pickle
from json import loads
# from sshtunnel import SSHTunnelForwarder

SSH_ADDRESS = 'thinlinc.compute.dtu.dk'


# SSH_ADDRESS = 'login2.hpc.dtu.dk'

def filter_DRDtrips_by_year(DRD_trips, sel_2021=False, sel_2020=False):
    DRD_trips['Datetime'] = pd.to_datetime(DRD_trips['Created_Date'])
    DRD_trips['Year'] = DRD_trips['Datetime'].apply(lambda row: row.year)
    if sel_2020:
        DRD_trips = DRD_trips[DRD_trips['Year'] == 2020]
    elif sel_2021:
        DRD_trips = DRD_trips[DRD_trips['Year'] == 2021]
    DRD_trips.drop(['Datetime'], axis=1, inplace=True)
    return DRD_trips


def drop_duplicates(DRD_data, iri):
    # Drop duplicate columns (due to ocassical errors in database)
    DRD_data = DRD_data.T.drop_duplicates().T  #
    if iri is not None:
        iri = iri.T.drop_duplicates().T
    return DRD_data, iri


def load_GM_data(GM_TaskId, ssh_tunnel, conn_data, prod_db=False, out_dir='.', all_sensors=False,
                 add_sensors=[], load_nrows=-1):
    # Set up connection
    print("\nConnecting to PostgreSQL database to load the DRD data")

    if prod_db:
        print("\nConnecting to production database")
        db_data = conn_data['prod']
    else:
        print("\nConnecting to development database")
        db_data = conn_data['dev']

    db = db_data['database']
    username = db_data['user']
    password = db_data['password']
    host = db_data['host']
    port = db_data['port']

    # Connection
    conn = psycopg2.connect(database=db, user=username, password=password, host=host,
                            port=port)

    # quory = 'SELECT "lat", "lon" FROM "Measurements" 
    # quory = 'SELECT  "lat", "lon" FROM "Measurements" WHERE "Trips"."TaskId"=\'{0}\' ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
    # quory = 'SELECT  "lat", "lon" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE "Trips"."TaskId"=\'{0}\' ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
    if all_sensors:
        print('Using all sensors')
        if load_nrows != -1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(
                GM_TaskId, load_nrows)
        else:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC'.format(
                GM_TaskId)

    else:
        sensors = ['track.pos', 'acc.xyz', 'obd.spd_veh']
        if add_sensors:
            sensors = sensors + add_sensors
        sensors = str(tuple(sensors))
        print('Loading: ', sensors)
        if load_nrows != -1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC LIMIT {2}'.format(
                GM_TaskId, sensors, load_nrows)
        else:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC'.format(
                GM_TaskId, sensors)

    # print(quory)
    print('Fetching from database')
    cursor = conn.cursor()
    meas_data = pd.read_sql(quory, conn, coerce_float=True)
    meas_data.reset_index(inplace=True, drop=True)
    print('Database fetch finished')

    # Extract message
    # =================#
    meas_data['Message'] = meas_data.message.apply(lambda msg: filter_keys(loads(msg)))
    meas_data.drop(columns=['message'], inplace=True, axis=1)
    meas_data.reset_index(inplace=True, drop=True)
    meas_data = meas_data[['TS_or_Distance', 'T', 'lat', 'lon', 'Message']]

    # Extract day and time
    # =================#
    meas_data['Date'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).date())
    meas_data['Time'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).time())
    meas_data.sort_values(by='Time', inplace=True)

    # Get GM trips info #
    # =================#
    print('Loading GM trip information')
    trip_info = get_GM_trips_info(conn_data, ssh_tunnel, task_id=GM_TaskId, prod_db=False, only_GM=True)

    # Close connection
    # ==============#
    if (conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")

    # Save files
    # ==============#
    if all_sensors:
        measurements_path = (out_dir + 'GM_db_meas_data_{0}_allsensors.csv').format(GM_TaskId)
    else:
        measurements_path = (out_dir + 'GM_db_meas_data_{0}.csv').format(GM_TaskId)

    measurements_path = measurements_path.replace('//', '/')
    print('Saving files')
    # Save loaded data
    meas_data.to_csv(measurements_path)
    meas_data.to_pickle(measurements_path.replace('.csv', '.pickle'))

    # Save trips info
    trip_info_path = out_dir + 'GM_db_trips_info_{0}.csv'.format(GM_TaskId)

    trip_info.to_csv(trip_info_path)
    trip_info.to_pickle(trip_info_path.replace('.csv', '.pickle'))
    print('Finished saving files')
    return meas_data, trip_info


def get_GM_trips_info(conn_data, ssh_tunnel, prod_db=False, task_id=None, only_GM=False, GM_year=None,
                      GM_min_distance=10):
    # Set up connection
    print("\nConnecting to PostgreSQL database to load the DRD data")

    if prod_db:
        print("\nConnecting to production database")
        db_data = conn_data['prod']
    else:
        print("\nConnecting to development database")
        db_data = conn_data['dev']

    db = db_data['database']
    username = db_data['user']
    password = db_data['password']
    host = db_data['host']
    port = db_data['port']

    # Connection    
    conn = psycopg2.connect(database=db, user=username, password=password, host=host,
                            port=port)

    # Quory
    quory = 'SELECT * FROM public."Trips" WHERE "Trips"."Fully_Imported"=\'True\' ORDER BY "TaskId" ASC'

    # Set cursor
    cursor = conn.cursor()

    d = pd.read_sql(quory, conn, coerce_float=True)
    d['Datetime'] = pd.to_datetime(d['StartTimeUtc'])

    # Select GM
    d = d[d['TaskId'] != 0]

    if GM_min_distance:
        d = d[d['DistanceKm'] > GM_min_distance]  # trips with distance>0km

    if GM_year:
        d['Year'] = d['Datetime'].apply(lambda row: row.year)
        d = d[d['Year'] == GM_year]
        d.drop(['Datetime', 'Year'], axis=1, inplace=True)
        d.reset_index(drop=True, inplace=True)

    if task_id:
        d = d[d['TaskId'] == task_id]

    # Close the connection
    cursor.close()
    conn.close()

    return d


def filter_keys(msg):
    remove_list = ['id', 'start_time_utc', 'end_time_utc', 'start_position_display',
                   'end_position_display', 'device', 'duration', 'distanceKm', 'tag',
                   'personal', '@ts', '@uid', '@t', 'obd.whl_trq_est', '@rec']
    msg = {k: v for k, v in msg.items() if k not in remove_list}
    return msg


def extract_string_column(sql_data, col_name='message'):
    # if json
    try:
        sql_data[col_name] = sql_data[col_name].apply(lambda message: loads(message))
    except:
        pass
    keys = sql_data[col_name].iloc[0].keys()
    n_keys = len(keys)
    for i, key in enumerate(keys):
        print('Key {0}/{1}'.format(i, n_keys))
        sql_data[key] = sql_data[col_name].apply(lambda col_data: col_data[key])

    sql_data.drop(columns=[col_name], inplace=True, axis=1)
    return sql_data


def check_nans(sql_data, is_aran=False, exclude_cols=[]):
    n_rows = sql_data.shape[0]
    for col in sql_data.columns:
        if col in exclude_cols:
            continue
        n_nans = sql_data[col].isna().sum()
        n_left = n_rows - n_nans
        print('Number of nans in {0}: {1}/{2}, left: {3}/{2}'.format(col, n_nans, n_rows, n_left))
    return
