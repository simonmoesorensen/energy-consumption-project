import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).parent.parent.parent
data_dir = root / 'data/processed'


def adjust_offsets(df):
    offsets = {
        'obd.strg_pos.value': 32768,
        'obd.strg_acc.value': 32768,
        'obd.strg_ang.value': 32768,
        'obd.acc_yaw.value': 2047,
        'obd.trac_cons.value': 160,
        'obd.xyz.z': 0.97,
    }

    for col in df:
        if col in offsets:
            df[col] = df[col] - offsets[col]

    return df


def rename_df(df):
    names = {
        'acc.xyz.z': 'acceleration_z',
        'acc.xyz.y': 'acceleration_y',
        'acc.xyz.x': 'acceleration_x',
        'obd.strg_pos.value': 'steering_position',
        'obd.strg_acc.value': 'steering_acceleration',
        'obd.strg_ang.value': 'steering_wheel_angle_offset',
        'obd.acc_yaw.value': 'yaw_rate',
        'obd.spd.value': 'velocity',
        'obd.ww_f_stat.value': 'front_wiper_status',
        'obd.odo.value': 'odometer',
        'obd.trac_cons.value': 'traction_instant_consumption',
        'obd.sb_rem_fl.value': 'driver_safety_belt_reminder',
    }

    return df.rename(names, axis=1)


def resample_df(df, sampling):
    df = df.resample(sampling).median()
    df = df.reset_index()
    df = df.set_index(['TS_or_Distance'])
    return df


def drop_cols(df):
    df = df.drop(['index',
                  'index_diff0',
                  '@vid',
                  'W',
                  's',
                  'start_hour',
                  'steering_wheel_angle_offset'],
                 axis=1)
    return df


def remove_outliers(df, columns, low_q=0.001, hi_q=0.999):
    for col in columns:
        q_low = df[col].quantile(low_q)
        q_hi = df[col].quantile(hi_q)

        df = df[(df[col] <= q_hi) & (df[col] >= q_low)]

    return df


def load_trip(route, trip, pass_=None, sampling='1s'):
    if pass_ is None:
        trip_pass = f"trip_{trip}_*.pickle"
    else:
        trip_pass = f"trip_{trip}_{pass_}.pickle"

    passes = dict()
    for i, file in enumerate(glob.glob(os.path.join(data_dir, route, trip_pass))):
        df = pd.read_pickle(file)

        df = df.set_index(['TS_or_Distance'])

        df = resample_df(df, sampling)

        df = adjust_offsets(df)
        df = rename_df(df)
        df = drop_cols(df)

        df['pass'] = i
        df['trip'] = trip
        df['route'] = route

        df = df[~df.isna().all(axis=1)]

        passes[i] = df

    return passes


def load_trips(routes, trips='all', sampling='1s'):
    assert type(routes) == list

    if trips != 'all':
        assert type(trips) == set or type(trips) == list
        trips = map(int, trips)
        trips = set(trips)

    all_df = []
    for route in routes:
        route_trips = set([int(path.split('_')[2]) for path in glob.glob(os.path.join(data_dir, route, '*.pickle'))])

        if trips == 'all':
            trips_iter = route_trips
        else:
            trips_iter = trips.intersection(route_trips)

        for trip in trips_iter:
            dfs = load_trip(route, trip, sampling=sampling)
            df = pd.concat(dfs.values())
            all_df.append(df)

    out_df = pd.concat(all_df)
    out_df = out_df.sort_index()
    return out_df


if __name__ == '__main__':
    df = load_trips(['M3_VH', 'CPH6_VH'], trips=[13201, 7448], sampling='1s')
    print(df)
