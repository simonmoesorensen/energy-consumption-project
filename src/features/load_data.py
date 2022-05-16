import glob
import os
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent.parent
data_dir = root / 'data/processed'


def adjust_offsets(df):
    offsets = {
        'obd.strg_pos.value': 32768,
        'obd.strg_acc.value': 32768,
        'obd.strg_ang.value': 32768,
        'obd.acc_yaw.value': 2047,
        'obd.trac_cons.value': 80,
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
        'obd.ww_f_stat.value': 'front_wiper_status',
        'obd.odo.value': 'odometer',
        'obd.trac_cons.value': 'traction_instant_consumption',
        'obd.sb_rem_fl.value': 'driver_safety_belt_reminder',
    }

    return df.rename(names, axis=1)


def resample_df(df, sampling):
    df = df.groupby('street_name_start').resample(sampling).mean()
    df = df.reset_index()
    df = df.set_index('TS_or_Distance')
    df = df.sort_index()
    return df


def drop_cols(df):
    df = df.drop(['index', 'index_diff0', '@vid', 'W', 's', 'start_hour'],
                 axis=1)
    return df


def load_trip(route, trip, pass_=None, sampling='5s'):
    if pass_ is None:
        trip_pass = f"trip_{trip}_*.pickle"
    else:
        trip_pass = f"trip_{trip}_{pass_}.pickle"

    passes = dict()
    for i, file in enumerate(glob.glob(os.path.join(data_dir, route, trip_pass))):
        df = pd.read_pickle(file, )
        df = df.set_index('TS_or_Distance')

        df = resample_df(df, sampling)
        df = adjust_offsets(df)
        df = rename_df(df)
        df = drop_cols(df)

        passes[i] = df

    return passes


if __name__ == '__main__':
    df = load_trip('M3_VH', '7448', '0', sampling='1s')
    print(df[0])
