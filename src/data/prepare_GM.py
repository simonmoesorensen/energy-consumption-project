"""
@author: Milena Bajic (DTU Compute)
"""

import sys, os, glob
import argparse, json
from pathlib import Path

import psycopg2
import pandas as pd
from utils.data_loaders import *
from utils.matching import *
from utils.plotting import *
from utils.helpers import *
from utils.analysis import *
import datetime
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.signal import (
    find_peaks,
    argrelmin,
    argrelextrema,
    find_peaks_cwt,
    peak_widths,
)
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# =================================#
# SETTINGS
# =================================#
dir_above = "/".join(os.getcwd().split("/")[0:])

# Script arguments
parser = argparse.ArgumentParser(description="Please provide command line arguments.")

# Route and trip
parser.add_argument(
    "--route", help="Process all trips on this route, given in json file."
)
parser.add_argument("--trip", type=int, help="Process this trip.")

# Json files
parser.add_argument(
    "--routes_file",
    default="{0}/config/routes.json".format(dir_above),
    help="Json file with route information.",
)
parser.add_argument(
    "--conn_file",
    default="{0}/config/.connection.json".format(dir_above),
    help="Json file with connection information.",
)

# Processing options
parser.add_argument("--skip_interp", action="store_true", help="Do not interpolate.")
parser.add_argument(
    "--load_add_sensors", action="store_true", help="Load additional sensors."
)

# Plotting
parser.add_argument(
    "--plot",
    action="store_true",
    help="To plot data on Open Streep Map, pass true. The default is False.",
)
parser.add_argument(
    "--only_load_pass_plots", action="store_true", help="Only load GM pass plots."
)

# Recreate
parser.add_argument(
    "--recreate",
    action="store_true",
    help="Recreate all files. If recreate is false and the files are present, the data will be loaded from files.",
)
parser.add_argument(
    "--recreate_interp",
    action="store_true",
    help="Recreate only interpolation files. If recreate is false and the files are present, the data will be loaded from files.",
)

parser.add_argument(
    "--preload_mapmatch",
    action="store_true",
    help="Preload mapmatching files for interpolation",
)

# Output directory
parser.add_argument("--out_dir", default="data", help="Output directory.")

# Development modes
parser.add_argument(
    "--dev_mode",
    action="store_true",
    help="Development mode. Will process a limited number of lines.",
)
parser.add_argument(
    "--dev_mode_n_lines",
    type=int,
    default=10000,
    help="Process this number of lines in development mode.",
)
parser.add_argument(
    "--only_load_trips_data",
    action="store_true",
    help="Only load GM trips from the database to explore. Skip the rest.",
)

# # Database ssh tunnel
# parser.add_argument(
#     "--ssh_user", help="Your DTU login id"
# )

# parser.add_argument(
#     "--ssh_pass", help="Your DTU login password"
# )

# Parse arguments
args = parser.parse_args()

# Setup
trip = args.trip
route = args.route
routes_file = args.routes_file
conn_file = args.conn_file
out_dir_base = (
    args.out_dir
)  # if you want results on dtu, use: --out_dir /dtu-compute/lira/ml_data/data/GM_processesed_data
load_add_sensors = args.load_add_sensors
skip_interpolation = args.skip_interp

# Recreate
recreate = args.recreate
recreate_interp = True if args.recreate else args.recreate_interp
preload_mapmatch = args.preload_mapmatch

# Plotting
plot = args.plot
only_load_pass_plots = args.only_load_pass_plots
plot_filename = lambda x: Path(__file__).parent.parent.parent / f'figures/{x}.png'

# Developmentand exploration modes
dev_mode = args.dev_mode
dev_nrows = args.dev_mode_n_lines
only_load_trips_data = args.only_load_trips_data

# SSH
# ssh_data = {'user': args.ssh_user,
#             'pass': args.ssh_pass}

plot_html_map = False
# =========================#
# PREPARATION
# =========================#
# Load route info file
with open(routes_file, "r") as f:
    route_data = json.load(f)

# Load connection info file
with open(conn_file, "r") as f:
    conn_data = json.load(f)


if only_load_trips_data:
    GM_trip_info = get_GM_trips_info(conn_data, None, prod_db=False, only_GM=True, GM_year=2021)
    sys.exit()

# Exit if both route and trip are passed
if route and trip:
    print(
        "Do not choose both route and trip. If a route is passed - all trips in the json file for this route will be used. If a trip is passed, only this trip will be used and the route name will be loaded from the json file."
    )
    sys.exit(0)

# If none passed, also exit
if not route and not trip:
    print(
        "Choose either a route or a trip. If a route is passed - all trips in the json file for this route will be used. If a trip is passed, only this trip will be used and the route name will be loaded from the json file."
    )
    sys.exit(0)


# If route passed, use all trips from the json file for this route
if route:
    trips_thisroute = route_data[route]["GM_trips"]

    # If no trips found for user passed route, then exit
    if not trips_thisroute:
        print(
            "No trips found for this route in the json file. Please add trips for this route."
        )
        sys.exit(0)

# If a trip pased, find route for this trip in the json file
if trip:
    trips_thisroute = [trip]
    route = find_route(trip, route_data)

# Additional sensors to load
add_sensors = []
if load_add_sensors:
    steering_sensors = ['obd.strg_pos', 'obd.strg_acc','obd.strg_ang']
    # wheel_pressure_sensors =  ['obd.whl_prs_rr', 'obd.whl_prs_rl','obd.whl_prs_fr','obd.whl_prs_fl']
    other_sensors = ['obd.acc_yaw', 'obd.trac_cons', 'obd.odo', 'obd.spd', 'obd.ww_f_stat', 'obd.sb_rem_fl']
    add_sensors = steering_sensors + other_sensors
    #'obd.sb_stat_rc', 'obd.sb_stat_rl', 'obd.sb_stat_rr', 'obd.ww_f_stat', 'obd.strg_ang'

# ==================================#
# ===== INPUT/OUTPUT DIRECTORY =====#

# Create output directory for this route if it does not exist
out_dir_raw = "{0}/raw/{1}/".format(out_dir_base, route)
out_dir_interim = "{0}/interim/{1}/".format(out_dir_base, route)


# if load_add_sensors:
#     out_dir_raw = "{0}_add_sensors".format(out_dir_raw)
#     out_dir_interim = "{0}_add_sensors".format(out_dir_interim)

#if not os.path.exists(out_dir_route):
#    os.makedirs(out_dir_route)

# # Create putput directory for route plots
# out_dir_plots = "{0}/plots".format(out_dir_route)
# if not os.path.exists(out_dir_plots):
#     os.makedirs(out_dir_plots)

# # Create putput directory for merged plots
# out_dir_plots_merged = "{0}/plots_merged".format(out_dir_route)

# Create output directory for passes
out_dir_base = out_dir_base.replace('//', '/')
out_dir_raw = out_dir_raw.replace('//', '/')
out_dir_interim = out_dir_interim.replace('//', '/')

if not os.path.exists(out_dir_interim):
    os.makedirs(out_dir_interim)

if not os.path.exists(out_dir_raw):
    os.makedirs(out_dir_raw)

print("\n=====================")
print("Route: ", route)
print("Trips: ", trips_thisroute)
print("Dev mode: ", dev_mode)
print("Load additional sensors: ", load_add_sensors)
print("Additional sensors: ", add_sensors)
print("Output directory raw: ", out_dir_raw)
print("Output directory processed: ", out_dir_interim)
print("=====================\n")

# =================================#
# ======== PROCESS TRIPS ==========#
for trip in trips_thisroute:
    # Load car data from db or file
    print("\nProcessing GM trip: ", trip)

    # ============== Only load options ===============#
    # Only load matplotlib plots to make html map plots (use this to make map plots)
    if only_load_pass_plots:
        # pattern = "{0}/GM_trip_*_pass_*.pickle".format(out_dir_plots)
        # for name in glob.glob(pattern):
        #     # if os.path.exists(name.replace('.png','_printout.png')):
        #     #    continue

        #     filename = name.split(out_dir_plots + "/")[1]
        #     print("Using: ", name)
        #     # plot_geolocation(DRD_data['lon_map'],  DRD_data['lat_map'], name = 'DRD_{0}_GPS_mapmatched_gpspoints'.format(trip), out_dir = out_dir_plots, plot_firstlast = 100, preload = preload_plots)

        #     plot_geolocation(
        #         full_filename=name, out_dir=out_dir_plots, plot_html_map=True
        #     )

        # # Create a merged pdf file with plots
        # pattern = "{0}/GM_trip*.png".format(out_dir_plots)
        # files = glob.glob(pattern)
        # files = [f for f in files if "_map.png" not in f]
        # files.sort(
        #     key=lambda f: (int(f.split("/")[-1].split("_")[2]), sort2(f), sort3(f))
        # )

        # from PIL import Image, ImageFont, ImageDraw

        # imagelist = []
        # for file in files:
        #     name = file.split("/")[-1].replace(".png", "")
        #     img = Image.open(file).convert("RGB")
        #     draw = ImageDraw.Draw(img)
        #     if "pass" in name:
        #         font = ImageFont.truetype(r"/Library/Fonts/Arial Unicode.ttf", 40)
        #         draw.text((50, 20), name, "black", font)
        #     elif "minima" in name:
        #         font = ImageFont.truetype(r"/Library/Fonts/Arial Unicode.ttf", 10)
        #         draw.text((50, 0), name, "black", font)
        #     else:
        #         font = ImageFont.truetype(r"/Library/Fonts/Arial Unicode.ttf", 10)
        #         draw.text((70, 20), name, "black", font)
        #     imagelist.append(img)

        # out_filename = "{0}/GM_route_{1}_merged_plots.pdf".format(
        #     out_dir_plots_merged, route
        # )
        # imagelist[0].save(out_filename, save_all=True, append_images=imagelist[1:])
        # print("Merge images saved as: ", out_filename)

        continue  # skip the rest of the code and go to the next trip

    # ============== Load the trip ===============#
    filename = "{0}GM_db_meas_data_{1}.pickle".format(out_dir_raw, trip)
    if os.path.exists(filename) and not recreate:
        print("Reloading GM trip from file: ", trip)
        GM_data = pd.read_pickle(filename)
        if dev_mode:
            GM_data = GM_data.head(dev_nrows)
        GM_trip_info = pd.read_pickle('{0}GM_db_trips_info_{1}.pickle'.format(out_dir_raw, trip))
    else:
        prod_db = False

        if prod_db:
            db_data = conn_data['prod']
        else:
            db_data = conn_data['dev']

        host = db_data['host']
        port = db_data['port']

        # print('Connecting to the SSH Tunnel...')

        # ssh_tunnel = SSHTunnelForwarder(
        #     ssh_address_or_host=(SSH_ADDRESS, 22),
        #     ssh_username=ssh_data['user'],
        #     ssh_password=ssh_data['pass'],
        #     remote_bind_address=(host, int(port))
        # )

        # ssh_tunnel.start()
        ssh_tunnel = None
        # print('Success')

        print("Loading GM trip from the database: ", trip)
        if dev_mode:
            GM_data, GM_trip_info = load_GM_data(
                trip,
                ssh_tunnel,
                conn_data=conn_data,
                out_dir=out_dir_raw,
                add_sensors=add_sensors,
                load_nrows=dev_nrows,
            )
        else:
            GM_data, GM_trip_info = load_GM_data(
                trip,
                ssh_tunnel,
                conn_data=conn_data,
                out_dir=out_dir_raw,
                add_sensors=add_sensors,
            )

    # ============== Map match the trip===============#
    mapmatched_dir = out_dir_interim + 'mapmatched'

    passes = os.listdir(mapmatched_dir)
    if os.path.exists(mapmatched_dir):
        has_mapmatched = len(passes) > 0
    else:
        has_mapmatched = False

    if not preload_mapmatch:
        has_mapmatched = False

    if not has_mapmatched:
        print("Starting map matching")

        # GPS dataframe
        gps = GM_data[GM_data["T"] == "track.pos"]
        GM_data = GM_data[GM_data["T"] != "track.pos"]

        map_filename = "mapmatched_gpspoints_fulltrip_{0}".format(trip)
        host = conn_data["osrm"]["host"]

        if not os.path.exists(mapmatched_dir):
            os.makedirs(mapmatched_dir)

        gps_mapmatched = map_match_gps_data(
            gps,
            host=host,
            is_GM=True,
            out_dir=mapmatched_dir,
            out_file_suff="GM_{0}".format(trip),
        )

        if gps_mapmatched is None:
            print(f'No mapmatching data found for {trip=}. Continuing to next trip.')
            continue

        # Plot map matched
        # plot_filename = "{0}/GM_trip_{1}_mapmatched_gpspoints_fulltrip.png".format(
        #     out_dir_plots, trip
        # )
        # ax = plt.scatter(gps_mapmatched["lon_map"], gps_mapmatched["lat_map"], s=5)
        # fig = ax.get_figure()
        # fig.suptitle("GM trip {0} ".format(trip))
        # fig.savefig(plot_filename.replace("distance0", "indexdiff0"))
        # print("Wrote to: ", plot_filename)

        # ============== Outlier det(DBScan) ===============#
        print("Removing outliers")
        model = DBSCAN(eps=0.01, min_samples=20).fit(gps_mapmatched[["lon_map", "lat_map"]])
        gps_mapmatched["label"] = model.labels_

        # Find and plot clusters
        # ax = plt.scatter(
        #     gps_mapmatched["lon_map"],
        #     gps_mapmatched["lat_map"],
        #     s=5,
        #     c=gps_mapmatched["label"],
        # )
        # fig.suptitle("GM trip {0}: Clusters".format(trip))
        # fig = ax.get_figure()
        # fig.savefig(
        #     plot_filename.replace("_mapmatched", "_wtr1stpoint_mapmatched").replace(
        #         ".png", "_clusters.png"
        #     )
        # )

        # Check which labels to keep
        nc = gps_mapmatched["label"].value_counts(normalize=True, sort=True).to_dict()
        keep_labels = []
        for l, count in nc.items():
            if count > 0.01:
                keep_labels.append(l)
        print(keep_labels)

        # Remove outliers
        gps_mapmatched = gps_mapmatched[gps_mapmatched["label"].isin(keep_labels)]
        gps_mapmatched.reset_index(drop=True, inplace=True)
        ax = plt.scatter(gps_mapmatched["lon_map"], gps_mapmatched["lat_map"], s=5)
        fig.suptitle("GM trip {0}: Removed outliers".format(trip))
        fig = ax.get_figure()
        fig.savefig(plot_filename("_removed_outliers"))
        print("Wrote to: ", plot_filename("_removed_outliers"))

        # Plot
        # plot_geolocation(gps_mapmatched['lon_map'], gps_mapmatched['lat_map'], name= map_filename,out_dir = out_dir_plots, plot_firstlast = 10, do_open = False)
        # plot_geolocation(gps_result['lon_map'][0:1000], gps_result['lat_map'][0:1000], name = 'GM_{0}_GPS_mapmatched_points_start'.format(trip), out_dir = our_dir_plots, plot_firstlast = 5)

        # ============== Split the trip into passes ===============#
        # GM_int_data = GM_int_data.iloc[:50000]
        print("Splitting into passes")

        gps_mapmatched.reset_index(drop=True, inplace=True)
        gps_mapmatched["index"] = gps_mapmatched.index

        # The first point
        lat0 = gps_mapmatched.iloc[0]["lat_map"]
        lon0 = gps_mapmatched.iloc[0]["lon_map"]
        t0 = gps_mapmatched.iloc[0]["TS_or_Distance"]
        i0 = gps_mapmatched.iloc[0]["index"]

        # Compute differences wtr to the first point
        gps_mapmatched["distance0"] = gps_mapmatched.apply(
            lambda row: haversine_distance(lat0, row["lat_map"], lon0, row["lon_map"]),
            axis=1,
        )
        gps_mapmatched["time_diff0"] = gps_mapmatched.apply(
            lambda row: pd.Timedelta(row["TS_or_Distance"] - t0), axis=1
        )
        gps_mapmatched["time_diff0"] = gps_mapmatched["time_diff0"].apply(
            lambda row: row.seconds / 60
        )
        gps_mapmatched["index_diff0"] = gps_mapmatched.apply(
            lambda row: row["index"] - i0, axis=1
        )

        # Fit index difference vs distance
        rmse = {}
        for d in list(range(10, 50, 5)):
            model = polynomial_model(degree=d)
            x = gps_mapmatched["index_diff0"].to_numpy().reshape(-1, 1)
            y = gps_mapmatched["distance0"].to_numpy().reshape(-1, 1)
            x = np.nan_to_num(x)
            y = np.nan_to_num(y)

            try:
                model.fit(x, y)
                pred = model.predict(x)
                rmse[d] = mean_squared_error(y, pred)
            except:
                continue

        # If all fits failed, skip this trip
        if len(rmse.keys()) == 0:
            continue

        # Best fit
        best_d = min(rmse, key=rmse.get)
        model = polynomial_model(degree=best_d)
        model.fit(
            gps_mapmatched["index_diff0"].to_numpy().reshape(-1, 1),
            gps_mapmatched["distance0"].to_numpy().reshape(-1, 1),
        )
        pred = model.predict(gps_mapmatched["index_diff0"].to_numpy().reshape(-1, 1))

        # Find valleys
        pred_inv = -1 * pred.reshape(-1)
        p = pred.max() / 8
        minima_indices_cand = find_peaks(pred_inv, prominence=p, distance=500)[0]
        # w = peak_widths(pred.reshape(-1), peaks)
        maxima_indices_cand = find_peaks(pred.reshape(-1), prominence=p, distance=500)[0]

        # Find array with minima/maxima
        # o = int(gps_mapmatched['distance0'].shape[0]/20)
        # minima_indices_cand = list(argrelmin(pred, order = o)[0]) # or pred
        minima_indices = []
        for i in minima_indices_cand:
            distance0 = gps_mapmatched[gps_mapmatched["index_diff0"] == i][
                "distance0"
            ].values[0]
            if i > 500:  # remove if those are first points when car is setting off
                minima_indices.append(i)

        for i in maxima_indices_cand:
            distance0 = gps_mapmatched[gps_mapmatched["index_diff0"] == i][
                "distance0"
            ].values[0]
            if i > 500:  # remove if those are first points when car is setting off
                minima_indices.append(i)

        print("Minima and maxima found at: ", minima_indices)

        # Plot distance difference wtr to time and save
        # plot_filename = (
        #     "{0}/GM_trip_{1}_distance0_wtr_time_mapmatched_gpspoints_fulltrip.png".format(
        #         out_dir_plots, trip
        #     )
        # )
        # ax = gps_mapmatched.plot("time_diff0", "distance0", kind="scatter", s=3)
        # fig = ax.get_figure()
        # ax.set_title("GM trip: {0}".format(trip))
        # # fig.savefig(plot_filename)

        # # Plot index wtr to time
        # ax = gps_mapmatched.plot(
        #     "index_diff0", "distance0", kind="scatter", s=3, label="Data"
        # )
        # ax.plot(
        #     gps_mapmatched["index_diff0"].to_numpy(), pred, c="red", label="Fitted function"
        # )

        # Add minima to the plot (new passes) and save the figure
        # for i in minima_indices:
        #     ax.axvline(x=i, c="b")
        # ax.legend(loc="lower right", frameon=False)
        # plt.tight_layout()
        # fig = ax.get_figure()
        # fig.savefig(plot_filename.replace("fulltrip", "fulltrip_minima"))
        # print("Wrote to: ", plot_filename)

        # List with borders of different passes
        lower_borders = [0] + minima_indices
        upper_borders = minima_indices + [gps_mapmatched.shape[0]]
        borders = list(zip(lower_borders, upper_borders))
        print(borders)

        # ============== Process different passes ==============#
        for i, (low, up) in enumerate(borders):
            # pass start gps, compute distance and take end from at least 100m from the start
            print("Processing trip: {0}, trip: {1}".format(trip, i))

            # if super small pass, ignore
            if up - low < 500:
                continue

            # upb = up-200
            upb = up - 1

            # Df for this pass
            gps_car_pass = gps_mapmatched[gps_mapmatched["index_diff0"].between(low, upb)]
            gps_car_pass.drop(["distance0", "time_diff0", "label"], axis=1, inplace=True)
            gps_car_pass.reset_index(drop=True, inplace=True)
            s = gps_car_pass.shape
            print("pass: {0}, borders: {1}-{2}, shape: {3}".format(i, low, upb, s))

            # Plot the pass
            # fig = plot_geolocation(
            #     gps_car_pass["lon_map"],
            #     gps_car_pass["lat_map"],
            #     name="GM_trip_{0}_pass_{1}_GPS_mapmatched_gpspoints".format(trip, i),
            #     out_dir=out_dir_plots,
            #     title="GM trip: {0}, pass:{1}".format(trip, i),
            #     plot_firstlast=100,
            # )

            # Find full GM data for this pass
            t0 = gps_car_pass["TS_or_Distance"].iloc[0]
            tf = gps_car_pass["TS_or_Distance"].iloc[-1]
            GM_pass = GM_data[GM_data["TS_or_Distance"].between(t0, tf)]

            # Merge map matched GPS with the full dataframe
            GM_pass_full_data = pd.concat(
                [GM_pass, gps_car_pass.drop(["Date", "Time"], axis=1)], ignore_index=True
            )
            GM_pass_full_data.sort_values(by="TS_or_Distance", ascending=True, inplace=True)

            # Remove not needed columns
            # GM_pass_full_data.drop([0,'Date','Time'],axis=1,inplace=True)

            # Set Message to nan if from GPS
            GM_pass_full_data["Message"].mask(
                GM_pass_full_data["T"] == "track.pos", inplace=True
            )
            GM_pass_full_data.reset_index(drop=True, inplace=True)

            # Save the pass df
            out_passes_noint_dir = out_dir_interim + 'no_interpolation'

            if not os.path.exists(out_passes_noint_dir):
                os.makedirs(out_passes_noint_dir)

            out_filename = "{0}/GM_trip_{1}_pass_{2}.pickle".format(
                out_passes_noint_dir, trip, i
            )

            GM_pass_full_data.to_pickle(out_filename)

            if not skip_interpolation:
                print("Interpolating.......")
                # Out filename
                inter_filename = "GM_trip_{0}_pass_{1}".format(trip, i)

                # Interpolate
                out_dir_interpolated = out_dir_interim + 'interpolated'
                if not os.path.exists(out_dir_interpolated):
                    os.makedirs(out_dir_interpolated)

                GM_int_data, gps = interpolate_trip(
                    all_sensor_data=GM_pass_full_data,
                    out_dir=out_dir_interpolated,
                    add_sensors=add_sensors,
                    file_suff=inter_filename,
                    recreate=recreate_interp,
                )
    else:
        print(f'Reloading {passes}')
        for i, trip in enumerate(passes):
            trip = trip.split('.')[0].split('_')[1]
            # Out filename
            inter_filename = "GM_trip_{0}_pass_{1}".format(trip, i)

            # Interpolate
            out_dir_interpolated = out_dir_interim + 'interpolated'
            if not os.path.exists(out_dir_interpolated):
                os.makedirs(out_dir_interpolated)

            # Save the pass df
            out_passes_noint_dir = out_dir_interim + 'no_interpolation'

            out_filename = "{0}/GM_trip_{1}_pass_{2}.pickle".format(
                out_passes_noint_dir, trip, i
            )

            GM_pass_full_data = pd.read_pickle(out_filename)
            GM_int_data, gps = interpolate_trip(
                all_sensor_data=GM_pass_full_data,
                out_dir=out_dir_interpolated,
                add_sensors=add_sensors,
                file_suff=inter_filename,
                recreate=recreate_interp,
            )
        # Interpolate the pass df
        # GM_map_matched_data = GM_map_matched_data.iloc[8000:9000]



            # Filter
            # GM_int_data = GM_int_data[GM_int_data["GPS_dt"] < 5]

            # Plot
            # GM_int_data['GPS_dt'].describe()
            #plot_geolocation(
            #     gps["lon_map"],
            #     gps["lat_map"],
            #     name="GM_trip_{0}_pass_{1}_GPS_mapmatched".format(trip, i),
            #     out_dir=out_dir_plots,
            #     plot_firstlast=10,
            #     plot_html_map=plot_html_map,
            #     title="GM trip: {0}, pass:{1}".format(trip, i),
            # )
            # plot_geolocation(
            #     GM_int_data["lon_int"][::300],
            #     GM_int_data["lat_int"][::300],
            #     name="GM_trip_{0}_pass_{1}_GPS_interpolated_300th".format(trip, i),
            #     out_dir=out_dir_plots,
            #     plot_firstlast=10,
            #     title="GM trip: {0}, pass:{1}, interpolated".format(trip, i),
            # )

    # Close all figures
    plt.close("all")
