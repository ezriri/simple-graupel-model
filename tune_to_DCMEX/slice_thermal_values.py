## pulling out min, max, mean, sd of environmental variables for thermal values

import sys
import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset # !! new - building netcdf
import os # file saving 
import xarray as xr
from pathlib import Path
import importlib.util
from glob import glob
import seaborn as sns
import imageio
import tempfile
import pandas as pd
from metpy.calc import wind_direction, wind_speed
from metpy.units import units
from tracemalloc import start

sys.path.append('/home/users/esree/dcmex_instruments/faam_data/')
# loads of homemade functions <3 <3
from faam_data_functions import faam_mo_files, time_adjust, get_nc_variable, mask_data, flatten_20hz_variable, mo_mean, datetime_to_sec, cloud_pass_time
from faam_data_functions_v2 import faam_mo_files_v2, make_cdp_psd_v2, make_cdp_conc, cloud_pass_time_v2

date = '220719'

save_pth = '/home/users/esree/simple-graupel-model/tune_to_DCMEX/thermal_values/'

faam_mo_paths_dic = faam_mo_files_v2() # dictionary containing dates + paths to nc files
date_dict = faam_mo_paths_dic[date]
core_phys_pth = date_dict['cloud_physics_1hz']
faam_pth = date_dict['faam_core']
faam_pth_high = date_dict['faam_core_high']
mo_path = date_dict['mo_non_core_1hz']
mo_path_64hz = date_dict['mo_non_core'] 

### cloud droplet concentration #######################################
cdp_conc = make_cdp_conc(date_dict, file_name = 'cloud_physics') # this has no averaging

## faam alt + temp + tas + w - this is default 1hz, cba to change to higher one
alt_unit, alt_longname, alt = get_nc_variable(faam_pth_high, 'ALT_GIN')

# other wind components
faam_u_unit, faam_u_longname, faam_u = get_nc_variable(faam_pth_high, 'U_C')
faam_u = faam_u.rename({'Time': 'time'})
u_flat, u_time_high = flatten_20hz_variable(faam_u)
u = u_flat.compute() * units('m/s')
faam_v_unit, faam_v_longname, faam_v = get_nc_variable(faam_pth_high, 'V_C')
faam_v = faam_v.rename({'Time': 'time'})
v_flat, v_time_high = flatten_20hz_variable(faam_v)
v = v_flat.compute() * units('m/s')
uv_wind_direction = wind_direction(u, v, convention="from") # where wind coming from
uv_wind_speed = wind_speed(u, v)

faam_w_unit_high, faam_w_longname_high, faam_w_high = get_nc_variable(faam_pth_high, 'W_C')
faam_w_high = faam_w_high.rename({'Time': 'time'})
w_high_flat, w_time_high = flatten_20hz_variable(faam_w_high)

tas_unit_high, tas_longname_high, tas_high = get_nc_variable(faam_pth_high, 'TAS_RVSM')
tas_high = tas_high.rename({'Time': 'time'})
tas_flat, tas_time_high = flatten_20hz_variable(tas_high)

LWC_unit_64hz, LWC_longname_64hz, LWC_64hz = mo_mean(mo_path_64hz, ('LWC1','LWC2'))

combined_dic = {
    'w_faam_high': {'data': w_high_flat, 'unit': faam_w_unit_high, 'time': w_time_high, 'colour': 'coral', 'label' : 'W faam 32hz'},
    'uv_wind_direction': {'data': uv_wind_direction, 'unit': uv_wind_direction.data.units, 'time': uv_wind_direction['time'].values, 'colour': 'seagreen', 'label' : 'UV Direction'},
    'uv_wind_speed': {'data': uv_wind_speed, 'unit': uv_wind_speed.data.units, 'time': uv_wind_speed['time'].values, 'colour': 'navy', 'label' : 'UV Speed'},
    'N_d_high': {'data': cdp_conc, 'unit': cdp_conc.attrs.get('units'), 'time': cdp_conc['time'].values, 'colour': 'gray', 'label' : 'N_d high rate'},
    'alt_gin': {'data': alt, 'unit': alt_unit, 'time': alt['Time'].values, 'colour': 'tab:red', 'label' : 'Alt'},
    'mean_LWC_64hz': {'data': LWC_64hz, 'unit': LWC_unit_64hz, 'time': LWC_64hz['Time'].values, 'colour': 'purple', 'label' : 'LWC 64hz'},
    'tas_high': {'data': tas_flat, 'unit': tas_unit_high, 'time': tas_time_high, 'colour': 'darkorange', 'label' : 'TAS 32hz'}}

### functions ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
def get_stats(data):
    """Calculate basic statistics and return numerical magnitudes."""

    def get_value(value):

        # Compute Dask data if necessary
        if hasattr(value, "compute"):
            value = value.compute()

        # Extract Pint magnitude
        if hasattr(value, "magnitude"):
            value = value.magnitude

        # Handle xarray DataArray
        if hasattr(value, "values"):
            value = value.values

        # Handle NumPy arrays
        value = np.asarray(value).squeeze()

        return float(value)

    return {
        "min": get_value(data.min()),
        "max": get_value(data.max()),
        "mean": get_value(data.mean()),
        "sd": get_value(data.std()),
    }

# one thing to note is that need to look on the radar for the location of the pass 
# as may have just skirted around the edge of the core, not gone through the centre
def thermal_idx(sliced_dic_f, w_threshold = 0, lwc_threshold = 0):
    """
    get idx for themal for the cloud pass - longest continuous sample above threshold (default 0 m/s) in the vertical velocity data.
    args:
        sliced_dic_f: dictionary of sliced data for a cloud pass, containing vertical velocity data (w_faam_high)
        w_threshold: threshold value for vertical velocity (default 0 m/s)
        lwc_threshold: threshold value for LWC (default 0 g/m³)

    returns:
        tuple: (start_idx, end_idx) of the longest continuous segment above threshold"""

    w_faam_high = sliced_dic_f['w_faam_high']['data']
    w_faam_high_masked = w_faam_high.where(w_faam_high >= w_threshold)
    
    lwc_data = sliced_dic_f['mean_LWC_64hz']['data']
    lwc_data = lwc_data.rename({'Time': 'time'})
    lwc_aligned = lwc_data.interp(time=w_faam_high['time'])
    lwc_masked = lwc_aligned.where(lwc_aligned >= lwc_threshold)


    # Identify valid samples
    #valid = ~np.isnan(w_faam_high_masked)
    #valid = (~np.isnan(w_faam_high_masked) & ~np.isnan(lwc_masked))
    valid = w_faam_high_masked.notnull() & lwc_masked.notnull()

    # Find continuous valid segments
    changes = np.diff(np.r_[False, valid, False].astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # No samples above threshold
    if len(starts) == 0:
        return np.nan

    # Find longest segment
    lengths = ends - starts
    longest_idx = np.argmax(lengths)

    start = starts[longest_idx]
    end = ends[longest_idx]
    return (start, end)


def thermal_width(idx_tuple, sliced_dic_f):
    if idx_tuple is np.nan:
        return np.nan
    start, end = idx_tuple
    
    # Extract TAS over the longest valid segment
    tas_segment = sliced_dic_f['tas_high']['data'][start:end]

    # Calculate distance (metres)
    distance = np.sum(tas_segment) / 32 # as 32hz 
    return distance

def thermal_times(idx_tuple, sliced_dic_f):
    """Convert positional (start, end) on the w axis to start/end timestamps."""
    start, end = idx_tuple
    t = pd.to_datetime(sliced_dic_f['w_faam_high']['time'])
    return t[start], t[end - 1]   # end is exclusive, so use end-1


def slice_by_time(var_entry, t0, t1):
    data = var_entry['data']
    if hasattr(data, "sel"):
        time_coord = "time" if "time" in data.coords else "Time"
        return data.sel({time_coord: slice(t0, t1)})
    # non-xarray (e.g. pint array): use stored time array
    t = pd.to_datetime(var_entry['time'])
    return data[(t >= t0) & (t <= t1)]

def segment_stats(idx_tuple, sliced_dic_f, variable):
    nan_stats = {"min": np.nan, "max": np.nan, "mean": np.nan, "sd": np.nan}
    if isinstance(idx_tuple, float) and np.isnan(idx_tuple):
        return nan_stats

    t0, t1 = thermal_times(idx_tuple, sliced_dic_f)
    segment = slice_by_time(sliced_dic_f[variable], t0, t1)

    if segment.size == 0:
        return nan_stats
    return get_stats(segment)

def concatenate_samples(data_list):
    """Concatenate samples from multiple cloud passes."""

    if len(data_list) == 0:
        return None

    if hasattr(data_list[0], "dims"):
        # Preserve xarray structure and units
        time_dim = data_list[0].dims[0]
        return xr.concat(data_list, dim=time_dim)

    # For NumPy arrays
    return np.concatenate(data_list)
### functions ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~


lwc_thres = 1
w_thresholds = [0, 2, 5, 7, 10]

thermal_variables = {
    'lwc': 'mean_LWC_64hz',
    'droplet_concentration': 'N_d_high',
    'w': 'w_faam_high',
    'uv': 'uv_wind_speed',}

start_times_buf, end_times_buf = cloud_pass_time_v2(mo_path_64hz, sec_buffer=5)

results = []

for w_thres in w_thresholds:
    print(f"\nProcessing w threshold: {w_thres} m/s")

    # Store all samples across the day for this threshold
    daily_thermal_data = {var: [] for var in thermal_variables}

    for a_time_idx, (dt_start, dt_end) in enumerate(zip(start_times_buf, end_times_buf)):
        print(f"Processing cloud pass {a_time_idx + 1}")

        # Slice all variables
        sliced_dic = {}
        for var, d in combined_dic.items():
            time = pd.to_datetime(d["time"])
            mask = (time >= dt_start) & (time <= dt_end)

            if hasattr(d["data"], "sel"):
                time_coord = ("time" if "time" in d["data"].coords else "Time")
                sliced_data = d["data"].sel({time_coord: slice(dt_start, dt_end)})
            else:
                sliced_data = d["data"][mask]

            sliced_dic[var] = {
                **d,
                "data": sliced_data,
                "time": time[mask],}
            
        # ----------------------------------
        # Identify thermal segment
        idx_tuple = thermal_idx(
            sliced_dic,
            w_threshold=w_thres,
            lwc_threshold=lwc_thres)

        # ----------------------------------
        # Initialise pass results
        # ----------------------------------
        row = {
            "w_threshold": w_thres,
            "lwc_threshold": lwc_thres,
            "pass_number": a_time_idx + 1,
            "start_time": pd.to_datetime(dt_start).strftime("%H:%M:%S"),
            "end_time": pd.to_datetime(dt_end).strftime("%H:%M:%S"),}

        # ----------------------------------
        # Check if thermal exists
        # ----------------------------------
        if (isinstance(idx_tuple, float) and np.isnan(idx_tuple)):
            print("No thermal identified in this pass.")
            for var in thermal_variables:
                for stat in ["min", "max", "mean", "sd"]:
                    row[f"thermal_{var}_{stat}"] = np.nan
            width = np.nan
            mean_altitude = np.nan

        else:
            start, end = idx_tuple
            print(var, len(sliced_dic[var]['data']), "start/end:", start, end)


            # ----------------------------------
            # Extract thermal-only samples
            # ----------------------------------
            t0, t1 = thermal_times(idx_tuple, sliced_dic)

            for output_var, data_key in thermal_variables.items():
                thermal_data = slice_by_time(sliced_dic[data_key], t0, t1)
                daily_thermal_data[output_var].append(thermal_data)

            # ----------------------------------
            # Thermal statistics
            # ----------------------------------
            for output_var, data_key in thermal_variables.items():

                stats = segment_stats(
                    idx_tuple,
                    sliced_dic,
                    data_key)

                for stat in ["min", "max", "mean", "sd"]:
                    row[f"thermal_{output_var}_{stat}"] = stats[stat]

            # ----------------------------------
            # Thermal width
            # ----------------------------------
            width = thermal_width(idx_tuple, sliced_dic)

            # ----------------------------------
            # Mean thermal altitude
            # ----------------------------------
            altitude_data = slice_by_time(sliced_dic['alt_gin'], t0, t1)
            mean_altitude = altitude_data.mean()
            if hasattr(mean_altitude, "compute"):
                mean_altitude = mean_altitude.compute()
            if hasattr(mean_altitude, "item"):
                mean_altitude = mean_altitude.item()

        # ----------------------------------
        # Store individual pass results
        # ----------------------------------
        row["thermal_width_m"] = (float(width) if not np.isnan(width) else np.nan)
        row["mean_altitude_m"] = float(mean_altitude)
        results.append(row)

    # ==================================
    # Daily thermal statistics
    # ==================================

    daily_row = {
        "w_threshold": w_thres,
        "lwc_threshold": lwc_thres,
        "pass_number": "Daily average",
        "start_time": np.nan,
        "end_time": np.nan,}

    # ----------------------------------
    # Daily statistics for each variable
    # ----------------------------------
    for var in thermal_variables:

        if daily_thermal_data[var]:

            daily_data = concatenate_samples(
                daily_thermal_data[var]
            )

            daily_stats = get_stats(daily_data)

        else:

            daily_stats = {
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "sd": np.nan,
            }

        for stat in ["min", "max", "mean", "sd"]:
            daily_row[f"thermal_{var}_{stat}"] = daily_stats[stat]

    # Thermal width is not calculated as a daily average
    daily_row["thermal_width_m"] = np.nan
    daily_row["mean_altitude_m"] = np.nan
    results.append(daily_row)


# ==================================
# Create and export CSV
# ==================================
df_results = pd.DataFrame(results)
df_results.to_csv(f"{save_pth}{date}_thermal_summary.csv", index=False)

