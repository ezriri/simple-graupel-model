## need to just make a csv for each day - decide if pass goes through main core / write notes on it

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
from tracemalloc import start

sys.path.append('/home/users/esree/dcmex_instruments/faam_data/')
# loads of homemade functions <3 <3
from faam_data_functions import faam_mo_files, time_adjust, get_nc_variable, mask_data, flatten_20hz_variable, mo_mean, datetime_to_sec, cloud_pass_time
from faam_data_functions_v2 import faam_mo_files_v2, make_cdp_psd_v2, make_cdp_conc, cloud_pass_time_v2


date = '220730'

save_pth = '/home/users/esree/simple-graupel-model/tune_to_DCMEX/pass_overview_csvs/'


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

faam_w_unit_high, faam_w_longname_high, faam_w_high = get_nc_variable(faam_pth_high, 'W_C')
faam_w_high = faam_w_high.rename({'Time': 'time'})
w_high_flat, w_time_high = flatten_20hz_variable(faam_w_high)

tas_unit_high, tas_longname_high, tas_high = get_nc_variable(faam_pth_high, 'TAS_RVSM')
tas_high = tas_high.rename({'Time': 'time'})
tas_flat, tas_time_high = flatten_20hz_variable(tas_high)

LWC_unit_64hz, LWC_longname_64hz, LWC_64hz = mo_mean(mo_path_64hz, ('LWC1','LWC2'))

combined_dic = {
    'w_faam_high': {'data': w_high_flat, 'unit': faam_w_unit_high, 'time': w_time_high, 'colour': 'coral', 'label' : 'W faam 32hz'},
    'N_d_high': {'data': cdp_conc, 'unit': cdp_conc.attrs.get('units'), 'time': cdp_conc['time'].values, 'colour': 'gray', 'label' : 'N_d high rate'},
    'alt_gin': {'data': alt, 'unit': alt_unit, 'time': alt['Time'].values, 'colour': 'tab:red', 'label' : 'Alt'},
    'mean_LWC_64hz': {'data': LWC_64hz, 'unit': LWC_unit_64hz, 'time': LWC_64hz['Time'].values, 'colour': 'purple', 'label' : 'LWC 64hz'},
    'tas_high': {'data': tas_flat, 'unit': tas_unit_high, 'time': tas_time_high, 'colour': 'darkorange', 'label' : 'TAS 32hz'}}

start_times_buf, end_times_buf = cloud_pass_time_v2(mo_path_64hz, sec_buffer=5)

# so for each cloud pass - want mean, min, max, SD of W / cloud drop conc / LWC 
# + mean alt + total distance


def get_segment(key, t0, t1):
    """Return the data for combined_dic[key] between t0 and t1 (inclusive)."""
    d = combined_dic[key]
    t = pd.to_datetime(np.asarray(d['time'])).values            # datetime64[ns]
    x = np.asarray(d['data'], dtype=float).squeeze()
    mask = (t >= t0) & (t <= t1)
    return t[mask], x[mask]

# variables to get mean / min / max / SD for
stat_vars = {'w_faam_high': 'W', 'N_d_high': 'Nd', 'mean_LWC_64hz': 'LWC'}

rows = []
for i, (t0, t1) in enumerate(zip(pd.to_datetime(start_times_buf),
                                 pd.to_datetime(end_times_buf))):
    t0, t1 = np.datetime64(t0, 'ns'), np.datetime64(t1, 'ns')
    row = {'pass': i, 'start': t0, 'end': t1,
           'duration_s': (t1 - t0) / np.timedelta64(1, 's')}

    # mean, min, max, SD of W / drop conc / LWC
    for key, name in stat_vars.items():
        _, x = get_segment(key, t0, t1)
        unit = combined_dic[key]['unit']
        row[f'{name}_mean ({unit})'] = np.nanmean(x) if x.size else np.nan
        row[f'{name}_min ({unit})']  = np.nanmin(x)  if x.size else np.nan
        row[f'{name}_max ({unit})']  = np.nanmax(x)  if x.size else np.nan
        row[f'{name}_std ({unit})']  = np.nanstd(x)  if x.size else np.nan

    # mean altitude
    _, alt_seg = get_segment('alt_gin', t0, t1)
    row[f"alt_mean ({combined_dic['alt_gin']['unit']})"] = np.nanmean(alt_seg) if alt_seg.size else np.nan

    # total distance = mean TAS x duration (assumes TAS in m/s -> km)
    _, tas_seg = get_segment('tas_high', t0, t1)
    row['distance_km'] = np.nanmean(tas_seg) * row['duration_s'] / 1000 if tas_seg.size else np.nan

    rows.append(row)

cloud_pass_stats = pd.DataFrame(rows)
cloud_pass_stats.to_csv(f'{save_pth}{date}_cloud_pass.csv', index=False)

