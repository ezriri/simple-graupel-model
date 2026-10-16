## something going wrong with the location of the particle - gif to look at this

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


save_loc_v3_2 = "/gws/ssde/j25a/dcmex/users/ezriab/graupel_model_output/model_output_v3_2/" # location to save output files
save_name = 'blyth_latham'
ncfile = xr.open_dataset(f"{save_loc_v3_2}{save_name}.nc", decode_times=False)
ncfile_envi = xr.open_dataset(f"{save_loc_v3_2}{save_name}.nc", group="environment", engine="netcdf4",decode_times=False)
ncfile_ice = xr.open_dataset(f"{save_loc_v3_2}{save_name}.nc", group="ice_particles", engine="netcdf4",decode_times=False)
ncfile_splinter = xr.open_dataset(f"{save_loc_v3_2}{save_name}.nc", group="splinter_particles", engine="netcdf4",decode_times=False)

# also pull out the variables put into the namelist file
namelist_file = '/home/users/esree/simple-graupel-model/namelist_new_thermal_200min.txt'#ncfile.attrs["namelist_file"]
with open(namelist_file, "r") as f:
    namelist_contents = f.read()
namelist = {} ## all the namelist inputs are stored in this dictionary
exec(namelist_contents, {"np": np}, namelist)

## pull out relevent functions
sys.path.append('/home/users/esree/simple-graupel-model/')
# i have moved all the smaller functions to a seperate file - so can be accessed when plotting too
from graupel_model_sub_functions import trev, ptz

gif_save_name = f'particle_5_alt_vs_horizontal'

# make a gif of this through time
specif_save_folder = '/home/users/esree/simple-graupel-model/'
ncfile_mins = ncfile['time'].values/60
zbase = ptz(namelist['pbase']) # height of cloud base (m)
thtop_1 = ncfile_envi['thermal_top_1']
cldtop = ncfile_envi['cloud_top']
same = np.isclose(thtop_1.values, cldtop.values, equal_nan=False)
zct = thtop_1 - namelist['ZCT']
zct_matched = zct.where(same)


with tempfile.TemporaryDirectory(dir=specif_save_folder, prefix="tmp_") as tmp_dir:
    frame_files = []
    for t in range(1000):#just first 1000 time steps ncfile_ice.sizes['time'][:1000]):
        frame_path = os.path.join(tmp_dir, f"frame_{t:04d}.png") # unique filenames
        t0_ice = ncfile_ice.isel(time=t)
        value = ncfile['time'].isel(time=t)
        
        fig = plt.figure(figsize=(8, 8))

        for s in range(5):
            p0_ice = t0_ice.isel(particle=s)
            plt.scatter(p0_ice['horizontal_position'], p0_ice['altitude']/1000)

        ## plot all thermals / debris base
        plt.axhline(y=zct_matched.isel(time=t), color='purple', linestyle=(0, (1, 1)), label='zct')
        plt.axhline(y= ncfile_envi['cloud_top'].isel(time=t), color='blue', linestyle='solid', label='cldtop', alpha=0.25)
        plt.axhline(y= ncfile_envi['debris_base'].isel(time=t), color='grey', linestyle='--', label='debris_base', alpha=0.25)
        plt.axhline(y= ncfile_envi['thermal_top_1'].isel(time=t), color='red', linestyle=(0, (1, 1)), label='thtop1', alpha=0.5)
        plt.axhline(y= ncfile_envi['thermal_base_1'].isel(time=t), color='red', linestyle=(0, (1, 1)), label='thbase1', alpha=0.5)
        plt.axhline(y= ncfile_envi['thermal_top_2'].isel(time=t), color='orange', linestyle=(0, (1, 1)), label='thtop2', alpha=0.5)
        plt.axhline(y= ncfile_envi['thermal_base_2'].isel(time=t), color='orange', linestyle=(0, (1, 1)), label='thbase2', alpha=0.5)

        # constants 
        plt.axvline(x=namelist['TWID'], color='black', linestyle='--', label='TWID')
        plt.axhline(y=zbase/1000, color='black', linestyle='--', label='cloud_base',alpha=0.5)
        plt.axhline(y=namelist['CTOP'], color='black', linestyle='-.', label='CTOP', alpha=0.5)

        plt.xlabel("horizontal position (km)")
        plt.ylabel("altitude (km)")
        plt.ylim(0, 9)
        plt.xlim(-0.5, 3.5)
        #plt.title(f"Time: {value.values:.2f} s")
        plt.title(f"Time: {value.values/60:.2f} min")

        plt.legend(loc='lower right', fontsize=8)
        plt.savefig(frame_path, bbox_inches='tight')

        plt.close()
        frame_files.append(frame_path) # all frames path

    frames = [imageio.v2.imread(f) for f in frame_files]


    gif_path = os.path.join(specif_save_folder, f'{gif_save_name}.gif')
    imageio.mimsave(gif_path, frames, fps=5)
        