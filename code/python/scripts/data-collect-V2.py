#!/usr/bin/env python
# coding: utf-8

# # File to be run in batch jobs. Collects data using function in utils, then saves it to be read by 001

# In[1]:


import sys
sys.path.append('/home/548/cd3022/aus-historical-solar-droughts/code/python/scripts')
import utils_V2
import os

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr


# In[2]:


# start_date = sys.argv[1]
# end_date = sys.argv[2]

start_date = '1-1-2022'
end_date = start_date


# In[3]:


# REZ mask
mask_file = "/home/548/cd3022/aus-historical-solar-droughts/data/boundary_files/REZ_mask.npz"
loaded_mask = np.load(mask_file)
mask = loaded_mask["mask"]


# In[4]:


# datetime object for finding files
dir_dt = datetime.strptime(start_date, "%d-%m-%Y")
end_dt = datetime.strptime(end_date, "%d-%m-%Y")

# num files for the loop, and for making output file shape
num_days = (end_dt - dir_dt).days+1
num_files = 103 * num_days # 103 files in each directory

# same lat/lon for all data
lat_len = 1726
lat_vals = np.linspace(-44.5, -10.0, lat_len).astype(np.float32)
lon_len = 2214
lon_vals = np.linspace(112, 156.26, lon_len).astype(np.float32)
lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
lat_grid = lat_grid[mask]
lon_grid = lon_grid[mask]

times = [] # collect times while iterating and add at the end

# make output array with correct shape first, for efficiency
output_data = xr.DataArray(
    np.full((num_files, lat_len, lon_len), np.nan),
    dims=('time', 'latitude', 'longitude'),
    coords={'latitude': lat_vals, 'longitude': lon_vals},
    name='capacity_factor'
)


# In[6]:


# loop through all files in date range
for i in range(num_files):
    dataset = utils_V2.get_irradiance_dataset(dir_dt=dir_dt)

    if dataset is not None: # only look at data if file exists

        # irradiance data for capacity factor calculations
        REZ_ghi = dataset['surface_global_irradiance'].squeeze(dim="time").values[mask]
        REZ_dni = dataset['direct_normal_irradiance'].squeeze(dim="time").values[mask]
        REZ_dhi = dataset['surface_diffuse_irradiance'].squeeze(dim="time").values[mask]
        dataset.close()
        
        # time, lat, and lon for solar angles needed for capacity factor calculations
        utc_dt = dir_dt - timedelta(hours=5, minutes=30)
        # time = pd.Timestamp(f'{utc_dt.year}-{utc_dt.month}-{utc_dt.day} {utc_dt.hour}:{utc_dt.minute}:00', tz='UTC') # for solar calcs
        time = pd.Timestamp(dir_dt, tz='UTC')
        
        # calculate capacity factors using pvlib
        capacity_factors = utils_V2.solar_cf(
            pv_model = 'Canadian_Solar_CS5P_220M___2009_',
            inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
            ghi=REZ_ghi,
            dni=REZ_dni,
            dhi=REZ_dhi,
            time=time,
            lat_grid=lat_grid,
            lon_grid=lon_grid
        )  
        
        # get data back into shape
        cf_reshaped = np.full(mask.shape, np.nan)
        cf_reshaped[mask] = capacity_factors
        
        output_data[i, :, :] = cf_reshaped
        
    
    times.append(np.datetime64(dir_dt))  # for output file
    dir_dt = dir_dt + timedelta(minutes=10)    
    
output_data = output_data.assign_coords(time=times) # add time values


# In[7]:


output_file = f'/g/data/er8/users/cd3022/solar_drought/hourly_capacity_factors/{start_date}___{end_date}.nc'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
output_data.to_netcdf(output_file)


# In[ ]:


if __name__ == '__main__':
    
    NOTEBOOK_PATH="/home/548/cd3022/aus-historical-solar-droughts/code/python/notebooks/data-collect-V2.ipynb"
    SCRIPT_PATH="/home/548/cd3022/aus-historical-solar-droughts/code/python/scripts/data-collect-V2"
    
    get_ipython().system('jupyter nbconvert --to script {NOTEBOOK_PATH} --output {SCRIPT_PATH}')

