#!/usr/bin/env python
# coding: utf-8

# # Time series for aggregated area

# Read himawari data, apply the solar PV model, and save the mean value for each time step.
# Hence, output is a 1D time series of PV generation

# In[1]:


import sys
sys.path.append('/home/548/cd3022/aus-historical-solar-droughts/code/python/scripts')
# sys.path.append('/home/548/pag548/code/aus-historical-solar-droughts/code/python/scripts')
import utils
import os

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import pathlib


# In[2]:


#start_date = sys.argv[1]
#end_date = sys.argv[2]

start_date = '1-2-2022'
end_date = '2-2-2022'


# In[3]:


# REZ mask
mask_file = "/home/548/cd3022/aus-historical-solar-droughts/data/boundary_files/REZ_mask.npz"
loaded_mask = np.load(mask_file)
mask = loaded_mask["mask"]


# In[4]:


start_dt = datetime.strptime(start_date, "%d-%m-%Y")
end_dt = datetime.strptime(end_date, "%d-%m-%Y")

# Generate a list of dates
date_range = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]


# In[ ]:


# list for time series
NEM_performance = []
# loop through all files in date range
for dir_dt in date_range:

    dataset = utils.get_irradiance_day(resolution='p1s', dir_dt=dir_dt)

    # apply REZ mask to lat/lon coordinates
    mask_da = xr.DataArray(mask, coords={"latitude": dataset.latitude, "longitude": dataset.longitude}, dims=["latitude", "longitude"])
    masked_ds = dataset.where(mask_da, drop=True)

    # get irradiance data, ensuring to flatten and remove all unnecessary nan values
    ghi = masked_ds.surface_global_irradiance.values.ravel()
    dni = masked_ds.direct_normal_irradiance.values.ravel()
    dhi = masked_ds.surface_diffuse_irradiance.values.ravel()
    nan_mask = np.isnan(ghi) # same for all vars
    ghi_clean = ghi[~nan_mask]
    dni_clean = dni[~nan_mask]
    dhi_clean = dhi[~nan_mask]

    # get correct time and coordinate data, so that it matches up with the remaining irradiance values
    lat_1d = masked_ds.latitude.values
    lon_1d = masked_ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")
    lat_grid_1d = lat_grid.ravel()
    lon_grid_1d = lon_grid.ravel()
    lat_1d_expanded = np.tile(lat_grid_1d, dataset.sizes["time"])  # Tile lat for all times
    lon_1d_expanded = np.tile(lon_grid_1d, dataset.sizes["time"])  # Tile lon for all times
    time_1d = np.repeat(masked_ds.time.values, len(lat_grid_1d))  # Repeat time for all lat/lon
    lat_1d_expanded_clean = lat_1d_expanded[~nan_mask]
    lon_1d_expanded_clean = lon_1d_expanded[~nan_mask]
    time_1d_clean = time_1d[~nan_mask]

    dataset.close()
        
    # calculate capacity factors using pvlib
    # the function defined in utils_V2 is essentially the same as the workflow in pv-output-tilting.ipynb
    actual_ideal_ratio = utils.tilting_panel_pr(
        pv_model = 'Canadian_Solar_CS5P_220M___2009_',
        inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        ghi=ghi_clean,
        dni=dni_clean,
        dhi=dhi_clean,
        time=time_1d_clean,
        lat=lat_1d_expanded_clean,
        lon=lon_1d_expanded_clean
    )  

    # template to refit data to
    mask_template = masked_ds.surface_global_irradiance
    
    # Now need to get data back in line with coordinates
    # fill cf array with nan values so it can fit back into lat/lon coords
    filled = np.empty_like(ghi)
    # nan values outside the data
    filled[nan_mask] = np.nan
    # add the data to the same mask the input irradiance data was taken from
    filled[~nan_mask] = actual_ideal_ratio
    # convert data back into 3D xarray
    reshaped = filled.reshape(mask_template.shape)
    ratio_da = xr.DataArray(reshaped, coords=mask_template.coords, dims=mask_template.dims)

    # get mean for each time slice and add it to the list
    mean_daily = ratio_da.mean(dim=["latitude", "longitude"], skipna=True)
    NEM_performance.append(mean_daily)


# In[ ]:


NEM_performance_timeseries = xr.concat(NEM_performance, dim='time')


# In[ ]:


file_path = '/g/data/er8/users/cd3022/solar_drought/REZ_tilting/ideal_ratio/NEM_timeseries'
os.makedirs(file_path, exist_ok=True)
NEM_performance_timeseries.to_netcdf(f'{file_path}/{start_date}___{end_date}.nc')


# In[ ]:


if __name__ == '__main__':
    
    NOTEBOOK_PATH="/home/548/cd3022/aus-historical-solar-droughts/code/python/notebooks/NEM-droughts.ipynb"
    SCRIPT_PATH="/home/548/cd3022/aus-historical-solar-droughts/code/python/scripts/NEM-droughts"
    
    get_ipython().system('jupyter nbconvert --to script {NOTEBOOK_PATH} --output {SCRIPT_PATH}')

