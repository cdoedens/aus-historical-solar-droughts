#!/usr/bin/env python
# coding: utf-8

# In[2]:


from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
from shapely.vectorized import contains
import subprocess
import sys
import os


# In[2]:


# TO DO: rename variables so that datetime objects and strings are recognisable

def get_irradiance_dataset(dir_dt):
    utc_dt = dir_dt - timedelta(hours=5, minutes=30)
    file_dt = utc_dt.strftime("%Y%m%d%H%M")
    filename='IDE00326.'+file_dt+'.nc'

    if utc_dt < datetime.strptime('2019-03-31', '%Y-%m-%d'):
        file = 'v1.0'
    else:
        file = 'v1.1'
    
    dirin=f'/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/{file}/'+f"{dir_dt.year:04}"+'/'+f"{dir_dt.month:02}"+'/'+f"{dir_dt.day:02}"+"/"
    try:
        dataset = Dataset(dirin+filename)
    except FileNotFoundError:
        print(f"File not found: {dirin + filename}.")
        dataset = None
    return dataset


# In[3]:


def get_coords(area_bounds=None):
    
    # same lat/lon in all files, pick any to get coordinate data
    dataset = get_irradiance_dataset(datetime.strptime('1-1-2023', "%d-%m-%Y"))
    
    latitudes=dataset.variables['latitude'][:]
    longitudes=dataset.variables['longitude'][:]
    if area_bounds == None:
        return latitudes, longitudes
    lat_indices=np.where((dataset.variables['latitude'][:] >= area_bounds["lat_min"]) & (dataset.variables['latitude'][:] <= area_bounds["lat_max"]))[0]
    lon_indices=np.where((dataset.variables['longitude'][:] >= area_bounds["lon_min"]) & (dataset.variables['longitude'][:] <= area_bounds["lon_max"]))[0]
    
    return latitudes, longitudes, lat_indices, lon_indices


# In[9]:


def read_irradiance(start_date, end_date, area_bounds=None, mask=None):

    '''
    INPUTS
    num_days: number of days of data to inspect

    start_df: first date to inspect

    area_bounds: dictionary containing min and max lat/lon coordinates
    '''

    # datetime object for finding files
    dir_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")
    num_days = (end_dt - dir_dt).days+1

    # coordinates to inspect
    if area_bounds:
        latitudes, longitudes, lat_indices, lon_indices = get_coords(area_bounds=area_bounds)

    # dict for DF
    rad_data = {'date':[], 'daily_mean':[]}

    # improvement for later, remove hard coded value for file_count
    file_count = 103 # number of files in each directory
    for i in range(file_count*num_days):
        dataset = get_irradiance_dataset(dir_dt=dir_dt)

        # first file of the day, add the date of that day to the dict
        if dir_dt.strftime('%H%M') == '0000':
            rad_data['date'].append(f'{dir_dt.day}-{dir_dt.month}-{dir_dt.year}')
            daily_data = []

        if dataset is not None:
            # only look at data if file exists
            if area_bounds is not None:
                # Extract and squeeze irradiance data for the area
                irradiance = np.squeeze(dataset.variables['surface_global_irradiance'][:, lat_indices, :][:, :, lon_indices])
                daily_data.append(irradiance)
                
            elif mask is not None:
                irradiance = np.squeeze(dataset.variables['surface_global_irradiance'][:,:,:][:,:,:])
                masked_irradiance = np.where(mask, irradiance, np.nan)
                daily_data.append(masked_irradiance)
            else:
                irradiance = np.squeeze(dataset.variables['surface_global_irradiance'][:,:,:][:,:,:])
                daily_data.append(irradiance)
                
            dataset.close()

        if dir_dt.strftime('%H%M') == '1700': # last file of day
            dir_dt = dir_dt + timedelta(hours=7)
            daily_mean = np.ma.mean(np.ma.stack(daily_data), axis=0)
            rad_data['daily_mean'].append(daily_mean)
            
        else:
            dir_dt = dir_dt + timedelta(minutes=10)
    return pd.DataFrame(rad_data).set_index('date')


# In[5]:


def plot_area(data, fig_name, area_bounds=None, vmax=None):

    if area_bounds is not None:
        latitudes, longitudes, lat_indices, lon_indices = get_coords(area_bounds=area_bounds)
        lon, lat = np.meshgrid(longitudes[lon_indices], latitudes[lat_indices])
    else:
        lat, lon = get_coords()
    
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.coastlines()
    
    # Plot the 2D masked array using pcolormesh with Cartopy CRS
    mesh = ax.pcolormesh(lon, lat, data, cmap='viridis', vmin=0, vmax=vmax, shading='auto', transform=ccrs.PlateCarree())
    
    # Add a colorbar
    plt.colorbar(mesh, ax=ax, label='Num. Days / season', shrink=0.5)
    
    # Set plot title and labels
    ax.set_title(fig_name)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(f'/home/548/cd3022/aus-historical-solar-droughts/figs/heatmaps/{fig_name}.png')
    plt.show()


# In[6]:


def get_region_mask(shape_file, regions):

    # https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
    gccsa_file = shape_file
    gccsa = gpd.read_file(gccsa_file, encoding='utf-8')
    gccsa = gccsa.to_crs("EPSG:4326")
    all_regions = gccsa[gccsa['GCC_CODE21'].isin(regions)]

    # lat and lon values from datasets
    latitudes, longitudes = get_coords()
    # Create a meshgrid of latitudes and longitudes
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    # Flatten the grids into 1D arrays
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    
    # Prepare the mask for all regions at once
    region_masks = np.array([
        contains(gccsa[gccsa['GCC_CODE21'] == city].unary_union, lon_flat, lat_flat).reshape(lon_grid.shape)
        for city in regions
    ])
    return region_masks


# In[4]:


def find_droughts(data, definition, threshold):
    '''
    data: DataFrame contain mean daily values as 2D masked arrays

    definition: how droughts are being defined

    threshold: cut-off for identifying droughts
    '''

    # Find baseline conditions (depending on definition) for national and regional data
    if definition == 'mean':
        total = np.ma.mean(np.ma.stack(data), axis=0)
    elif definition == 'max':
        total = np.ma.max(np.ma.stack(data), axis=0)
    else:
        raise ValueError(f"Invalid definition '{definition}'. Please provide 'mean' or 'max'.")
        
    # total.mask = total < 10 # remove edge values with small mean that have significantly more drought days
        
    # Produce list of 2D arrays with bool values to indicate if data point is a drought
    is_drought_day = [
        np.ma.masked_array(
            dm < (total * threshold),
            mask = dm < 10 
        )
        for dm in data
    ]
  
    # count the number of droughts occuring at the same time
    coincident_droughts = [np.sum(droughts) for droughts in is_drought_day]
    
    return pd.DataFrame({'is_drought_day': is_drought_day, 'coincident_droughts': coincident_droughts})


# In[ ]:


def regional_drought_lengths(df, regions):

# Find consecutive droughts days, count length of drought and number of unique droughts

# makes a dict of regional dicts
# each regional dict contains a dict for how many times that region experienced a drought of different legnths

# Generate 2D array same as "is_drought_day", but cumulative days increment by 1
# Create an array for cumulative drought counts
    drought_stack = np.stack(df['is_drought_day'].values)
    cumulative_droughts = np.zeros_like(drought_stack, dtype=int)
    
    
    # Iterate through the time axis while resetting counts on drought breaks
    cumulative_droughts[0] = drought_stack[0]  # Initialize the first time step
    for t in range(1, drought_stack.shape[0]):
        # Increment drought count where drought continues
        cumulative_droughts[t] = (
            (cumulative_droughts[t - 1] + 1) * drought_stack[t]
        )

    # Make dict of drought lengths, and count how many droughts of each length there are
    regional_drought_lengths = {}
    for i, region in enumerate(regions):
        # Make dict of possible drought lengths up to 7 days
        drought_lengths = {length: 0 for length in [str(i) for i in range(1, 8)]}

        # count droughts
        for j in range(len(cumulative_droughts[:, i])-1):
            if cumulative_droughts[j, i] == 0:
                continue
            if cumulative_droughts[j+1, i] != 0:
                continue
            length = cumulative_droughts[j, i]
            drought_lengths[f'{length}'] += 1
            
        if cumulative_droughts[-1, i] != 0:
            length = cumulative_droughts[-1, i]
            if f'{length}' not in drought_lengths:
                drought_lengths[f'{length}'] = 1
        regional_drought_lengths[region] = drought_lengths
    return regional_drought_lengths, cumulative_droughts


# In[1]:


# Convert to .py file to be imported by other modules
if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script "/home/548/cd3022/aus-historical-solar-droughts/code/python/notebooks/utils.ipynb" --output "/home/548/cd3022/aus-historical-solar-droughts/code/python/scripts/utils"')
    print('name == main')

