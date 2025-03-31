#!/usr/bin/env python
# coding: utf-8

# In[1]:


from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.vectorized import contains


# In[2]:


# Hard coded variables
year = 2022
month = 5
day = 31
hour = 18
minute = 30
utc_dt = datetime(year, month, day, hour, minute)
dir_dt = utc_dt + timedelta(hours=5, minutes=30)
num_days = 90
maxv=800

regions = [
    '1GSYD',
    '2GMEL',
    '3GBRI',
    '4GADE',
    '5GPER',
    '6GHOB',
    '7GDAR',
    '8ACTE'
]


# In[3]:


# Load data

# https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
gccsa_file = '/home/548/cd3022/data/boundary_files/GCCSA/GCCSA_2021_AUST_GDA2020.shp'
gccsa = gpd.read_file(gccsa_file, encoding='utf-8')
gccsa = gccsa.to_crs("EPSG:4326")

all_regions = gccsa[gccsa['GCC_CODE21'].isin(regions)]

# Find number of files in each directory
dirin='/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest/'+f"{dir_dt.year:04}"+'/'+f"{dir_dt.month:02}"+'/'+f"{dir_dt.day:02}"+"/"
file_count = len([f for f in os.listdir(dirin) if os.path.isfile(os.path.join(dirin, f))])

# Find lat/lon indices to use on all datasets
file_dt = utc_dt.strftime("%Y%m%d%H%M")
filename='IDE00326.'+file_dt+'.nc'
dataset = Dataset(dirin+filename)

latitudes = dataset.variables['latitude'][:]
longitudes = dataset.variables['longitude'][:]
irradiance = np.squeeze(dataset.variables['surface_global_irradiance'][:,:,:])


# In[4]:


# Prepare masking variables

# Create a meshgrid of latitudes and longitudes
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Flatten the grids into 1D arrays
lon_flat = lon_grid.ravel()
lat_flat = lat_grid.ravel()

# Vectorized check: Flattened grid of points
points = np.array([lon_flat, lat_flat]).T

# Check which points are inside the polygon(s)
mask_flat = contains(all_regions.unary_union, lon_flat, lat_flat)

# Reshape back to the grid shape
mask = mask_flat.reshape(lon_grid.shape)

# Apply the mask
masked_data = np.ma.masked_where(~mask, irradiance)

# Prepare the mask for all regions at once
region_masks = np.array([
    contains(gccsa[gccsa['GCC_CODE21'] == city].unary_union, lon_flat, lat_flat).reshape(lon_grid.shape)
    for city in regions
])


# In[ ]:


# dict for DF
rad_data = {'date':[], 'daily_mean':[]}

# Loop over all the files, for the specified number of days
for i in range(file_count*num_days):
    file_dt = utc_dt.strftime("%Y%m%d%H%M")
    filename='IDE00326.'+file_dt+'.nc'
    dir_dt = utc_dt + timedelta(hours=5, minutes=30)

    # first file of the day, add the date of that day to the dict
    if file_dt[-4:] == '1830':
        rad_data['date'].append(f'{dir_dt.day}/{dir_dt.month}/{dir_dt.year}')
        daily_data = []
        
    dirin='/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest/'+f"{dir_dt.year:04}"+'/'+f"{dir_dt.month:02}"+'/'+f"{dir_dt.day:02}"+"/"
    try: # some data files missing, so move onto next time step if file cannot be found
        dataset = Dataset(dirin+filename)
    except FileNotFoundError:
        print(f"File not found: {dirin + filename}, skipping to next file.")
        
        if file_dt[-4:] == '1130': # last file of day
            utc_dt = utc_dt + timedelta(hours=7)
            daily_mean = np.ma.mean(np.ma.stack(daily_data), axis=0)
            rad_data['daily_mean'].append(daily_mean)
        else:
            utc_dt = utc_dt + timedelta(minutes=10)
                                        
        continue

    # Extract and squeeze irradiance data for the area
    irradiance = np.squeeze(dataset.variables['surface_global_irradiance'][:,:,:][:,:,:])
    
    # Calculate the mean irradiance for each region
    masked_data = np.ma.masked_array(np.repeat(irradiance[None, ...], len(regions), axis=0), mask=~region_masks)
    region_means = masked_data.mean(axis=(1, 2))
    # # Create mean arrays covering whole region
    # mean_arrays = np.ma.masked_array(
    #     np.broadcast_to(region_means[:, None, None], region_masks.shape), mask=~region_masks
    # )


    daily_data.append(region_means) # region_means is just a single value for each region, no spatial array
    
    if file_dt[-4:] == '1130':
        # END OF DAY
        utc_dt = utc_dt + timedelta(hours=7)
        daily_mean = np.ma.mean(np.ma.stack(daily_data), axis=0)
        rad_data['daily_mean'].append(daily_mean)
        
    else:
        utc_dt = utc_dt + timedelta(minutes=10)
    dataset.close()


# In[ ]:


rad_df = pd.DataFrame(rad_data)
rad_df.set_index('date', inplace=True)

mean_stack = np.ma.stack(rad_df['daily_mean'].values)
total_mean = np.ma.mean(mean_stack, axis=0)

# Find drought days using the definition: daily average > 50% of mean day
is_drought_day = [
    np.ma.masked_array(
        dm < total_mean / 2,  # Condition for drought
        mask=np.ma.getmask(dm) | np.ma.getmask(total_mean)  # Combine masks
    )
    for dm in rad_df['daily_mean']
]

rad_df = pd.concat([rad_df, pd.DataFrame({'is_drought_day': is_drought_day}, index = rad_df.index)], axis=1)


# In[ ]:


# Generate 2D array same as "is_drought_day", but cumulative days increment by 1

drought_stack = np.stack(is_drought_day)

# Create an array for cumulative drought counts
cumulative_droughts = np.zeros_like(drought_stack, dtype=int)

# Iterate through the time axis while resetting counts on drought breaks
cumulative_droughts[0] = drought_stack[0]  # Initialize the first time step
for t in range(1, drought_stack.shape[0]):
    # Increment drought count where drought continues
    cumulative_droughts[t] = (
        (cumulative_droughts[t - 1] + 1) * drought_stack[t]
    )


# In[ ]:


# makes a dict of regional dicts
# each regional dict contains a dict for how many times that region experienced a drought of different legnths

regional_drought_lengths = {}
for i, region in enumerate(regions):
    drought_lengths = {}
    for j in range(len(cumulative_droughts[:, i])-1):
        if cumulative_droughts[j, i] == 0:
            continue
        if cumulative_droughts[j+1, i] != 0:
            continue
        length = cumulative_droughts[j, i]
        if f'{length}' not in drought_lengths:
            drought_lengths[f'{length}'] = 1
        else:
            drought_lengths[f'{length}'] += 1
    if cumulative_droughts[-1, i] != 0:
        length = cumulative_droughts[-1, i]
        if f'{length}' not in drought_lengths:
            drought_lengths[f'{length}'] = 1
    regional_drought_lengths[region] = drought_lengths
regional_drought_lengths


# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(10, 6))  # 4 rows and 2 columns of subplots
axes = axes.ravel()

# Plot data for each region
for idx, (region, data) in enumerate(regional_drought_lengths.items()):
    ax = axes[idx]
    ax.bar(data.keys(), data.values())
    ax.set_title(region)
    # ax.set_xlabel('Drought Length')
    # ax.set_ylabel('Number of Droughts')

# Add global x and y labels
fig.text(0.5, 0.04, 'Drought Length', ha='center', fontsize=16)  # x-label
fig.text(0.04, 0.5, 'Number of Droughts', va='center', rotation='vertical', fontsize=16)  # y-label

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig('/home/548/cd3022/figs/regional drought histogram')
plt.show()


# In[ ]:


# Produce Figure 4 from Raynaud et al. (2018)
# Make tuple (x,y) where x is the mean drought length in that region,
# and y is the total number of drought days experienced.

raynaud_plot_data = {}
for i, region in enumerate(regional_drought_lengths):
    droughts = regional_drought_lengths[region]
    num_droughts = sum(droughts.values())  # Sum all drought counts
    if num_droughts == 0:
        raynaud_plot_data[region] = (0, 0)
    else:
        raynaud_plot_data[region] = (num_drought_days[i] / num_droughts, num_droughts)

# Extract x and y values for plotting
x_values = [coords[0] for coords in raynaud_plot_data.values()]
y_values = [coords[1] for coords in raynaud_plot_data.values()]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color="blue", label="Regions")

# Annotate points with region names
for region, (x, y) in raynaud_plot_data.items():
    plt.text(x, y, region, fontsize=9, ha="right")

# Add contours of constant total drought days (x * y)
x = np.linspace(0, max(x_values) * 1.2, 100)
y = np.linspace(0, max(y_values) * 1.2, 100)
X, Y = np.meshgrid(x, y)
Z = X * Y  # Total drought days

# Contour levels
contour_levels = [1, 2, 3, 4, 5, 6, 7, 8]  # Adjust based on your data
contours = plt.contour(X, Y, Z, levels=contour_levels, colors='gray', linestyles='dotted')
plt.clabel(contours, inline=True, fontsize=8, fmt='%d')  # Label contours

# Final plot customizations
plt.xlabel("Mean Drought Length")
plt.ylabel("Number of Droughts")
plt.title("Capital City Solar Droughts")
plt.grid(True)
plt.legend()
plt.savefig('/home/548/cd3022/figs/Raynaud et al plot')
plt.show()

