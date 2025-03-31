#!/usr/bin/env python
# coding: utf-8

# # Analyse solar droughts over small area

# In[1]:


from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import cartopy.crs as ccrs
import pandas as pd
import sys
sys.path.append('code/python/scripts')
import utils


# In[5]:


# Hard coded variables
# start_date = '1-1-2023'
# num_days = 10

# System arg values
# OUT OF DATE, NEED TO CHANGE END_DATE
start_date = sys.argv[1]
end_date = int(sys.argv[2])

# Melbourne coordinates
area_bounds = {
"lat_min": -38.6,
"lat_max": -37.3,
"lon_min": 143.8,
"lon_max": 145.8,
}


# In[6]:


rad_df = utils.read_irradiance(
    start_date=start_date,
    end_date=end_date,
    area_bounds=area_bounds
)
rad_df


# In[7]:


if 'is_drought_day' not in rad_df.columns:
    total_mean = np.ma.mean(np.ma.stack(rad_df['daily_mean'].values), axis=0)
    
    # Find drought days using the definition: daily average > 50% of mean day
    is_drought_day = [
        np.ma.masked_array(
            dm < total_mean / 2,  # Condition for drought
            mask=np.ma.getmask(dm) | np.ma.getmask(total_mean)  # Combine masks
        )
        for dm in rad_df['daily_mean']
    ]
    rad_df = pd.concat([rad_df, pd.DataFrame({'is_drought_day': is_drought_day}, index = rad_df.index)], axis=1)

if 'coincident_droughts' not in rad_df.columns:
    # count the number of droughts occuring at the same time
    coincident_droughts = [np.sum(droughts) for droughts in is_drought_day]
    rad_df = pd.concat([rad_df, pd.DataFrame({'coincident_droughts': coincident_droughts}, index=rad_df.index)], axis=1)

# Calculate the number of drought days
num_drought_days = np.ma.sum(np.ma.stack(is_drought_day), axis=0)


# In[9]:


ax = rad_df['coincident_droughts'].plot()
plt.title('Number of Coincident Droughts')
plt.savefig(f'/home/548/cd3022/figs/melb_coincident_drought_timeline_{start_date}_{start_date}')
plt.ylabel('Number of Droughts')
plt.show()


# In[ ]:


fig_name = f'melb-heatmap_{start_date}_{num_days}_days.png'
utils.plot_area(
    data=num_drought_days,
    area_bounds=area_bounds,
    fig_name=fig_name
)


# In[ ]:


rad_df.to_csv(f'/home/548/cd3022/data/csv_files/melb-droughts_{start_date}_{num_days}_days.csv')

