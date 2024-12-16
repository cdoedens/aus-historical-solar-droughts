#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/548/cd3022/code/python/scripts')
import utils


# In[ ]:


import os
print("Environment:", os.environ.get('CONDA_DEFAULT_ENV'))
print("Current working directory:", os.getcwd())


# In[2]:


# System arg values
start_date = sys.argv[1]
end_date = sys.argv[2]


# In[3]:


rad_df = utils.read_irradiance(
    start_date=start_date,
    end_date=end_date
)
print('read_irradiance complete')


# In[5]:


rad_df.to_pickle(f'/g/data/er8/users/cd3022/solar_drought/seasonal_daily_means/{start_date}_{end_date}.pkl')

