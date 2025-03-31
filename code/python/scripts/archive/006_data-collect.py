#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/548/cd3022/code/python/scripts')
import utils
import subprocess


# In[2]:


import os
print("Environment:", os.environ.get('CONDA_DEFAULT_ENV'))
print("Current working directory:", os.getcwd())


# In[3]:


# System arg values
start_date = sys.argv[1]
end_date = sys.argv[2]
print(f'start: {start_date}')
print(f'end: {end_date}')


# In[4]:


rad_df = utils.read_irradiance(
    start_date=start_date,
    end_date=end_date
)
print('read_irradiance complete')


# In[ ]:


rad_df.to_pickle(f'/g/data/er8/users/cd3022/solar_drought/seasonal_daily_means/{start_date}_{end_date}.pkl')
print('data saved')


# In[5]:


if __name__ == '__main__':
    
    NOTEBOOK_PATH="/home/548/cd3022/code/python/notebooks/006_data-collect.ipynb"
    SCRIPT_PATH="/home/548/cd3022/code/python/scripts/006_data-collect"
    
    get_ipython().system('jupyter nbconvert --to script {NOTEBOOK_PATH} --output {SCRIPT_PATH}')

