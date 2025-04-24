import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectrum import pmtm
import pywt
from scipy.signal import welch


'''
UPDATE DROUGHT FUNCTIONS TO JUST PRODUCE DATA, PLOTTING DONE MANUALLY IN SCRIPT.
WILL HELP TRANSFERABILITY TO DIFFERENT TIME SCALES
'''

def clip_dusk_dawn(da, n=1):
    """Set the first and last `n` non-NaN values of each day in a 1D time series to NaN."""
    da_out = da.copy()
    grouped = da.groupby('time.date')

    for date, day_group in grouped:
        # Get the original indexes of the day's data
        indices = da.time.to_index().get_indexer(day_group.time)

        # Find non-NaN values
        numeric = day_group.values
        valid_idxs = np.where(~np.isnan(numeric))[0]

        # Indices to set to NaN: first n and last n non-NaNs
        to_nan = np.concatenate([valid_idxs[:n], valid_idxs[-n:]])
        times_to_nan = day_group.time[to_nan]
        da_out.loc[{'time': times_to_nan}] = np.nan

    return da_out

def constant_below_threshold(da, threshold, linestyle='-', multiplot=False, save=False):
    seasons = {
        'summer': [12,1,2],
        'autumn': [3,4,5],
        'winter': [6,7,8],
        'spring': [9,10,11]
    }

    da_droughts = xr.where(da < threshold, 1, 0)

    results = {}
    for season in seasons:
        data = da_droughts.where(da_droughts.time.dt.month.isin(seasons[season]), drop=True).values
    
        drought_lengths = [0]
        for i_time in range(1, len(data)):
            drought_past = data[i_time - 1]
            drought_now = data[i_time]
            if (drought_now != 0) and (drought_past != 0):
                drought_now += drought_lengths[i_time - 1]
            drought_lengths.append(drought_now)
    
        length, freq = np.unique(drought_lengths, return_counts = True)
        length = np.array(length[1:]) / 6
        # num / season in data
        freq = freq[1:] / len(np.unique(da.time.dt.year))
        results[season] = (length, freq)

    return results

def mean_below_threshold(da, threshold, max_hours):
    seasons = {
        'summer': [12,1,2],
        'autumn': [3,4,5],
        'winter': [6,7,8],
        'spring': [9,10,11]
    }

    drought_dict = {
        'window_size (hrs)':[],
        'summer':[],
        'autumn':[],
        'winter':[],
        'spring':[]
    }
    
    windows = range(1, max_hours * 6)
    
    for window in windows:
    
        drought_dict['window_size (hrs)'].append(window / 6)
        window_mean = da.rolling(time=window, center=False).mean()
        window_droughts = xr.where(window_mean < threshold, 1, 0)

        # 
        shifted = window_droughts.shift(time=1, fill_value=0)
        drought_starts = xr.where((window_droughts == 1) & (shifted == 0), 1, 0)
    
        for season in seasons:
            droughts = drought_starts.where(drought_starts.time.dt.month.isin(seasons[season]), drop=True).values.sum() / len(np.unique(da.time.dt.year))
            drought_dict[season].append(droughts)
    
    drought_df = pd.DataFrame(drought_dict)
    drought_df.set_index('window_size (hrs)', inplace = True)
    return drought_df


'''
LOTS OF ROOM IN THIS FUNCTION TO PLAY WITH THRESHOLD DEFINITIONS
'''
def daily_drought(da, threshold):
    res = []
    dates = []
    for date, day_data in da.groupby('time.date'):
        day_drought = xr.where(day_data < threshold, 1, 0).data
    
        # get the maximum "drought" length for this day
        drought_lengths = [0]
        for i in range(1, len(day_drought)):
            drought_past = day_drought[i - 1]
            drought_now = day_drought[i]
            if (drought_now != 0) and (drought_past != 0):
                drought_now += drought_lengths[i - 1]
            drought_lengths.append(drought_now)
        max_length = np.max(drought_lengths)
    
        
        # Different criteria for assessing if this day is a "drought"
        if np.sum(day_drought) > 12: # total 2 hrs below threshold
            res.append(1)
        elif max_length >= 6: # at least 1 hour below threshold
            
            res.append(1)
        elif day_data.mean().data < threshold: # mean for the whole day below threshold
            
            res.append(1)
        else:
            res.append(0)
        dates.append(np.datetime64(date))
    
    temp_da = xr.DataArray(res, coords={'time': dates}, dims='time')
    
    seasons = {
        'summer': [12,1,2],
        'autumn': [3,4,5],
        'winter': [6,7,8],
        'spring': [9,10,11]
    }
        
    results = {}
    for season in seasons:
        data = temp_da.where(temp_da.time.dt.month.isin(seasons[season]), drop=True).values
    
        drought_lengths = [0]
        for i_time in range(1, len(data)):
            drought_past = data[i_time - 1]
            drought_now = data[i_time]
            if (drought_now != 0) and (drought_past != 0):
                drought_now += drought_lengths[i_time - 1]
            drought_lengths.append(drought_now)
    
        length, freq = np.unique(drought_lengths, return_counts = True)
        length = np.array(length[1:])
        # num / season in data
        freq = freq[1:] / len(np.unique(temp_da.time.dt.year))
        results[season] = (length, freq)
    return results


def day_time_df(da):
    
    da_f = da.copy()
    
    time = da['time'].to_index()
    
    da_f.coords['time_of_day'] = ('time', time.time)
    da_f.coords['day_of_year'] = ('time', time.dayofyear)
    
    da_f = da_f.assign_coords(
        time_of_day=("time", time.time),
        day_of_year=("time", time.dayofyear)
    )
    
    df = da_f.to_dataframe(name="value").reset_index()
    full_times = pd.date_range("04:30", "21:30", freq="10min").time
    
    return df.pivot_table(
        index="day_of_year",
        columns="time_of_day",
        values="value",
        aggfunc="mean"
    ).reindex(columns=full_times)


def day_time_heatmap(df, vrange=(1,0)):
    
    plt.figure(figsize=(15, 8))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    plt.imshow(df, aspect="auto", origin="lower", cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    
    # Set x-axis labels (time of day)
    xticks = np.linspace(0, len(df.columns) - 1, 12, dtype=int)  # Select 12 evenly spaced time labels
    xtick_labels = [df.columns[i].strftime("%H:%M") for i in xticks]  # Format as HH:MM
    plt.xticks(xticks, xtick_labels, rotation=45)
    
    # Y-axis (Convert Day of Year → Month-Day)
    yticks = np.linspace(0, len(df.index) - 1, 11, dtype=int)  # Select 10 evenly spaced day labels
    ytick_labels = [(pd.Timestamp(f"2024-01-01") + pd.Timedelta(days=int(df.index[i]) - 1)).strftime("%b %d") for i in yticks]
    plt.yticks(yticks, ytick_labels)
    
    plt.xlabel("Time of Day")
    plt.ylabel("Day of Year")
    plt.colorbar(label="Mean Performance")
    plt.tight_layout()
    plt.show()


def spectral_fft(da, time_res='1D', clim=False):
    daily_mean = da.resample(time=time_res).mean()
    if clim:
        clim = daily_mean.groupby("time.dayofyear").mean("time")
        daily_mean = daily_mean.groupby("time.dayofyear") - clim
    clean = daily_mean.dropna(dim="time")
    data = (clean - clean.mean(dim='time')).values

    N = len(data)
    fft_vals = np.fft.fft(data)
    freqs = np.fft.fftfreq(N, d=1)
    
    positive = freqs > 0
    freqs_pos = freqs[positive]
    power = np.abs(fft_vals[positive])**2 / N**2

    # Normalize to match variance via Parseval
    delta_f = freqs_pos[1] - freqs_pos[0]
    power *= np.var(data) / (np.sum(power) * delta_f)

    periods = 1 / freqs_pos
    return periods, power

def spectral_multitaper(da, NW=3, k=5):
    daily = da.resample(time="1D").mean()
    clean = daily.dropna(dim="time")
    signal = (clean - clean.mean(dim="time")).values

    Sk, weights, _ = pmtm(signal, NW=NW, k=k, method='adapt', show=False)
    weights = weights.T  # Ensure correct shape
    power = (np.abs(Sk)**2 * weights).mean(axis=0)

    freqs = np.linspace(0, 1, len(power), endpoint=False)  # cycles/day
    delta_f = freqs[1] - freqs[0]

    # Normalize using Parseval
    power *= np.var(signal) / (np.sum(power) * delta_f)

    periods = 1 / freqs[1:]
    power = power[1:]
    return periods, power

def spectral_welch(da, nperseg=2048, time_res='1D', clim=False):
    daily_mean = da.resample(time=time_res).mean()
    if clim:
        clim = daily_mean.groupby("time.dayofyear").mean("time")
        daily_mean = daily_mean.groupby("time.dayofyear") - clim
    clean = daily_mean.dropna(dim="time")
    signal = (clean - clean.mean(dim="time")).values

    fs = 1  # cycles/day
    freqs, pxx = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))

    # Normalize using Parseval
    delta_f = freqs[1] - freqs[0]
    pxx *= np.var(signal) / (np.sum(pxx) * delta_f)

    valid = freqs > 0
    periods = 1 / freqs[valid]
    power = pxx[valid]
    return periods, power 