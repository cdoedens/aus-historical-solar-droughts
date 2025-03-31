import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pvlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# get dataset for a single time slice
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
        dataset = xr.open_dataset(dirin+filename)
    except FileNotFoundError:
        print(f"File not found: {dirin + filename}.")
        dataset = None
    return dataset

################
# improved version of above function,
# retrieves data from all files (i.e. all time steps) in a day
def get_irradiance_day(resolution, dir_dt):
    
    if dir_dt <= datetime.strptime('2019-03-31', '%Y-%m-%d'):
        ver = 'v1.0'
    else:
        ver = 'v1.1'

    dirin=f'/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/{resolution}/{ver}/'+f"{dir_dt.year:04}"+'/'+f"{dir_dt.month:02}"+'/'+f"{dir_dt.day:02}"+"/"

    files = f'{dirin}/*.nc'
    daily_data = xr.open_mfdataset(files, combine="by_coords")
    return daily_data
################
# find the capacity factor of fixed tilt solar pv based on irradiance data
def solar_cf(
    pv_model,
    inverter_model,
    ghi,
    dni,
    dhi,
    time,
    lat_grid,
    lon_grid,
):
    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules[pv_model]
    inverter = sapm_inverters[inverter_model]

    # panel orientation
    surface_azimuth = 0
    surface_tilt = -lat_grid.ravel()

    # Define PV system
    system = {
        'module': module,
        'inverter': inverter,
        'surface_azimuth': surface_azimuth,
        'surface_tilt': surface_tilt
    }

    # Compute solar position for all grid cells at once
    time_array = np.full_like(lat_grid.ravel(), time, dtype=object)
    solpos = pvlib.solarposition.get_solarposition(
        time_array,
        lat_grid.ravel(),
        lon_grid.ravel()
    )
    
    # Find the angle of incidence
    aoi = pvlib.irradiance.aoi(
        surface_tilt=surface_tilt.data,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
    )

    # Compute air mass
    airmass_relative = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'].values
    )
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
        airmass_relative,
    )

    # Compute irradiancees
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt.data,
        surface_azimuth=surface_azimuth,
        dni=dni.ravel(),
        ghi=ghi.ravel(),
        dhi=dhi.ravel(),
        solar_zenith=solpos['apparent_zenith'].values,
        solar_azimuth=solpos['azimuth'].values
    )

    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        poa_direct=total_irradiance['poa_direct'],
        poa_diffuse=total_irradiance['poa_diffuse'],
        airmass_absolute=airmass_absolute,
        aoi=aoi,
        module=module,
    )

    effective_irradiance_QC = effective_irradiance.clip(lower=0)

    # Compute power
    dc = pvlib.pvsystem.sapm(
        effective_irradiance=effective_irradiance_QC,
        temp_cell=np.full_like(effective_irradiance, 25), # assume temperature of 18 deg C
        module=module
    )
    ac = pvlib.inverter.sandia(
        v_dc=dc['v_mp'],
        p_dc=dc['p_mp'],
        inverter=inverter
    )
    ac_QC = np.where(ac < 0, np.nan, ac)
    ac_aus = ac_QC.reshape(lat_grid.shape)
    rated_capacity = module.loc['Impo'] * module.loc['Vmpo']

    return ac_aus / rated_capacity # capacity factor

##################

def plot_data(data, lat, lon):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    mesh=ax.pcolormesh(lon, lat, data, cmap='viridis',vmin=0, transform=ccrs.PlateCarree())
    
    ax.coastlines()
    cbar = plt.colorbar(mesh,ax=ax,shrink=0.5)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('Value', fontsize=5) 
     
    plt.tight_layout()
    
    plt.show()

#######################
# find the ratio between solar pv generation from a tilting panel relative to ideal clear sky conditions
def tilting_panel_pr(
    pv_model,
    inverter_model,
    time,
    lat,
    lon,
    dni,
    ghi,
    dhi
    
):
    '''
    Other than pv and inverter models, all other arguments must be a flat 1D array of equal size
    '''
    
    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules[pv_model]
    inverter = sapm_inverters[inverter_model]

    # Compute solar position for all grid cells at once
    solpos = pvlib.solarposition.get_solarposition(
        time,
        lat,
        lon,
    )
    # get panel/solar angles for a tilting panel system
    tracking = pvlib.tracking.singleaxis(
        apparent_zenith=solpos["apparent_zenith"],
        apparent_azimuth=solpos["azimuth"]
    )
    # compute airmass data
    airmass_relative = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'].values
    )
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
        airmass_relative,
    )

    # copmute irradiances
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracking['surface_tilt'],
        surface_azimuth=tracking['surface_azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos['apparent_zenith'].values,
        solar_azimuth=solpos['azimuth'].values
    )
    
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        poa_direct=total_irradiance['poa_direct'],
        poa_diffuse=total_irradiance['poa_diffuse'],
        airmass_absolute=airmass_absolute,
        aoi=tracking['aoi'],
        module=module,
    )

    # compute ideal conditions
    linke_turbidity = np.maximum(2 + 0.1 * airmass_absolute, 2.5) # simplified parameterisation from ChatGPT as pvlib function was not working with array
    doy = pd.to_datetime(time).dayofyear
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    ideal_conditions = pvlib.clearsky.ineichen(
        apparent_zenith=solpos['apparent_zenith'].values,
        airmass_absolute=airmass_absolute,
        linke_turbidity=linke_turbidity,
        dni_extra=dni_extra
    )
    ideal_total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracking['surface_tilt'],
        surface_azimuth=tracking['surface_azimuth'],
        dni=ideal_conditions['dni'],
        ghi=ideal_conditions['ghi'],
        dhi=ideal_conditions['dhi'],
        solar_zenith=solpos['apparent_zenith'].values,
        solar_azimuth=solpos['azimuth'].values
    )
        
    ideal_effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        poa_direct=ideal_total_irradiance['poa_direct'],
        poa_diffuse=ideal_total_irradiance['poa_diffuse'],
        airmass_absolute=airmass_absolute,
        aoi=tracking['aoi'],
        module=module,
    )

    # Compute power outputs
    dc = pvlib.pvsystem.sapm(
        effective_irradiance=effective_irradiance.values,
        temp_cell=np.full_like(effective_irradiance, 18), # assume temperature of 18 deg C
        module=module
    )
    
    ac = pvlib.inverter.sandia(
        v_dc=dc['v_mp'],
        p_dc=dc['p_mp'],
        inverter=inverter
    )
    ac_QC = np.where(ac < 0, np.nan, ac)
    # ideal power output
    dc_ideal = pvlib.pvsystem.sapm(
        effective_irradiance=ideal_effective_irradiance.values,
        temp_cell=np.full_like(effective_irradiance, 18), # assume temperature of 18 deg C
        module=module
    )
    ac_ideal = pvlib.inverter.sandia(
        v_dc=dc_ideal['v_mp'],
        p_dc=dc_ideal['p_mp'],
        inverter=inverter
    )
    ac_ideal_QC = np.where(ac_ideal < 0, np.nan, ac_ideal)
    actual_ideal_ratio = ac_QC / ac_ideal_QC

    return actual_ideal_ratio

