import pvlib
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
import dask
from dask import delayed
import os
import logger

def clear_sky_performance(ds):

    LOG = logger.get_logger(__name__)
   
    ghi = ds.surface_global_irradiance.data
    dni = ds.direct_normal_irradiance.data
    dhi = ds.surface_diffuse_irradiance.data
    nan_mask = da.isnan(ghi)
    nan_mask_flat = nan_mask.ravel()
    
    # Apply mask lazily with Dask
    ghi_clean = da.compress(~nan_mask_flat, ghi, axis=None)
    dni_clean = da.compress(~nan_mask_flat, dni, axis=None)
    dhi_clean = da.compress(~nan_mask_flat, dhi, axis=None)

    # get correct time and coordinate data, so that it matches up with the remaining irradiance values
    lat_1d = ds.latitude.data
    lon_1d = ds.longitude.data
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")

    lat_grid_1d = lat_grid.ravel()
    lon_grid_1d = lon_grid.ravel()

    lat_1d_expanded = da.tile(lat_grid_1d, ds.sizes["time"])
    lon_1d_expanded = da.tile(lon_grid_1d, ds.sizes["time"])
    time_1d = da.repeat(da.from_array(ds.time.data, chunks="auto"), len(lat_grid_1d))

    lat_1d_expanded_clean = da.compress(~nan_mask_flat, lat_1d_expanded, axis=None)
    lon_1d_expanded_clean = da.compress(~nan_mask_flat, lon_1d_expanded, axis=None)
    time_1d_clean = da.compress(~nan_mask_flat, time_1d, axis=None)

    # calculate capacity factors using pvlib
    actual_ideal_ratio = tilting_panel_pr(
        pv_model = 'Canadian_Solar_CS5P_220M___2009_',
        inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        ghi=ghi_clean,
        dni=dni_clean,
        dhi=dhi_clean,
        time=time_1d_clean,
        lat=lat_1d_expanded_clean,
        lon=lon_1d_expanded_clean
    )
    
    # Create a full flattened output array, filling with NaN (or another default value)
    result_flat = np.full(ghi.size, np.nan)
    
    # Insert the processed values into the correct positions.
    result_flat[~nan_mask_flat] = actual_ideal_ratio
    
    # Reshape the flat array back into the original shape.
    result = result_flat.reshape(ghi.shape)
    ratio_da = xr.DataArray(result, coords=ds.coords, dims=ds.dims)
    
    return ratio_da

def save_NEM_timeseries(da):
    LOG = logger.get_logger(__name__)

    nem_timeseries = da.mean(dim=["latitude", "longitude"], skipna=True)
    
    file_path = '/g/data/er8/users/cd3022/solar_drought/REZ_tilting/TEST'
    os.makedirs(file_path, exist_ok=True)
    LOG.info(f'Writing data to {file_path}/TEST.nc')
    nem_timeseries.to_netcdf(f'{file_path}/TEST.nc')
    return





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