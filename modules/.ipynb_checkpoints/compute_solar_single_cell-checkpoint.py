import pvlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import os
import logger
import rasterio
from rasterstats import zonal_stats
import odc.geo.xr
from datetime import datetime
from pathlib import Path
from glob import glob
import dask.array as da

LOG = logger.get_logger(__name__)


# STRUCTURE OF FUNCTIONS
'''
SOLAR_WORKFLOW
    | READ_DATA
    |   | _PREPROCESS
    |   | GET_FILES
    | CLEAR_SKY_PERFORMANCE
    |   | SOLAR_PV_GENERATION
'''


def solar_workflow(date, tilt, lat, lon):
    LOG.info(f'START SOLAR_WORKFLOW with: date = {date}, tilt = {tilt}')
    ds = read_data(date, lat, lon)
    solar = clear_sky_performance(ds, tilt)
    LOG.info('END OF SOLAR_WORKFLOW')
    return solar

    
def read_data(date, lat, lon):
    LOG.info(f'reading data for {date}')
    # get correct file path based on the date
    if len(date) > 7:
        dir_dt = datetime.strptime(date, "%Y/%m/%d")
    else:
        dir_dt = datetime.strptime(date, "%Y/%m")
        
    if dir_dt <= datetime.strptime('2019-03-31', '%Y-%m-%d'):
        version = 'v1.0'
    else:
        version = 'v1.1'

    # ensures lat_min etc. is available to _preprocess
    def _preprocess(ds):
        return ds[
        ['surface_global_irradiance', 'direct_normal_irradiance', 'surface_diffuse_irradiance']
        ].sel(latitude=lat, longitude=lon, method='nearest')
    
    files = get_files(date, version)
    LOG.info(f"Loading {len(files)} files")
    ds = xr.open_mfdataset(
        files,
        preprocess=_preprocess,
        combine='by_coords',
        engine="h5netcdf",
    )
   
    return ds

def get_files(date, version):
    directory = Path(f'/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/{version}/{date}')
    return sorted(str(p) for p in directory.rglob("*.nc"))
    

def clear_sky_performance(ds, tilt):

    LOG.info('reading dataset variables')
    ghi = ds.surface_global_irradiance.values
    dni = ds.direct_normal_irradiance.values
    dhi = ds.surface_diffuse_irradiance.values

    # get correct time and coordinate data, so that it matches up with the remaining irradiance values
    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values

    # BARRA-R2 temperature data
    LOG.info('getting BARRA-R2 temperature')
    year = pd.to_datetime(ds.isel(time=50).time.values.item()).year
    month = pd.to_datetime(ds.isel(time=50).time.values.item()).month

    barra_file = f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/tas/latest/tas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
    barra = xr.open_dataset(
        barra_file,
        engine='h5netcdf'
    )
    LOG.info('BARRA file opened')
    points = xr.Dataset(
        {
            "time": ("points", time),
        }
    )
    temp_barra = barra["tas"].sel(
        lat=lat,
        lon=lon,
        time=points["time"],
        method="nearest"
    )
    temp_clean = temp_barra.values - 273.15


    # calculate capacity factors using pvlib
    LOG.info(f'running pvlib functions')
    actual, ideal = solar_pv_generation(
        pv_model = 'Canadian_Solar_CS5P_220M___2009_',
        inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        time=time,
        lat=lat,
        lon=lon,
        temp=temp_clean,
        tilt=tilt
    )
    
    return xr.Dataset(
        data_vars={
            "actual": (ds.dims, actual),
            "ideal": (ds.dims, ideal)
        },
        coords=ds.coords,
        )

def solar_pv_generation(
    pv_model,
    inverter_model,
    time,
    lat,
    lon,
    dni,
    ghi,
    dhi,
    temp,
    tilt
):
    '''
    Other than pv and inverter models, all other arguments must be a flat 1D array of equal size
    '''

    if tilt[0] not in ['fixed', 'single_axis']:
        raise ValueError(f'Unrecognised tilt: {tilt[0]}. tilt must be "fixed" or "single_axis"')
    
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

    # TILTING VS FIXED AXIS VALUES
    if tilt[0] == 'single_axis':  
        # get panel/solar angles for a tilting panel system
        tracking = pvlib.tracking.singleaxis(
            apparent_zenith=solpos["apparent_zenith"],
            apparent_azimuth=solpos["azimuth"]
        )
        surface_tilt = tracking['surface_tilt']
        surface_azimuth = tracking['surface_azimuth']
        aoi = tracking['aoi']
    
    elif tilt[0] == 'fixed':
        # Find the angle of incidence
        surface_tilt = tilt[1]
        surface_azimuth = tilt[2]

        aoi = pvlib.irradiance.aoi(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solpos["apparent_zenith"],
            solar_azimuth=solpos["azimuth"],
        )

    # compute airmass data
    airmass_relative = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'].values
    )
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
        airmass_relative,
    )

    # compute irradiances
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
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
        aoi=aoi,
        module=module,
    )

    # compute ideal conditions
    linke_turbidity = np.maximum(2 + 0.1 * airmass_absolute, 2.5)
    doy = pd.to_datetime(time).dayofyear
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    ideal_conditions = pvlib.clearsky.ineichen(
        apparent_zenith=solpos['apparent_zenith'].values,
        airmass_absolute=airmass_absolute,
        linke_turbidity=linke_turbidity,
        dni_extra=dni_extra
    )
    ideal_total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
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
        aoi=aoi,
        module=module,
    )

    # Compute power outputs
    dc = pvlib.pvsystem.sapm(
        effective_irradiance=effective_irradiance.values,
        temp_cell=temp,
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
        temp_cell=temp, # assume temperature of 18 deg C
        module=module
    )
    ac_ideal = pvlib.inverter.sandia(
        v_dc=dc_ideal['v_mp'],
        p_dc=dc_ideal['p_mp'],
        inverter=inverter
    )
    ac_ideal_QC = np.where(ac_ideal < 0, np.nan, ac_ideal)

    return ac_QC, ac_ideal_QC