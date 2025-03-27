import datetime as dt
import xarray as xr
from modules import logger
from pathlib import Path


LOG = logger.get_logger(__name__)

HIMAWARI_SOLAR_DIR=Path('/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/')

def get_version(date):
    """
    Determine the version of the dataset
    v1 = valid before April 1st 2019
    v1.1 = valid after April 1st 2019
    """
    valid_date = dt.datetime(2019,4,1)

    # Check input date is a datetime object
    if isinstance(date,dt.datetime):
        pass
    else:
        LOG.fatal(f'{date} is not a datetime object')
        LOG.fatal('Check the processing of datetime objects for irradiance retrieval')
        raise SystemExit
    
    if date < valid_date:
        return 'v1.0'
    elif date >= valid_date:
        return 'v1.1'


def load_day(resolution,date):
    """
    Loads a day's worth of irradiance data
    """

    version = get_version(date)

    dir = HIMAWARI_SOLAR_DIR / resolution / version / f"{date.year:04}" / f"{date.month:02}" / f"{date.day:02}"
    
    day = xr.open_mfdataset(dir.glob('*.nc'), combine="by_coords")

    return day