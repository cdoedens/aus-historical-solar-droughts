import datetime as dt
import xarray as xr
from modules import logger

LOG = logger.get_logger(__name__)


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
        print ('raise SystemExit')
    
    if date < valid_date:
        return 'v1.0'
    elif date >= valid_date:
        return 'v1.1'


def load_day(resolution,date):
    """
    Loads a day's worth of irradiance data
    """
