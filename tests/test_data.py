import datetime as dt
import xarray as xr
import sys, os
import pytest

from config import *
from modules import logger
from modules.load_data import get_version

LOG = logger.get_logger(__name__)

def test_version():
    """
    Test the version code works for a variety of inputs.
    """
    dates = [dt.datetime(2019,3,30),
             dt.datetime(2019,3,31),
             dt.datetime(2019,4,1),
             dt.datetime(2019,4,2)]

    versions = [ get_version(date) for date in dates ]

    assert versions == ['v1.0', 'v1.0', 'v1.1', 'v1.1']

    # Test the SystemExist for bad values
    with pytest.raises(SystemExit) as pytest_exit:
        get_version(20190401)
    assert pytest_exit.type == SystemExit

    with pytest.raises(SystemExit) as pytest_exit:
        get_version('20190401')
    assert pytest_exit.type == SystemExit