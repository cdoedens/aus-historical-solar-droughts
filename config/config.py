from pathlib import Path
import os

user_id = os.environ['USER']

HIMAWARI_SOLAR_DIR=Path('/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/')

TIMESERIES_DIR=Path(f'/g/data/er8/users/{user_id}/solar_drought')