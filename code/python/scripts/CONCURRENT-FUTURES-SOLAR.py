import concurrent.futures
import os,sys
from datetime import datetime, timedelta
from pathlib import Path
import logger
from compute_solar import solar_workflow

LOG = logger.get_logger(__name__)

if __name__ == '__main__':
    
    n_procs = os.cpu_count()
    worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=10)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    region = sys.argv[3]
    var = sys.argv[4]
    tilt = sys.argv[5]
    
    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")
    
    # Generate a list of dates
    date_range = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]
    
    futures = {}
    # Loop over the dates
    for date in date_range: 
        date_s = date.strftime('%Y/%m/%d')
        LOG.info(f'Computing ratio of tilting to clearsky irradiance for {date_s}')
        future = worker_pool.submit(solar_workflow, date_s, region, var, tilt) 
        futures[future] = f"The job for {date}"