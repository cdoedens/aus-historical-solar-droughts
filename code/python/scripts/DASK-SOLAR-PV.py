import sys
from dask.distributed import as_completed
from datetime import datetime, timedelta
from compute_solar import solar_workflow_DASK
import climtas.nci
import logger

LOG = logger.get_logger(__name__)

start_date = sys.argv[1]
end_date = sys.argv[2]
region = sys.argv[3]


if __name__ == '__main__':
    client = climtas.nci.GadiClient()
    
    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")
    date_range = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]
    
    # Loop over the dates
    futures = {}
    for date in date_range: 
        date_s = date.strftime('%Y/%m/%d')
        LOG.info(f"Submitting job for {date_s}")
        future = client.submit(solar_workflow_DASK, date_s, region, pure=False) 
        futures[future] = f"The job for {date}"
    
    # Iterate through each result as they are completed
    for future in as_completed(futures):
        _ = future.result()  # Raises if the function failed
    
        # Remove reference to free up memory
        del futures[future]