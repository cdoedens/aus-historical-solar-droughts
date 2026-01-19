import concurrent.futures
import os, sys
from datetime import datetime, timedelta
import logger
from compute_solar import solar_workflow

LOG = logger.get_logger(__name__)

if __name__ == "__main__":
    start_date = sys.argv[1]
    end_date   = sys.argv[2]
    region     = sys.argv[3]
    tilt       = sys.argv[4]

    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt   = datetime.strptime(end_date, "%d-%m-%Y")
    date_range = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]

    # Keep it <= PBS ncpus; or set from env: int(os.environ.get("PBS_NCPUS", 1))
    max_workers = int(os.environ.get("PBS_NCPUS", "1"))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for date in date_range:
            date_s = date.strftime("%Y/%m/%d")
            LOG.info(f"Submitting {date_s} to worker pool")
            futures.append(pool.submit(solar_workflow, date_s, region, tilt))

        LOG.info("All jobs submitted; waiting for completion")

        for fut in concurrent.futures.as_completed(futures):
            # If a worker crashes, this will raise here and you’ll get a traceback in the PBS log
            fut.result()

    LOG.info("All jobs completed")