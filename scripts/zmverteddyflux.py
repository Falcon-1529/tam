import xarray as xr
import time
import logging
import dask
import sys
import gc
import psutil
import os

def log_memory():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    logging.info(f"Current memory usage: {mem_gb:.2f} GB")

# logging stuff
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Input
year1_input = int(sys.argv[1])
year1 = str(year1_input)

logging.info("You inputted the year of " + year1)

# Begin
logging.info("Starting script to fetch " + year1 + " ERA5 u and w data for vertical momentum eddy flux calculations.")


logging.info("Processing " + year1)

pl = 'e5.oper.an.pl/'   # pressure level data
sl = 'e5.oper.an.sfc/'  # surface data
era5 = '/glade/campaign/collections/rda/data/d633000/'  # path to the era5 dataset
date = year1 + '*' # the * is to access everything in the inputted year

ufile = era5 + pl + date + '/e5.oper.an.pl.128_131_u.ll025uv.' + date + '.nc'
wfile = era5 + pl + date + '/e5.oper.an.pl.128_135_w.ll025sc.' + date + '.nc'


logging.info("Opening datasets for u and w")
uds = xr.open_mfdataset(ufile)
wds = xr.open_mfdataset(wfile)

u = uds.U
w = wds.W

logging.info("Calculating zonal means for u and w")
# Calculate zonal means for u, v and T
start_time = time.perf_counter()
zmu = u.mean(dim = 'longitude').compute()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
logging.info(f"Time taken for zonal mean u: {elapsed_time:.6f} seconds")

start_time = time.perf_counter()
zmw = w.mean(dim = 'longitude').compute()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
logging.info(f"Time taken for zonal mean w: {elapsed_time:.6f} seconds")

logging.info("Calculating anomalies for u and w")

# Calculate the deviations from the zonal mean
uprime = u - zmu
wprime = w - zmw

# Drop u and w. We don't need them anymore (need to save memory)
del u
del w
del uds
del wds

gc.collect() 
log_memory()

logging.info("Computing daily mean vertical momentum eddy flux for " + year1)
momeflux_daily = (uprime * wprime).resample(time='1D').mean('time')
momeflux_daily.name = 'upwp'

log_memory()

del uprime
del wprime
del zmu
del zmw

gc.collect()
log_memory()

with dask.config.set(array__slicing__split_large_chunks=True):

    # Process vertical momentum eddy flux (u'w')
    logging.info("Computing zonal means in parallel for each level for vertical momentum flux " + year1)
    levels = momeflux_daily.level.values
    tasks = [momeflux_daily.sel(level=lev).mean(dim='longitude') for lev in levels]
    results = dask.compute(*tasks)
    log_memory()
    logging.info("Reassembling zonal means for vertical momentum flux...")
    zm_momeflux_daily = xr.concat(results, dim='level')
    
    logging.info("Saving vertical momentum flux to netCDF...")
    zm_momeflux_daily.to_netcdf(path='zm_vmomeflux' + year1 + '.nc')
    logging.info("Done with vertical momentum flux for " + year1)
    
    del momeflux_daily
    del zm_momeflux_daily
    gc.collect() 


logging.info("All processing complete for " + year1 + "!")
