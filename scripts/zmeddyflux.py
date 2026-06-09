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
logging.info("Starting script to fetch " + year1 + " ERA5 U, V and T data for momentum + heat eddy flux calculations.")


logging.info("Processing " + year1)

# Need u, v, and T
pl = 'e5.oper.an.pl/'   # pressure level data
sl = 'e5.oper.an.sfc/'  # surface data
era5 = '/glade/campaign/collections/rda/data/d633000/'  # path to the era5 dataset
date = year1 + '*' # the * is to access everything in the inputted year

ufile = era5 + pl + date + '/e5.oper.an.pl.128_131_u.ll025uv.' + date + '.nc'
vfile = era5 + pl + date + '/e5.oper.an.pl.128_132_v.ll025uv.' + date + '.nc'
tfile = era5 + pl + date + '/e5.oper.an.pl.128_130_t.ll025sc.' + date + '.nc'

logging.info("Opening datasets for u, v and T...")
uds = xr.open_mfdataset(ufile)
vds = xr.open_mfdataset(vfile)
tds = xr.open_mfdataset(tfile)

u = uds.U
v = vds.V
t = tds.T

logging.info("Calculating zonal means for u, v and T...")
# Calculate zonal means for u, v and T
start_time = time.perf_counter()
zmu = u.mean(dim = 'longitude').compute()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
logging.info(f"Time taken for zonal mean u: {elapsed_time:.6f} seconds")

start_time = time.perf_counter()
zmv = v.mean(dim = 'longitude').compute()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
logging.info(f"Time taken for zonal mean v: {elapsed_time:.6f} seconds")

start_time = time.perf_counter()
zmt = t.mean(dim = 'longitude').compute()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
logging.info(f"Time taken for zonal mean t: {elapsed_time:.6f} seconds")

logging.info("Calculating anomalies for u, v and T")

# Calculate the deviations from the zonal mean
uprime = u - zmu
vprime = v - zmv
tprime = t - zmt

# Drop u, v and t. We don't need them anymore (need to save memory)
del u
del v
del t
del uds
del vds
del tds
gc.collect() 
log_memory()

logging.info("Computing daily mean momentum eddy flux for " + year1)
momeflux_daily = (uprime * vprime).resample(time='1D').mean('time')
momeflux_daily.name = 'upvp'

log_memory()

logging.info("Computing daily mean meridional heat eddy flux for " + year1)
meridheatflux_daily = (vprime * tprime).resample(time='1D').mean('time')
meridheatflux_daily.name = 'vptp'

del uprime
del vprime
del tprime
del zmu
del zmv
del zmt
gc.collect()
log_memory()

with dask.config.set(array__slicing__split_large_chunks=True):

    # Process momentum eddy flux (u'v')
    logging.info("Computing zonal means in parallel for each level for momentum flux " + year1)
    levels = momeflux_daily.level.values
    tasks = [momeflux_daily.sel(level=lev).mean(dim='longitude') for lev in levels]
    results = dask.compute(*tasks)
    log_memory()
    logging.info("Reassembling zonal means for momentum flux...")
    zm_momeflux_daily = xr.concat(results, dim='level')
    
    logging.info("Saving momentum flux to netCDF...")
    zm_momeflux_daily.to_netcdf(path='zm_momeflux' + year1 + '.nc')
    logging.info("Done with momentum flux for " + year1)
    
    del momeflux_daily
    del zm_momeflux_daily
    gc.collect() 

    # Process meridional heat eddy flux (v'T')
    logging.info("Computing zonal means in parallel for each level for heat flux " + year1)
    tasks = [meridheatflux_daily.sel(level=lev).mean(dim='longitude') for lev in levels]
    results = dask.compute(*tasks)
    log_memory()
    logging.info("Reassembling zonal means for heat flux...")
    zm_meridheatflux_daily = xr.concat(results, dim='level')
    
    logging.info("Saving heat flux to netCDF...")
    zm_meridheatflux_daily.to_netcdf(path='zm_meridheatflux' + year1 + '.nc')
    logging.info("Done with heat flux for " + year1)

logging.info("All processing complete for " + year1 + "!")