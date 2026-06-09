import xarray as xr
import logging
import dask
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

year1_input = int(sys.argv[1])
year1 = str(year1_input)

logging.info("Processing " + year1)

fmf = 'e5.oper.fc.sfc.meanflux/'
era5 = '/glade/campaign/collections/rda/data/d633000/'
date = year1 + '*'

precipfluxfile = era5 + fmf + date + '/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc.' + date + '.nc'

with dask.config.set(array__slicing__split_large_chunks=True):
    logging.info("Opening dataset for " + year1)
    mslds = xr.open_mfdataset(precipfluxfile)
    mtpr = mslds.MTPR
    
    logging.info("Converting forecast structure to continuous time...")
    init_times, fcast_hours = np.meshgrid(
        mtpr.forecast_initial_time.values,
        mtpr.forecast_hour.values,
        indexing='ij'
    )
    actual_times = init_times.ravel() + (fcast_hours.ravel().astype('timedelta64[h]'))
    
    mtpr_flat = mtpr.stack(time_flat=('forecast_initial_time', 'forecast_hour'))
    mtpr_flat = mtpr_flat.assign_coords(time=('time_flat', actual_times))
    mtpr_flat = mtpr_flat.swap_dims({'time_flat': 'time'})
    mtpr_flat = mtpr_flat.drop_vars(['forecast_initial_time', 'forecast_hour', 'time_flat'])
    mtpr_flat = mtpr_flat.sortby('time')
    
    logging.info("Computing daily total accumulation...")
    mtpr_daily = mtpr_flat.resample(time='1D').mean('time')
    mtpr_daily_m = mtpr_daily * 86400 / 1000
    mtpr_daily_m.attrs['units'] = 'm'
    
    logging.info("Computing zonal means for " + year1)

    zmmtprdaily = mtpr_daily_m.mean(dim='longitude').compute()

    logging.info("Writing to netCDF...")
    zmmtprdaily.to_netcdf('zmMTPRdaily' + year1 + '.nc')
    logging.info("Done with " + year1 + " dataset!")