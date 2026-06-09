import xarray as xr
import logging
import dask
import sys

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

year1_input = int(sys.argv[1])
year1 = str(year1_input)

logging.info("You inputted the year of " + year1)
logging.info("Starting script to fetch " + year1 + " ERA5 V data")

pl = 'e5.oper.an.pl/'
era5 = '/glade/campaign/collections/rda/data/d633000/'
date = year1 + '*'

vfile = era5 + pl + date + '/e5.oper.an.pl.128_132_v.ll025uv.' + date + '.nc'

with dask.config.set(array__slicing__split_large_chunks=True):
    logging.info("Opening dataset for " + year1)
    vds = xr.open_mfdataset(vfile) 

    logging.info("Getting variable V...")
    v = vds.V

    logging.info("Computing daily mean for " + year1)
    vdaily = v.resample(time='1D').mean('time')

    logging.info("Computing zonal means in parallel for each level for " + year1)
    levels = vdaily.level.values
    tasks = [vdaily.sel(level=lev).mean(dim='longitude') for lev in levels]
    results = dask.compute(*tasks)
    
    logging.info("Reassembling zonal means...")
    zmVdaily = xr.concat(results, dim='level')
    
    zmVdaily.to_netcdf(path='zmVdaily' + year1 + '.nc')
    logging.info("Done with " + year1 + " dataset!")