import xarray as xr
import logging
import dask
import sys

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
logging.info("Starting script to fetch " + year1 + " ERA5 U data")


logging.info("Processing " + year1)

# information for ERA5 dataset.  See https://rda.ucar.edu/datasets/d633000/ 
pl = 'e5.oper.an.pl/'   # pressure level data
sl = 'e5.oper.an.sfc/'  # surface data
era5 = '/glade/campaign/collections/rda/data/d633000/'  # path to the era5 dataset
date = year1 + '*' # the * is to access everything in the inputted year

# file header to get files 
gzfile = pl + date + '/e5.oper.an.pl.128_129_z.ll025sc.' + date + '.nc'

with dask.config.set(array__slicing__split_large_chunks=True):
    logging.info("Opening dataset for " + year1)
    gz = xr.open_mfdataset(gzfile, chunks={'time': 30}) 

    logging.info("Calculating geopotential height")
    z=gz.Z / 9.8

    logging.info("Computing daily mean for " + year1)
    zdaily = z.resample(time='1D').mean('time') # for daily, we don't need hourly data

    logging.info("Computing zonal means in parallel for each level for " + year1)
    levels = zdaily.level.values
    # logging.info(levels) # checking stuff
    tasks = [zdaily.sel(level=lev).mean(dim='longitude') for lev in levels]
    results = dask.compute(*tasks)
    
    logging.info("Reassembling zonal means...")
    zmzdaily = xr.concat(results, dim='level')
    # zmzdaily['level'] = levels  
    
    zmzdaily_file = zmzdaily.to_netcdf(path = 'zmudaily' + year1 + '.nc')
    logging.info("Done with " + year1 + " dataset!")


