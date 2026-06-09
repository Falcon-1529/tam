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
logging.info("Starting script to fetch " + year1 + " ERA5 MSLP data")

logging.info("Processing " + year1)

# information for ERA5 dataset.  See https://rda.ucar.edu/datasets/d633000/ 
sl = 'e5.oper.an.sfc/'  # surface data
era5 = '/glade/campaign/collections/rda/data/d633000/'  # path to the era5 dataset
date = year1 + '*' # the * is to access everything in the inputted year

mslfile = era5 + sl + date + '/e5.oper.an.sfc.128_151_msl.ll025sc.' + date + '.nc'

with dask.config.set(array__slicing__split_large_chunks=True):
    logging.info("Opening dataset for " + year1)
    mslds = xr.open_mfdataset(mslfile)  

    logging.info("Getting variable MSL...")
    msl = mslds.MSL

    logging.info("Computing daily mean for " + year1)
    msldaily = msl.resample(time='1D').mean('time').compute() 
    
    logging.info("Writing to netCDF...")
    msldaily.to_netcdf('MSLdaily' + year1 + '.nc')  
    logging.info("Done with " + year1 + " dataset!")