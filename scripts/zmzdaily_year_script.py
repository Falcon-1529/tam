#!/glade/u/apps/opt/conda/envs/npl/bin/python
#PBS -N zmzdaily_year_job
#PBS -A UNYU0024
#PBS -j oe
#PBS -k eod
#PBS -q casper
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=13GB

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
year2_input = int(sys.argv[2])
year2 = str(year2_input)
logging.info("You inputted the years of " + year1 + " to " + year2)

# Begin
logging.info("Starting script to fetch " + year1 + " to " + year2 + " ERA5 Z data")

for year in range(year1_input, year2_input + 1):
    yearstr = str(year)
    logging.info("Processing " + yearstr)
    # information for ERA5 dataset.  See https://rda.ucar.edu/datasets/d633000/ 
    pl = 'e5.oper.an.pl/'   # pressure level data
    sl = 'e5.oper.an.sfc/'  # surface data
    era5 = '/glade/campaign/collections/rda/data/d633000/'  # path to the era5 dataset
    date = yearstr + '*' # the * is to access everything in the inputted year

    gzfile = era5 + pl + date + '/e5.oper.an.pl.128_129_z.ll025sc.' + date + '.nc'

    logging.info("Opening dataset...")
    gz = xr.open_mfdataset(gzfile, chunks={'time': 30}) # chunking so the memory won't explode

    logging.info("Calculating geopotential height")
    z=gz.Z / 9.8

    logging.info("Computing daily mean...")
    zdaily = z.resample(time='1D').mean('time') # for daily, we don't need hourly data

    logging.info("Computing zonal means in parallel for each level...")
    levels = zdaily.level.values
  # logging.info(levels) # checking stuff
    tasks = [zdaily.sel(level=lev).mean(dim='longitude') for lev in levels]
    results = dask.compute(*tasks)
    
    logging.info("Reassembling zonal means...")
    zmzdaily = xr.concat(results, dim='level')
    # zmzdaily['level'] = levels  
    
    zmzdaily_file = zmzdaily.to_netcdf(path = 'zmzdaily' + yearstr + '.nc')
    logging.info("Done with " + yearstr + " dataset!")


