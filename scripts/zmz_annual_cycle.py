import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

filepath = '/glade/u/home/leonardgu/tam/Z_zonal_means/'

def interpleapyear(z_data, origtime, year):
    level, time, latitude = z_data.shape
    print(str(year) + ' is a leap year!')
    orig = np.linspace(0, 1, 366)
    new = np.linspace(0, 1, 365)
    
    z_interp = np.zeros((level, 365, latitude))
    
    for lev in range(level):
        for lat in range(latitude):
            f = interp1d(orig, z_data[lev, :, lat], kind='linear',)
            z_interp[lev, :, lat] = f(new)
        
    newtime = pd.date_range(start=f'{year}-01-01T00:00:00.000000000', periods=365, freq='D')
   
    z_interp_final = xr.DataArray(z_interp, dims = ['level', 'time', 'latitude'], coords = {'level': z_data.level, 'time': newtime, 'latitude': z_data.latitude})
    
    return z_interp_final
    
def annual_cycle(filepath):
    yeardata = []
    level_coords = None
    lat_coords = None
    #shapes = [] # for debugging purposes, resolved already
    for year in range(1979, 2024):
        file_path = f"{filepath}/zmzdaily{year}.nc"

        ds = xr.open_dataset(file_path)
        z_data = ds.Z.load()
        origtime = ds.time.values
        print(f"Year {year}: original shape = {z_data.shape}")

        # Storing coords for a non-leap year
        if level_coords is None:
            level_coords = z_data.level
            lat_coords = z_data.latitude
        if year == 1979:
            z_data = z_data.transpose('level','time','latitude') # because for some reason 1979 saved with diff order of dims
        if z_data.shape[1] == 366:
            z_interped = interpleapyear(z_data, origtime, year)

        else:
            z_interped = z_data
            
        #print(f"  -> Final shape = {z_interped.shape}")
        #shapes.append(z_interped.shape)
        yeardata.append(z_interped)
    #unique_shapes = list(set(shapes))
   # print(f"\nUnique shapes found: {unique_shapes}")
    """
    if len(unique_shapes) > 1:
        print("ERROR: Not all arrays have the same shape!")
        for i, shape in enumerate(shapes):
            print(f"Year {1979 + i}: {shape}")
        return None, None  
    """ 
    clim_time = np.arange(1,366)
    totalyears = np.stack(yeardata, axis=0)
    annualcycle_val = np.mean(totalyears, axis=0)
    annualcycle = xr.DataArray(annualcycle_val, dims=['level', 'time', 'latitude'], coords = {'level': level_coords, 'time': clim_time, 'latitude': lat_coords})
    return annualcycle

annualcycda = annual_cycle(filepath)
anncycfile = annualcycda.to_netcdf('zmzdaily_annual_cycle.nc')