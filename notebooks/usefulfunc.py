# File for useful functions 
import xarray as xr
from scipy.interpolate import interp1d
from scipy.signal import correlate, butter, filtfilt
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import re
import gc
import os

# Environment variables
WORK = os.environ["WORK"]
SCRATCH = os.environ["SCRATCH"]

#######################################################################
# Interpolation for leap years via normalization of the time dimension.
# See the annual cycle function below for an example of use.
#######################################################################
def interpleapyear(data, year, surface=False):
    n_days = len(data.time)
    print(str(year) + ' is a leap year or does not have 365 time coordinates!')
    
    orig = np.linspace(0, 1, n_days)
    new = np.linspace(0, 1, 365)
    
    def interp_1d(timeseries):
        f = interp1d(orig, timeseries, kind='linear')
        return f(new)
    
    if surface:
        # 2D: (time, latitude)
        interp = xr.apply_ufunc(
            interp_1d,
            data,
            input_core_dims=[['time']],
            output_core_dims=[['new_time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            output_sizes={'new_time': 365}
        )
    else:
        # 3D: (level, time, latitude)
        interp = xr.apply_ufunc(
            interp_1d,
            data,
            input_core_dims=[['time']],
            output_core_dims=[['new_time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            output_sizes={'new_time': 365}
        )
        
    interp = interp.rename({'new_time': 'time'})
    interp = interp.assign_coords({'time': np.arange(1, 366)})
    
    if surface:
        interp = interp.transpose('time', 'latitude')
    else:
        interp = interp.transpose('level', 'time', 'latitude')
    
    return interp
    
###########################################################
# Interpolation function for 2D latitude-longitude fields #
###########################################################
def interpleapyear_2d(data, year):

    n_days = len(data.time)
    print(str(year) + ' is a leap year or does not have 365 time coordinates!')
    
    orig = np.linspace(0, 1, n_days)
    new = np.linspace(0, 1, 365)
    
    def interp_1d(timeseries):
        f = interp1d(orig, timeseries, kind='linear')
        return f(new)
    
    interp = xr.apply_ufunc(
        interp_1d,
        data,
        input_core_dims=[['time']],
        output_core_dims=[['new_time']],     
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes={'new_time': 365}       
    )
    
    # Rename and reassign coordinates
    interp = interp.rename({'new_time': 'time'})
    interp = interp.assign_coords({'time': np.arange(1, 366)})
    print(f"Year {year}: NEW interpolated shape = {interp.shape}")
    interpt = interp.transpose('time', 'latitude', 'longitude') # explicit ordering of dims
    print(f"Year {year}: TRANSPOSED interpolated shape = {interpt.shape}")
    
    return interpt
############################################################################################################
# Annual cycle computation for zonal means with interpolation. The files must be named like {name}{year}.nc 
# and must be in a directory with yearly datasets of that variable.
# Example: meantemp2005.nc, MSLP2005.nc, etc. that are in a directory for, say, meantemp from 1979-2020.
# Originally, I used this function for files like zmzdaily2005.nc, meaning "Daily zonal mean of Z for 2005"
# Dimensions of the dataset should be level, time, latitude
############################################################################################################



##########################################################################
# General ann cyc#

def annual_cycle(filepath, name, varname, start, end, surface=False):
    yeardata = []
    level_coords = None
    lat_coords = None
    for year in range(start, end + 1):
        file_path = f"{filepath}/{name}{year}.nc"

        ds = xr.open_dataset(file_path)
        data = ds[varname].load()
        print(f"Year {year}: original shape = {data.shape}")

        # Storing coords for a non-leap year
        if lat_coords is None:
            lat_coords = data.latitude
        if not surface and level_coords is None:
            level_coords = data.level
            
        if year == 1979:
            if surface:
                data = data.transpose('time','latitude')
            else:
                data = data.transpose('level','time','latitude')
                
        if data.shape[1 if not surface else 0] != 365:
            interped = interpleapyear(data, year, surface=surface)
        else:
            interped = data

        yeardata.append(interped)

       

    clim_time = np.arange(1,366)
    totalyears = np.stack(yeardata, axis=0)
    annualcycle_val = np.mean(totalyears, axis=0)
    
    if surface:
        annualcycle = xr.DataArray(annualcycle_val, dims=['time', 'latitude'], coords={'time': clim_time, 'latitude': lat_coords}, name=varname)
    else:
        annualcycle = xr.DataArray(annualcycle_val, dims=['level', 'time', 'latitude'], coords={'level': level_coords, 'time': clim_time, 'latitude': lat_coords}, name=varname)
        
    return annualcycle

#######################################################################
# Annual cycle for 2D spatial fields
#######################################################################
def annual_cycle_2d(filepath, name, varname, start, end, outputpath):

    yeardata = []
    lat_coords = None
    lon_coords = None
    
    for year in range(start, end + 1):
        file_path = f"{filepath}/{name}{year}.nc"
        ds = xr.open_dataset(file_path)
        data = ds[varname]
        data.transpose('time', 'latitude', 'longitude')
        print(f"Year {year}: original shape = {data.shape}")
        
        # Store coords from first file
        if lat_coords is None:
            lat_coords = data.latitude
            lon_coords = data.longitude
        
        # Handle leap years
        if data.shape[0] != 365:
            interped = interpleapyear_2d(data, year)
        else:
            interped = data
        
        yeardata.append(interped)
        
    clim_time = np.arange(1, 366)
    totalyears = np.stack(yeardata, axis=0)  # Stack along new axis (year, time, lat, lon)
    annualcycle_val = np.mean(totalyears, axis=0)  # Average over years

    del totalyears
    gc.collect()
    
    annualcycle = xr.DataArray(
        annualcycle_val, 
        dims=['time', 'latitude', 'longitude'], 
        coords={'time': clim_time, 'latitude': lat_coords, 'longitude': lon_coords}, 
        name=varname
    )
    
    print(f"Computing and writing annual cycle to {outputpath}")
    annualcycle.to_netcdf(outputpath)
    print("Done!")

    del annualcycle
    gc.collect()
    
    return None


# Quick function using regex to extract year from filename
def extract_year(filename):
    # Match 4 digits at the end of basename (before .nc)
    match = re.search(r'(\d{4})\.nc', filename)
    return match.group(1) if match else None

#######################################################################################################
# Anomaly field calculation function. Simply subtracts the given annual cycle from the dataset provided. 
# Accepts two arguments: the annual cycle, which should be in the format of a DataArray, while for the 
# yeardata it should be a path + filename. I plan to make conditionals to automatically convert datasets
# to DataArrays. Just a bit more flexibility in the future. 
#######################################################################################################

def calcanomaly(anncyc, yeardata, varname, surface=False):
    #anncycl =  xr.open_mfdataset(anncyc)[varname]
    anncycl = anncyc[varname]
    yearstr = extract_year(yeardata)
    yearda = xr.open_mfdataset(yeardata)[varname]
    cheese = np.arange(1,366)
    if yearda.shape[yearda.dims.index('time')] != 365:
        interped = interpleapyear(yearda, yearstr, surface=surface).assign_coords({'time': cheese})
    else:
        cheese = np.arange(1,366)
        interped = yearda.assign_coords({'time': cheese})
        
    anomalyda = interped - anncycl
    return anomalyda

# stacking for yearly dzm anomalies
def stackanoms(name, var, surface=False):
    anoms = []
    for year in range(1979, 2024):
        ds = xr.open_mfdataset(f'/glade/work/leonardgu/tam/anoms/{name}_anomalies/{name}dailyanom{year}.nc')
        data = ds[list(ds.data_vars)[0]]
        #data.name = var
        if surface:
            data = data.transpose('time', 'latitude')
        else:
            data = data.transpose('level', 'latitude', 'time')
        data = data.assign_coords({'year': year})
        
        anoms.append(data)
    
    danom1979_2023 = xr.concat(anoms, dim='year')
    danom1979_2023.name = var
    return danom1979_2023

#######################################################################
# Calculate anomalies for 2D spatial fields ##########################
#######################################################################
def calcanomaly_2d(anncyc, yeardata, varname):

     # Load annual cycle if it's a path
    if isinstance(anncyc, str):
        anncycl = xr.open_dataset(anncyc)[varname]
    else:
        anncycl = anncyc[varname]
    yearstr = extract_year(yeardata)
    yearda = xr.open_mfdataset(yeardata)[varname]
    
    cheese = np.arange(1, 366)
    
    # Handle leap years
    if yearda.shape[yearda.dims.index('time')] != 365:
        interped = interpleapyear_2d(yearda, yearstr).assign_coords({'time': cheese})
    else:
        interped = yearda.assign_coords({'time': cheese})
    
    anomalyda = interped - anncycl

    del interped, yearda, anncycl
    gc.collect()
    
    return anomalyda

#######################################################################
# Stack anomalies across years for 2D spatial fields 
#######################################################################
def stackanoms_2d(name, var):

    anoms = []
    
    for year in range(1979, 2024):
        ds = xr.open_mfdataset(f'/glade/work/leonardgu/tam/anoms/{name}_anomalies/{name}dailyanom{year}.nc')
        data = ds[list(ds.data_vars)[0]]
        
        data = data.transpose('time', 'latitude', 'longitude')
        data = data.assign_coords({'year': year})
        
        anoms.append(data)
        ds.close()
        
    danom1979_2023 = xr.concat(anoms, dim='year')
    danom1979_2023.name = var
    
    return danom1979_2023
    
####################################################################################################################
# Cross correlation function of two timeseries, accepts the 850 hPa timeseries (or whatever SINGLE LEVEL timeseries)
# and the DataArray of the timeseries of all levels of the same variable
####################################################################################################################
def level_lag_corr(v850, vp):
    ts850 = v850.values
    tsp = vp.values    
    max_lag = 40
    lags = np.arange(-max_lag, max_lag + 1)

    # this handles the surface case e.g. tsp.ndim means that it only has one level
    if tsp.ndim == 1:
        correlations = np.zeros(len(lags))
        corr = correlate(tsp, ts850, mode='same', method='auto')
        corr = corr/(np.std(tsp)*np.std(ts850)*len(ts850))
        center = len(corr) // 2
        correlations = corr[center - max_lag : center + max_lag + 1]
        
        corr_da = xr.DataArray(
            correlations,
            coords={'lag': lags},
            dims=['lag'],
            name='correlation'
        )
        return corr_da
    else:
        levels = vp.level.values
        correlations = np.zeros((tsp.shape[0], len(lags)))

        for i in range(tsp.shape[0]):
            corr = correlate(tsp[i, :], ts850, mode='same', method='auto')
            corr = corr/(np.std(tsp[i, :])*np.std(ts850)*len(ts850))
            center = len(corr) // 2
            correlations[i, :] = corr[center - max_lag : center + max_lag + 1]

        corr_da = xr.DataArray(
            correlations,
            coords={
                'level': levels,
                'lag': lags
            },
            dims=['level', 'lag'],
            name='correlation'
    )
    
    return corr_da
        
####################################################################
####################################################################
def lat_lag_corr(v850, vp):
    ts850 = v850.values
    tsp = vp.values    
    latitudes = vp.latitude.values
    
    max_lag = 40
    lags = np.arange(-max_lag, max_lag + 1)
    
    correlations = np.zeros((len(latitudes), len(lags))) # just a blank zero array of the dims
    
    for i in range(len(latitudes)):
        corr = correlate(tsp[i, :], ts850, mode='same', method='auto')
        corr = corr / (np.std(tsp[i, :]) * np.std(ts850) * len(ts850))
        
        center = len(corr) // 2
        correlations[i, :] = corr[center - max_lag : center + max_lag + 1]
    
    # Create DataArray with proper coordinates
    corr_da = xr.DataArray(
        correlations,
        coords={
            'latitude': latitudes,
            'lag': lags
        },
        dims=['latitude', 'lag'],
        name='correlation'
    )
    
    return corr_da
##########################################################
##########################################################

# full 3d corr of the correlation matrix
def lag_corr(v850, vp):
    """
    Compute lag correlations between a reference timeseries (v850) and 
    a 3D spatial field (vp) with dimensions (level, latitude, time).
    
    Returns an xarray DataArray with dimensions (level, latitude, lag)
    """
    ts850 = v850.values
    tsp = vp.values    
    latitudes = vp.latitude.values
    levels = vp.level.values
    
    max_lag = 40
    lags = np.arange(-max_lag, max_lag + 1)
    
    correlations = np.zeros((len(levels), len(latitudes), len(lags))) # zeros of the dims
    
    for k in range(len(levels)):
        for i in range(len(latitudes)):
            corr = correlate(tsp[k, i, :], ts850, mode='same', method='auto')
            corr = corr / (np.std(tsp[k, i, :]) * np.std(ts850) * len(ts850))
            
            center = len(corr) // 2
            correlations[k, i, :] = corr[center - max_lag : center + max_lag + 1]
    
    # Create DataArray with proper coordinates
    corr_da = xr.DataArray(
        correlations,
        coords={
            'level': levels,
            'latitude': latitudes,
            'lag': lags
        },
        dims=['level', 'latitude', 'lag'],
        name='correlation'
    )
    
    return corr_da
    
##########################################################
# Corr for 2D spatial fields with a reference timeseries #
##########################################################

def lag_corr_2d(index, field, lag, temp_dir, lat_chunk_size=50):
    """
    Compute lagged correlation in latitude chunks to manage memory
    """
    # Shift timeseries based on lag
    if lag > 0:
        idx_slice = slice(lag, None)
        field_slice = slice(None, -lag)
    elif lag < 0:
        idx_slice = slice(None, lag)
        field_slice = slice(-lag, None)
    else:
        idx_slice = slice(None)
        field_slice = slice(None)
    
    index_shifted = index.isel(alltime=idx_slice)
    field_shifted = field.isel(alltime=field_slice)
    
    # Standardize index once
    index_anom = (index_shifted - index_shifted.mean()) / index_shifted.std()
    n = len(index_anom)
    
    os.makedirs(temp_dir, exist_ok=True)
    n_lats = len(field_shifted.latitude)
    chunk_files = []
    
    for lat_start in range(0, n_lats, lat_chunk_size):
        lat_end = min(lat_start + lat_chunk_size, n_lats)
        print(f"Processing lats {lat_start}-{lat_end}")
        
        # Load this latitude chunk
        field_chunk = field_shifted.isel(latitude=slice(lat_start, lat_end)).load()
        
        # Standardize field chunk
        field_anom = (field_chunk - field_chunk.mean('alltime')) / field_chunk.std('alltime')
        
        # Compute correlation
        corr_chunk = (field_anom * index_anom).sum('alltime') / (n - 1)
        
        # Save to temp file
        chunk_file = os.path.join(temp_dir, f'corr_chunk_{lat_start}.nc')
        corr_chunk.to_netcdf(chunk_file)
        chunk_files.append(chunk_file)
        
        del field_chunk, field_anom, corr_chunk
        gc.collect()
    
    print("Concatenating correlation chunks...")
    corr = xr.open_mfdataset(chunk_files, combine='nested', concat_dim='latitude')
    
    return corr, chunk_files
#########################################################################################
# 1 year highpass Butterworth Filter for Daily Zonal Mean Anomalies. 
# Accepts a timeseries and returns both the highpass and lowpass components of the signal. 
#########################################################################################
def butter1yhighp(orig, order):
    cutoff_period = 365  
    sampling_rate = 1    # samples per day
    nyquist_freq = sampling_rate / 2
    
    # Cutoff frequency (normalized to Nyquist) 
    cutoff_freq = (1/cutoff_period)/nyquist_freq
    
     # discuss later
    b, a = butter(order, cutoff_freq, btype='high')
    
     # Check for NaNs
    if np.isnan(orig.values).any():
        print(f"WARNING: Found {np.isnan(orig.values).sum()} NaNs in input data")
        orig_filled = orig.interpolate_na(dim='alltime', method='linear')
        filtda = orig_filled.values
    else:
        filtda = orig.values
        
    if filtda.ndim == 1:
        high_freq_vals = filtfilt(b, a, filtda)
    else:
        time_axis = orig.dims.index('alltime')
        high_freq_vals = filtfilt(b, a, filtda, axis=time_axis)
    
    high_freq = xr.DataArray(
        high_freq_vals,
        coords=orig.coords,
        dims=orig.dims,
        attrs=orig.attrs  
    )
    low_freq = orig - high_freq

    return high_freq, low_freq

####################################################################
# 1 year highpass Butterworth filter for 2D spatial anomaly fields #
####################################################################

def butter1yhighp_2d(orig, order, temp_dir, lat_chunk_size=50):
    cutoff_period = 365  
    sampling_rate = 1
    nyquist_freq = sampling_rate / 2
    cutoff_freq = (1/cutoff_period)/nyquist_freq
    b, a = butter(order, cutoff_freq, btype='high')
    
    # Reset MultiIndex if present
    if isinstance(orig.indexes.get('alltime'), pd.MultiIndex):
        orig = orig.reset_index('alltime')
    
    os.makedirs(temp_dir, exist_ok=True)
    
    n_lats = len(orig.latitude)
    chunk_files_high = []
    chunk_files_low = []
    
    for lat_start in range(0, n_lats, lat_chunk_size):
        lat_end = min(lat_start + lat_chunk_size, n_lats)
        print(f"Processing lats {lat_start}-{lat_end}")
        
        chunk = orig.isel(latitude=slice(lat_start, lat_end)).load()
        
        if np.isnan(chunk.values).any():
            chunk = chunk.interpolate_na(dim='alltime', method='linear')
        
        def filter_timeseries(ts):
            return filtfilt(b, a, ts)
        
        high_chunk = xr.apply_ufunc(
            filter_timeseries, chunk,
            input_core_dims=[['alltime']], 
            output_core_dims=[['alltime']],
            vectorize=True, 
            output_dtypes=[float]
        )
        
        low_chunk = chunk - high_chunk
        
        high_file = os.path.join(temp_dir, f'temp_high_{lat_start}.nc')
        low_file = os.path.join(temp_dir, f'temp_low_{lat_start}.nc')
        high_chunk.to_netcdf(high_file)
        low_chunk.to_netcdf(low_file)
        chunk_files_high.append(high_file)
        chunk_files_low.append(low_file)
        
        del chunk, high_chunk, low_chunk
        gc.collect()
    
    print("Concatenating chunks...")
    high_freq = xr.open_mfdataset(chunk_files_high, combine='nested', concat_dim='latitude')
    low_freq = xr.open_mfdataset(chunk_files_low, combine='nested', concat_dim='latitude')
    
    return high_freq, low_freq, chunk_files_high, chunk_files_low

"""
high, low, h_files, l_files = butter1yhighp_2d_chunked(
    z850_anoms, 
    order=5, 
    temp_dir='/glade/scratch/leonardgu/temp_filter/'
)

high.to_netcdf('z850_high.nc')
low.to_netcdf('z850_low.nc')

# Cleanup
for f in h_files + l_files:
    os.remove(f)
"""

# --------------- function to compute signifiance on timeseries correlation matrices ----------- #
def compute_significance_mask(corr_da, N_eff, alpha=0.05):
    """
    Pointwise two-sided t-test for correlation coefficients.
    Returns boolean mask, True where significant at (1-alpha) confidence.

    Parameters
    ----------
    corr_da : xr.DataArray 
    N_eff   : int -- effective sample size (N_total / tau)
    alpha   : float -- significance level, default 0.05

    Returns
    -------
    xr.DataArray (bool), same dims as corr_da
    """
    r   = corr_da.values
    df  = N_eff - 2 # minus 2 to the degrees of freedom because pearson correlation

    # Clip to avoid division by zero at r = \pm 1
    r_safe = np.clip(r, -0.9999, 0.9999)

    t_stat = r_safe * np.sqrt(df / (1 - r_safe**2)) # t statistic calc
    p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df)  # get p values 

    return xr.DataArray(
        p_vals < alpha,
        coords=corr_da.coords,
        dims=corr_da.dims
    )

# ------------- Plot correlation with sig hatching ------------------------
def plot_corr_with_significance(ax, corr_da, tau, total_days=16059, alpha=0.05,
                                yscale='linear', invert_y=False,
                                yticks=None, yticklabels=None,
                                vmin=None, vmax=None, cmap='RdBu_r',
                                xlabel='Lag (days)', ylabel=''):
    """
    Generalized correlation plot with significance stippling.
    Works for any 2D corr DataArray with lag as one dimension.

    Parameters
    ----------
    ax           : matplotlib Axes
    corr_da      : xr.DataArray, dims (y_dim, lag)
    tau          : int -- decorrelation timescale in days (e.g. 40 for omega/GPH, 8 for precip)
    total_days   : int -- total days in dataset (default 16059)
    alpha        : float -- significance level (default 0.05)
    yscale       : 'linear' or 'log'
    invert_y     : bool -- True for pressure axis
    yticks       : array-like, optional explicit ticks
    yticklabels  : array-like, optional tick labels
    vmin/vmax    : floats, optional -- defaults to symmetric about 0
    cmap         : str, colormap (default 'RdBu_r')
    xlabel       : str
    ylabel       : str

    Returns
    -------
    cf : contourf object (for attaching colorbar externally)
    """
    N_eff = int(total_days / tau)

    lags = corr_da.lag.values
    # Infer y-dim as whichever dim is not lag
    y_dim = [d for d in corr_da.dims if d != 'lag'][0]
    y_vals = corr_da[y_dim].values

    # Symmetric norm by default
    if vmin is None or vmax is None:
        vmax = float(max(abs(corr_da.min()), abs(corr_da.max())))
        vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)

    # Correlation fill
    cf = ax.contourf(lags, y_vals, corr_da.values,
                     levels=levels, norm=norm, cmap=cmap)

    # Significance mask -- stipple where NOT significant
    sig_mask = compute_significance_mask(corr_da, N_eff, alpha=alpha)
    insig = ~sig_mask.values
    lag_grid, y_grid = np.meshgrid(lags, y_vals)
    ax.scatter(lag_grid[insig], y_grid[insig],
               s=10, c='black', marker='.', alpha=1, linewidths=0)

    # Axis formatting
    ax.set_yscale(yscale)
    if invert_y:
        ax.invert_yaxis()
    if yticks is not None:
        ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    return cf

# Latitude weighted averaging function #
def latavg(data, lat_min, lat_max):
        # Ensure lat_max > lat_min
    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    
    # Select latitude band (order doesn't matter with min/max)
    data_slice = data.sel(latitude=slice(lat_hi, lat_lo))
    
    # Create weights (cos of latitude)
    weights = np.cos(np.deg2rad(data_slice.latitude))
    
    # Use xarray's weighted method
    weighted_avg = data_slice.weighted(weights).mean(dim='latitude')
    
    return weighted_avg


# ------------- seasonal functions ---------------- #




################################################################################################################
# Tyler's functions ( provided by Tyler Greiner, ttg2015@nyu.edu) Functions to extract precip from mtpr. More generally, convert forecast arrays into single time dim ds
############################################################################################################
def convert_forecast_to_time(
    ds: xr.Dataset | xr.DataArray,
    time_name: str = "forecast_initial_time",
    step_name: str = "forecast_hour",
    drop_init: bool = True,
    remove_duplicates: bool = True,
) -> xr.Dataset | xr.DataArray:
    """
    Convert ERA5-style forecast arrays (init_time x step) into a single `time` dimension
    of forecast valid times.

    - Works with both xr.Dataset and xr.DataArray (returns same type as input).
    - If `step` values are numeric, they are interpreted as hours.
    - The resulting dimension is named 'time' (a real dimension), with a datetime-like
      coordinate attached to it.
    - Duplicate valid times (common when forecasts overlap) are removed if
      remove_duplicates=True (keeps the first occurrence).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input data with dims [forecast_initial_time, forecast_hour, ...].
    time_name : str
        Name of the forecast reference-time dim (default "forecast_initial_time").
    step_name : str
        Name of the forecast lead-time dim (default "forecast_hour").
    drop_init : bool
        Drop the original init and step variables/coords if True.
    remove_duplicates : bool
        Remove duplicate valid times (keep first occurrence).

    Returns
    -------
    xr.Dataset or xr.DataArray
        Converted data with a single `time` dimension.
    """
    input_was_da = isinstance(ds, xr.DataArray)
    # If DataArray, convert to Dataset to ease manipulation, keep original name
    if input_was_da:
        var_name = ds.name or "variable"
        ds = ds.to_dataset(name=var_name)

    # Basic checks
    if time_name not in ds.dims or step_name not in ds.dims:
        raise ValueError(f"Dataset must have both dims '{time_name}' and '{step_name}'")

    # Broadcast the init and step coordinates into full 2D arrays
    init_da, step_da = xr.broadcast(ds[time_name], ds[step_name])
    # At this point init_da and step_da have shape (n_init, n_step)

    # Extract 1D init and step vectors (we take first column/row of broadcasted arrays)
    init_1d = init_da[:, 0].values      # shape (n_init,)
    step_1d = step_da[0, :].values     # shape (n_step,)

    # --- Convert step_1d to numpy.timedelta64 array (interpreting numeric as hours) ---
    # If step is already timedelta64, keep it.
    if np.issubdtype(step_1d.dtype, np.timedelta64):
        step_td = step_1d
    # If step is numeric (int/float), interpret as hours
    elif np.issubdtype(step_1d.dtype, np.integer) or np.issubdtype(step_1d.dtype, np.floating):
        step_td = pd.to_timedelta(step_1d, unit="h").to_numpy()
    else:
        # Try pandas to_timedelta (handles strings like "1H", "00:30:00", etc.)
        try:
            step_td = pd.to_timedelta(step_1d).to_numpy()
        except Exception as exc:
            raise TypeError("Unsupported dtype for step values. Must be timedelta, numeric (hours), or parseable by pd.to_timedelta.") from exc

    # --- Build flattened valid-time array in the same order as xarray.stack will produce ---
    n_init = ds.sizes[time_name]
    n_step = ds.sizes[step_name]

    # Validate shapes
    if len(init_1d) != n_init or len(step_1d) != n_step:
        # fallback: use unique values from original coords
        init_1d = ds[time_name].values
        step_1d = ds[step_name].values
        if np.issubdtype(step_1d.dtype, np.timedelta64):
            step_td = step_1d
        else:
            step_td = pd.to_timedelta(step_1d, unit="h").to_numpy()

    # Attempt to ensure init_1d are numpy datetimes; if not, convert via pandas
    if not np.issubdtype(init_1d.dtype, np.datetime64):
        try:
            init_pd = pd.to_datetime(init_1d)
            init_vals = init_pd.to_numpy()
        except Exception:
            # If we cannot convert (e.g. cftime objects), raise explicit error with guidance.
            raise TypeError(
                "forecast_initial_time values are not numpy datetime64 and could not be coerced. "
                "This function currently expects numpy datetime64 (or values convertible by pandas.to_datetime). "
                "If you use cftime, convert times before calling this function or request cftime support."
            )
    else:
        init_vals = init_1d

    # Now create the flattened valid-time array: step varies fastest (stack time=[init,step])
    valid_flat = np.repeat(init_vals, n_step) + np.tile(step_td, n_init)
    # valid_flat is a 1D numpy array of dtype datetime64 (or compatible)

    # --- Stack the dataset so we get a single 'time' dimension ---
    stacked = ds.stack(time=(time_name, step_name))

    # Assign the computed 1D valid-times as the 'time' coordinate
    stacked = stacked.assign_coords(time=("time", valid_flat))

    # Sort by the new time coordinate
    stacked = stacked.sortby("time")

    # Optionally drop the original init/step coordinate variables (if present)
    if drop_init:
        for v in (time_name, step_name):
            if v in stacked.coords:
                stacked = stacked.drop_vars(v)

    # Optionally remove duplicate valid times (keep first occurrence)
    if remove_duplicates:
        # work on numpy values for speed and robustness
        times_arr = stacked["time"].values
        # np.unique preserves order only with return_index; we want first occurrences
        _, idx = np.unique(times_arr, return_index=True)
        idx_sorted = np.sort(idx)  # keep chronological order of first occurrences
        stacked = stacked.isel(time=idx_sorted)

    # Return same type the user passed in
    if input_was_da:
        return stacked[var_name]
    else:
        return stacked


# ---------------- Sped up profile selections by Tyler Greiner  --------------------------------
def select_loc_time_space(ds, latitude, longitude, time):
    return(ds.sel(latitude = latitude, longitude = longitude, time = time, method = 'nearest'))

def load_loc_time_space(paths, var, lati, loni, timei):
    return(
        xr.open_mfdataset(
            paths,
            combine='by_coords',
            preprocess=lambda ds: select_loc_time_space(ds,lati, loni, timei),

    )[var])



def select_loc_space(ds, latitude, longitude):
    return(ds.sel(latitude = latitude, longitude = longitude, method = 'nearest'))

def load_loc_space(paths, var, lati, loni):
    return(
        xr.open_mfdataset(
            paths,
            combine='by_coords',
            preprocess=lambda ds: select_loc_space(ds,lati, loni),

    )[var])

def select_loc_space_level(ds, latitude, longitude, level):
    return(ds.sel(latitude = latitude, longitude = longitude, method = 'nearest'))

def load_loc_space_level(paths, var, lati, loni, level):
    return(
        xr.open_mfdataset(
            paths,
            combine='by_coords',
            preprocess=lambda ds: select_loc_space(ds,lati, loni),

    )[var])

# Calculate the specific humidity
# Some nice functions
def e_sat(t): # [K][Pa]
    """ 
    This function will take a provided temperature in kelvin 
    and return the saturated vapore pressure in Pascals
    """
    return 611.21*np.exp(17.502*((t-273.16)/(t-32.19)))

def e2q(e,p):
    """
    This function takes the surface pressure as well as the
    current vapor pressure to calculate the specific humidity 
    """
    return 0.621981 * e/(p-(1-0.621981)*e)
