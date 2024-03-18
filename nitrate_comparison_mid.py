import numpy as np
from netCDF4 import Dataset,num2date
from numpy import *
import glob
import xarray as xr
import os.path
import sys

print ("The script has the name %s" % (sys.argv[0]))
arguments = len(sys.argv) - 1
print ("The script is called with %i arguments" % (arguments))
i=int(sys.argv[1])

# Load observation data 
filename = "/home/fid000/WORK7/ANALYSIS/model_evaluation/DATA/Argo/nitrate_argo_monthly_"+str(i)+"_CREG025.nc"
creg = xr.open_mfdataset(filename, format="NETCDF4")
obs = creg['nitrate_adjusted'][:,:,:]
nav_lon = creg['nav_lon'][:,:].values
nav_lat = creg['nav_lat'][:,:].values
month = [str(i)]

obs = obs[:,:,:].values

# Load model data
flist = glob.glob("/home/fid000/WORK7/ANALYSIS/model_evaluation/DATA/VJC007b/CDF/CREG025_LIM3_CANOE-VJC007b_5d_ptrc_T_2016*nc")                                                            
flist.sort() 
model = xr.open_mfdataset(flist)
model_monthly_means = model.groupby('time_counter.month').mean()

model = model_monthly_means['NO3'][i-1,:,:,:].values

# Isolate depth variable
depth = model_monthly_means['deptht'].values

# Model, shallow
ld, ly, lx = np.shape(model)
bnd = (np.where((depth >= 100) & (depth < 500)))
bz_m = np.zeros([ly,lx])
bz_o = np.zeros([ly,lx])

for ix in np.arange(lx):
        for iy in np.arange(ly):

            ym = model[bnd[0],iy,ix].copy()
            yo = obs[bnd[0],iy,ix].copy()
            ym[ym==0] = np.nan
            mask = (~np.isnan(ym)) & (~np.isnan(yo))
            #print(np.shape(ym), np.shape(yo))
            #print(np.shape(mask), np.shape(depth))
            yo = yo[mask]; ym = ym[mask]
            d = depth[bnd[0]]
            d = d[mask]

            if (len(ym) > 1) & (sum(ym) > 0):
                b_o = np.trapz(y = yo, x = d)
                bz_o[iy, ix] = b_o
                b_m = np.trapz(y = ym, x = d)
                bz_m[iy, ix] = b_m
            #print("min az_o",np.min(az_o),"min az_m", np.min(az_m),"max az_o", np.max(az_o),"max az_m", np.max(az_m))

print("Analysis done, writing to netcdf now")

outfile_obs = "/home/fid000/WORK7/ANALYSIS/model_evaluation/DATA/mid_obs_2016month"+str(i)+".nc"
outfile_model = "/home/fid000/WORK7/ANALYSIS/model_evaluation/DATA/mid_model_2016month"+str(i)+".nc"

ds = xr.Dataset(
    data_vars=dict(
        NO3_m=([ "y", "x"], bz_m),
        NO3_o=([ "y", "x"], bz_o),
    ),
    coords=dict(
        nav_lon=(["y", "x"], nav_lon),
        nav_lat=(["y", "x"], nav_lat),
    ),
    attrs=dict(description="monthly ARGO nitrate integrations on CREG"),
 )
print("NO3_m", np.max(ds.NO3_m))
ds.NO3_o.attrs['units'] = "mmol/m3"
ds.NO3_m.attrs['units'] = "mmol/m3"
ds.nav_lat.attrs['units'] = "degrees_north"
ds.nav_lon.attrs['units'] = "degrees_east"
comp = dict(zlib=True, complevel=1)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(outfile_obs)#,unlimited_dims='time', encoding=encoding)
ds.to_netcdf(outfile_model)#,unlimited_dims='time', encoding=encoding)
