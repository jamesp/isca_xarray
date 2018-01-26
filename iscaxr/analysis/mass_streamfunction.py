import numpy as np
import xarray as xr

from iscaxr.constants import grav, Rad_earth

def mass_streamfunction(data, v_field='vcomp', a=Rad_earth, g=grav):
    """Calculate the mass streamfunction for the atmosphere.

    Based on a vertical integral of the meridional wind.
    Ref: Physics of Climate, Peixoto & Oort, 1992.  p158.

    Parameters
    ----------
    data :  xarray.DataSet
        Isca output data
    v_field : str, optional
        The name of the meridional flow field in `data`.  Default: 'vcomp'
    a : float, optional
        The radius of the planet. Default: Earth 6317km
    g : float, optional
        Surface gravity. Default: Earth 9.8m/s^2

    Returns
    -------
    streamfunction : xarray.DataArray
        The meridional mass streamfunction.
    """
    if 'lon' in data[v_field].dims:
        vbar = data[v_field].mean('lon')
    c = 2*np.pi*a*np.cos(vbar.lat*np.pi/180) / g
    # take a diff of half levels, and assign to pfull coordinates
    dp = xr.DataArray(data.phalf.diff('phalf').values*100, coords=[('pfull', data.pfull)])
    return c*(vbar*dp).cumsum(dim='pfull')
