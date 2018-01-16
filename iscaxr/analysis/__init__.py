import numpy as np
import xarray as xr

from iscaxr.constants import grav, Rad_earth, kappa

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
    vbar = data[v_field].mean('lon')
    c = 2*np.pi*a*np.cos(vbar.lat*np.pi/180) / g
    # take a diff of half levels, and assign to pfull coordinates
    dp = xr.DataArray(data.phalf.diff('phalf').values*100, coords=[('pfull', data.pfull)])
    return c*(vbar*dp).cumsum(dim='pfull')

def pot_temp(data, p0=None, temp_field='temp', kappa=kappa):
    """Calculate potential temperature from an Isca DataSet.

    theta = T (p0/p)**kappa
    kappa = R/cp

    Parameters
    ----------
    data : xarray.DataSet
        Isca output data
    p0 : float, optional
        Reference pressure for potential temperature.  If None, use surface
        pressure determined from max of data.phalf.
    temp_field : str, optional
        The name of the temperature field.  Default: 'temp'
    kappa : float, optional
        Value for exponent R/cp.  Default: dry air, 2/7.

    Returns a DataArray `pot_temp`."""
    if p0 is None:
        p0 = data.phalf.max()
    t = data[temp_field]
    p = data.pfull
    theta =  t * (p0 / p) ** kappa
    theta.name = 'pot_temp'
    return theta

