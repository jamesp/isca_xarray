from functools import partial

import numpy as np
import xarray as xr

import iscaxr.domain
from iscaxr.util import grid_var
from iscaxr.constants import Rad_earth

def sublon(time, omega, alpha, a=Rad_earth):
    """Calculate substellar point, given rotation rate (omega) and speed of
    the substellar point over the surface of the planet."""
    gamma = float(alpha)/a + omega
    sublon = (gamma - omega)*time.in_units('s') + np.pi
    sublon = (np.mod(sublon,2*np.pi)*180/np.pi)
    return sublon


def g_sublon(dataset, omega, alpha, a=Rad_earth):
    """Generate a sublon(t) function that returns the substellar longitude
    at given time t, for a set of experimental parameters."""
    lon = dataset.lon.copy()
    def msublon(t):
        return grid_var(sublon(t, omega, alpha, a), lon)
    return msublon


def pipeline(field, fns):
    """Take xarray `field`, and pipe through all the given `fns` in order."""
    _field = field
    for fn in fns:
        _field = _field.pipe(fn)
    return _field

def piper(*fns):
    """Return a pipeline of the given functions."""
    return partial(pipeline, fns=fns)

def centre_zero(field):
    if 'lon' in field.dims:
        return iscaxr.domain.center_lon(field, lon=0, wrap=True)
    else:
        return iscaxr.domain.center_lon(field, lon0=0, wrap=True)

surface = lambda f: f.sel(pfull=f.pfull.max())

def norm(field):
    if 'lon' in field.dims:
        return iscaxr.util.normalize(field, dims='lon')
    else:
        return iscaxr.util.normalize(field, dims='lon0')

def lon_to_xi(field, lon0, wrap=True):
    """Move a DataArray from a fixed (lat, lon) frame of reference
    to (lat, substellar lon) that moves with the forcing."""
    if field.time.shape:  # is an array, iterate over recursively(!)
        return xr.concat([lon_to_xi(field.sel(time=t), lon0, wrap=wrap) for t in field.time], dim=field.time)
    else:                 # single snapshot in time
        lon0 = lon0(field.time) if callable(lon0) else lon0
        return  field.pipe(iscaxr.domain.center_lon, lon=lon0, wrap=wrap)#.rename({'lon': 'xi'})

def make_phase_curve_calculator(domain, radius=Rad_earth):
    rad  = np.pi / 180
    radlon = domain.lon * rad
    lon0 = radlon.copy(deep=True)
    lon0.name = 'lon0'
    lon0 = lon0.rename({'lon': 'lon0'})

    plon = np.cos(radlon - lon0)
    plon.values[plon.values < 0] = 0     # adjust for max(cos(lon - lon0), 0.0)

    coslat = np.cos(domain.lat * rad)
    dA = iscaxr.domain.calculate_dA(domain)
    def phase_curve(field):
        pc = (field*dA*coslat*plon).sum(('lat', 'lon'))
        return pc
    return phase_curve