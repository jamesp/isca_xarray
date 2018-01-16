import numpy as np
import scipy.signal
import scipy.interpolate
import xarray as xr

from iscaxr.util import rng

def calculate_dlatlon(domain):
    """Calculate the grid size, in radians, for a dataset.

    Parameters
    ----------
    domain : IscaDataSet (xarray.DataSet)

    Returns
    -------
    dlat, dlon : xarray.DataArray, xarray.DataArray
        dlat, dlon in radians for the grid.  Indexed by `lat` and `lon`.
    """
    rad = np.pi / 180
    coslat = np.cos(domain.lat * rad)

    dlon = (domain.lonb*rad).diff('lonb')
    dlon.name = 'dlon'
    dlon = dlon.rename({'lonb': 'lon'})
    dlon['lon'] = domain.lon.values

    dlat = (domain.latb*rad).diff('latb')
    dlat.name = 'dlat'
    dlat = dlat.rename({'latb': 'lat'})
    dlat['lat'] = domain.lat.values

    return dlat, dlon

def calculate_dA(domain):
    rad = np.pi / 180
    coslat = np.cos(domain.lat * rad)
    dlat, dlon = calculate_dlatlon(domain)

    dA = dlat*dlon*coslat
    return dA

def make_surf_integrator(domain, radius=1.0):
    """Generate a surface integrator.

    For a given Isca domain on a sphere of given `radius`, returns
    a function that calculates the area weighted sum
    of a scalar field over latitude and longitude.

    Parameters
    ----------
    domain : IscaDataSet : xarray.DataSet
        The domain on which fields are discretised.  Should have coordinates
            `lat`, `latb`, `lon`, `lonb`.
    radius : float, optional (default=1.0)
        The radius of the sphere

    Returns
    -------
    integrator : function
        A function that can be applied to xarray.DataArrays to reduce
        the `lat` and `lon` dimensions.
    """
    dA = calculate_dA(domain)
    def integrator(field):
        return radius**2*(field*dA).sum(('lat', 'lon'))
    return integrator

def calculate_dp(domain):
    return xr.DataArray(domain.phalf.diff('phalf').values*100, coords=[('pfull', domain.pfull)])

def pfull_to_phalf(field, domain):
    """Move a field from pfull levels to
    phalf levels (except top and bottom) using the arithmetic mean."""
    # calculate the arithmetic mean between pairs of values along the pfull dim
    avg = field.rolling(pfull=2).mean().dropna('pfull')
    avg = _map_pfull_inside_phalf(avg, domain)
    return avg

def diff_pfull(field, domain):
    """Calculate the vertical derivative of a pfull field and map onto phalf levels"""
    dfield = field.diff('pfull')
    dfield.name = 'd{}'.format(field.name)
    dfield = _map_pfull_inside_phalf(dfield, domain)
    return dfield

def dfdp(field, domain):
    """Calculate d(field)/dpfull.

    Returns a new dfield on phalf levels."""
    dp = diff_pfull(domain.pfull*100, domain)
    df = diff_pfull(field, domain)
    return df/dp

def _map_pfull_inside_phalf(field, domain):
    # xarray thinks field is on the pfull coords, but values are at phalf now.
    # rename pfull -> phalf and replace the coord values with the domain phalf values.
    field = field.rename({'pfull': 'phalf'})
    field.phalf.values = domain.phalf[1:-1]
    return field


def resample_latlon(field, nlat=None, nlon=None, lats=None, lons=None, method='interpolate'):
    if nlat is None:
        nlat = len(field.coords['lat'])//2
    if nlon is None:
        nlon = len(field.coords['lon'])//2
    dims = field.coords.dims
    ilat = dims.index('lat')
    ilon = dims.index('lon')
    lat = field.coords['lat'].values
    lon = field.coords['lon'].values
    if lats is None:
        minlat, maxlat = rng(lat)
        newlat = np.linspace(minlat, maxlat, nlat)
    else:
        newlat = np.asarray(lats)
    if lons is None:
        minlon, maxlon = rng(lon)
        newlon = np.linspace(minlon, maxlon, nlon)
    else:
        newlon = np.asarray(lons)
    if method == 'interpolate':
        # resample lon form in fourier space
        lon_scale, newlon = scipy.signal.resample(field.values, nlon, t=lon, axis=ilon)
        # resample lat using interpolator
        f = scipy.interpolate.interp1d(field.coords['lat'].values, lon_scale, axis=ilat)
        rescaled = f(newlat)
        newcoords = [field.coords[d].values for d in dims]
        newcoords[ilat] = newlat
        newcoords[ilon] = newlon
        return xr.DataArray(rescaled, coords=newcoords, dims=dims, name=field.name)
    elif method == 'nearest':
        rescaled = field.sel(lat=newlat, lon=newlon, method='nearest')
        return rescaled
    else:
        raise AttributeError('unknown resampling method %r' % method)

def center_lon(field, wrap=False, nearest=False, **kwargs):
    """Redefine longitude coordinates with a new origin.

    Parameters
    ----------
    field : The xarray field to be rebased
    wrap : boolean, optional
        If `True`, centre the new longitude axis so that it ranges from [-180, 180].
        If `False` (default), recentred axis ranges [0, 360].
    nearest : boolean, optional
        If `True`, find the nearest lon value in the field and use that.  Ensures
        that lon coordinate labels are the same as input data.
    **kwargs : dict
        The longitude axes and new origins for the field.  e.g. lon=180.

    Returns
    -------
    recentred : xarray.DataArray
        A copy of `field` with the origin of lon axis moved.
    """
    q  = field.copy(deep=False)
    for dim in kwargs:
        if nearest:
            lon0 = q[dim].sel(method='nearest', **{dim:kwargs[dim]}).values
        else:
            lon0 = kwargs[dim]
        newlon = field[dim] - lon0
        newlon = np.mod(newlon+360.0, 360.0)
        if wrap:
            newlon[newlon > 180] = (newlon - 360)[newlon > 180]
        q[dim] = newlon
        ilon = q[dim].argsort()
        q = q.isel(**{dim: ilon})
    return q