# -*- coding:utf-8 -*-

import numpy as np
import xarray as xr


def window_taper(field, n=30, dim='time'):
    """Taper the ends of a field.  Useful for making non-periodic signals
    e.g. a partial time-series, and making them periodic for frequency analysis.
    Best to use eddy fields for this, i.e. remove mean value first.

    Parameters
    ----------
        dim : the dimension to taper along
        n : the size of taper at each end

    Returns the field, tapered to zero at each end of the given dimension.
    """
    taper = field[dim]*0 + 1
    taper[:n] = (np.cos(np.linspace(-np.pi/2, 0, n))**2)
    taper[-n:] = (np.cos(np.linspace(0, np.pi/2, n))**2)
    return field*taper

def fft(field, dim=None, axis=None, scaledim=None):
    """Calculate the FFT of a field along given dimensions.

    Parameters
    ----------

    dim : str or sequence of str, optional
        Dimension(s) over which to apply the transform.
    axis : int or sequence of int, optional
        Axis(es) over which to apply the transform. Only one of the 'dim'
        and 'axis' arguments can be supplied. If neither are supplied, then
        the transform is calculated over axes.
    scaledim : str or sequence of str, optional
        Dimensions for which the transformed coordinates should be scaled
        according to the input coordinates.  If not specified, transformed
        coordinates are given by whole wavenumbers over the given coordinate span.

    Returns
    -------

    transformed : xarray.DataArray
        The (complex) Fourier Transformed data.  Coordinate dimensions that have been transormed
        are named 'F_`dim`', other dimensions retain their original names.
        The transformed data array has the same shape as the original DataArray.
    """
    if dim is not None and axis is not None:
        raise ValueError("cannot supply both 'axis' and 'dim' arguments")

    if dim is not None:
        axis = np.atleast_1d(field.get_axis_num(dim))

    if scaledim is None:
        scaleaxis = []
    else:
        scaleaxis = np.atleast_1d(field.get_axis_num(scaledim))

    data = np.fft.fftn(field.values, axes=axis)
    data = np.fft.fftshift(data, axes=axis)
    transformed_axes = (range(field.ndim) if axis is None else axis % field.ndim)
    coords = []
    for ax, dim in enumerate(field.dims):
        if ax in transformed_axes:
            # assume a regular sample spacing, regardless of scaling
            if ax in scaleaxis:
                dx = field[dim].diff(dim).values[0]
            else:
                dx = 1./len(field[dim])
            tcoord = np.fft.fftshift(np.fft.fftfreq(len(field[dim]), dx))
            coords.append(('F_{}'.format(dim), tcoord))
        else:
            coords.append((dim, field[dim].values))
    return xr.DataArray(data=data, coords=coords)


def zonal_dispersion(field, dt=1):
    """Calculate the power spectra in time and longitude for an Isca dataset.

    Parameters
    ----------

        field : an Isca DataArray to transform.  At a minimum, must have 'time' and 'lon' dimensions.
        dt : Time interval, in days, between samples.

    Returns a xarray DataArray with dimensions:
        - 'time' replaced by 'freq', in cycles per day,
        - 'lon' replaced by 'k', in global wavenumber.
        - All other dimensions of `field` are preserved.
    """
    time_dim = field.dims.index('time')
    lon_dim = field.dims.index('lon')
    ft = np.fft.fft2(field, axes=(time_dim, lon_dim))
    # fourier transform in numpy is defined by exp(-2π i (kx + wt))
    # but we want exp(kx - wt) so need to negate the x-domain
    ft = ft[::-1]
    ft = np.fft.fftshift(ft)

    # convert to a power spectra
    ft = np.abs(ft)

    om = np.fft.fftshift(np.fft.fftfreq(len(field.time), d=dt))
    k = np.fft.fftshift(np.fft.fftfreq(len(field.lon), d=1./len(field.lon)))

    coords = []
    for c in field.dims:
        if c == 'time':
            coords.append(('freq', om))
        elif c == 'lon':
            coords.append(('k', k))
        else:
            coords.append((c, field[c].data))

    ftr = xr.DataArray(ft, coords=coords)

    # keep only the positive frequency domain, the other have of the spectrum
    # is identical as input was a real number signal
    ftr = ftr.sel(freq=slice(0, None))
    return ftr

def equatorial_waves(field, lat_cutoff=8, symmetric=True):
    """Calculate zonal equatorial wavenumbers.

    Either symmetric or antisymmetric waves.

    Returns the wave spectra in the zonal direction."""
    nh = field.sel(lat=slice(0, lat_cutoff))
    sh = field.sel(lat=slice(-lat_cutoff, 0))
    sh.lat.values = -sh.lat.values

    if symmetric:
        sym_field = 0.5*(nh + sh)
    else:
        sym_field = 0.5*(nh - sh)

    return sym_field.pipe(fft, dim='lon').pipe(np.abs).mean('lat')