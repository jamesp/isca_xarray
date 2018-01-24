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