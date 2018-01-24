import numpy as np
import xarray as xr


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

    om = np.fft.fftshift(np.fft.fftfreq(len(e.time), d=dt))
    k = np.fft.fftshift(np.fft.fftfreq(len(e.lon), d=1./len(e.lon)))

    coords = []
    for c in field.dims:
        if c == 'time':
            coords.append(('freq', om))
        elif c == 'lon':
            coords.append(('k', k))
        else:
            coords.append((c, field[c].data))

    ftr = xr.DataArray(ft, coords=coords)

    return ftr