import numpy as np
import xarray as xr

from iscaxr.analysis.spectral import equatorial_waves

def make_eq_signal(wavenum, power=1):
    lat = np.linspace(-10, 10, 10)
    lon = np.linspace(0, 360, 42, endpoint=False)
    signal = xr.DataArray(data=(np.sin(lat/10*np.pi)**power)[:, np.newaxis]*np.cos(lon/360*np.pi*wavenum*2), coords=(('lat', lat), ('lon', lon)))
    return signal

def test_assymmetric_eq_waves():
    wav = 5
    signal = make_eq_signal(wav, 1) # assymetric input (sin(lat))
    spec = equatorial_waves(signal)
    # symmetric spectrum should have zeros everywhere
    assert np.allclose(spec, 0)

    spec = equatorial_waves(signal, symmetric=False)
    values = spec.where(np.abs(spec.F_lon) != wav).dropna('F_lon').values
    assert np.allclose(values, 0)
    values = spec.where(np.abs(spec.F_lon) == wav).dropna('F_lon').values
    assert np.all(values > 1)
    return spec

def test_symmetric_eq_waves():
    wav = 5
    signal = make_eq_signal(wav, 2) # symmetric input (sin^2(lat))
    spec = equatorial_waves(signal)
    # symmetric spectrum should have zeros everywhere except the wavenumber
    assert np.allclose(spec.where(np.abs(spec.F_lon) != wav).dropna('F_lon'), 0)
    assert np.all(spec.where(np.abs(spec.F_lon) == wav).dropna('F_lon') > 1)

    # antisymmetric spectrum should be all zeros
    spec = equatorial_waves(signal, symmetric=False)
    assert np.allclose(spec, 0)
