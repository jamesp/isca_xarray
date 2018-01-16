# -*- coding:utf-8 -*-
import numpy as np
import xarray.ufuncs as xruf

import iscaxr as ixr
from iscaxr.constants import grav, R_dry, omega


def brunt_vaisala(data):
    """Calculate the Brunt-Vaisala (N^2) frequency for an Isca dataset.
    Following Vallis (2017) p.97

    N^2 = g/θ*dθ/dz

    Parameters
    ----------
        data : xarray.DataSet
        The Isca dataset.  Requires fields 'temp', 'ps', 'pfull' and 'phalf'

    Returns a new xarray.DataArray of N^2 values on phalf levels, in s^-2.
    """
    # Brunt Vaisala on phalf as per src/atmos_param/mg_drag/mg_drag.f90
    theta = ixr.pot_temp(data)
    theta_h = ixr.domain.pfull_to_phalf(theta, data)
    dz = ixr.domain.calculate_dz(data)
    dtheta = ixr.domain.diff_pfull(theta, data)
    return grav/theta_h*dtheta/dz

def eady_growth_rate(data):
    """Calculate the local Eady Growth rate.
    Following Vallis (2017) p.354.

        EGR = 0.31*du/dz*f/N

    Parameters
    ----------
        data : xarray.DataSet
        The Isca dataset.  Requires fields 'temp', 'ps', 'pfull' and 'phalf'

    Returns a new xarray.DataArray of growth rate values on phalf levels,
    in s^-1.
    """
    N2 = ixr.brunt_vaisala(data)
    f = 2.0*omega*xruf.sin(xruf.deg2rad(data.lat))

    dz = ixr.domain.calculate_dz(data)
    du = ixr.domain.diff_pfull(data.ucomp, data)

    N = xruf.sqrt(N2.where(N2 > 0))

    egr = 0.31*du/dz*f/N
    return np.abs(egr)