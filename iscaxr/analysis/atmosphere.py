import iscaxr as ixr
from iscaxr.constants import grav, R_dry


def brunt_vaisala(data):
    """Calculate the Brunt-Vaisala (N^2) frequency for an Isca dataset.

    N^2 = -g^2*(p/RT)*(1/theta)*dtheta/dp.

    Parameters
    ----------
        data : xarray.DataSet
        The Isca dataset.  Requires fields 'temp', 'ps', 'pfull' and 'phalf'

    Returns a new xarray.DataArray of N2 values on phalf levels.
    """
    # Brunt Vaisala on phalf as per src/atmos_param/mg_drag/mg_drag.f90
    pfull = (data.pfull/data.phalf.max())*data.ps  # pfull = sigmafull * ps
    p = ixr.domain.pfull_to_phalf(pfull, data)
    theta = ixr.pot_temp(data)
    temp = ixr.domain.pfull_to_phalf(data.temp, data)
    dtheta_dp = ixr.domain.dfdp(theta, data)
    theta_h = ixr.domain. pfull_to_phalf(theta, data)

    N2 = -(grav*grav/R_dry)*p*dtheta_dp / (theta_h*temp)
    return N2
