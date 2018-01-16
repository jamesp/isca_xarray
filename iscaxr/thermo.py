# -*- coding:utf-8 -*-
# Useful thermodynamics

import numpy as np


def sat_press(T, ps0=610.0, Lv=2.5e6, Rv=461.5, T0=273.16):
    """Calculates the saturation pressure based on constant Lv.

    T is temperature in K.

    Default values are for water vapour about a reference
    point T0=273.16.
        T0  = 273.16 K
        ps0 = 610.0 hPa
        Lv  = 2.5 x 10^6 J.kg^-1
        Rv  = 461.5 J.kg^-1.K^-1

    Returns a pressure in hPa.

    ref: Frierson, D. M. W., Held, I. M. & Zurita-Gotor, P.
         A gray-radiation aquaplanet moist GCM. Part I: Static stability and eddy scale.
         J. Atmos. Sci. 63, 2548–2566 (2006).
    """
    return ps0 * np.exp(-(Lv/Rv)*(1/T - 1/T0))

def sat_press_magnus(T):
    """Calculate the saturation vapour pressure of water at a given temperature.

    Uses the Magnus-Tetens approximation for pure water over
    a plane surface of water. Valid in the range -40C to +50C.

    Temperature should be given in K.
    Returns saturation pressure in hPa.

    ref: Alduchov, O.A., and R.E. Eskridge. Improved Magnus` Form Approximation
            of Saturation Vapor Pressure.
         United States: N. p., 1997. Web. doi:10.2172/548871.
    """
    T = T - 273.16  # convert to celsius
    return 6.1094e2 * np.exp(17.625*T / (T + 243.04))


def spec_hum(p, e, epsilon=287.04/461.5, simple=False):
    """Specific humidity of vapour at a given atmospheric pressure.

    Atmospheric Pressure p in hPa, vapour pressure e in hPa.
    Epsilon is the ratio of dry:wet gas constants Rd/Rv.

    `simple = True` assumes p >> (1-epslion)*e.
    Returns specific humidity in kg.kg^-1

    ref: Peixoto, J. P. & Oort, A. H. Physics of Climate. p52
    """
    eps = epsilon  # Rd/Rv
    if simple:
        # assumes (1-epsilon)*e << p
        q = epsilon*e/p
    else:
        q = ((eps * e) / (p - (1-eps) * e))
    return q

# def qs(p, T, es_fn=sat_press_magnus):
#     """Saturation specific humidity."""
#     e = es_fn(T)
#     return spec_hum(p, e)


# def rel_hum(q, p, T):
#     """Calculate relative humidity for specific humidity q (kg.kg^-1)
#     at temperature T (K) and pressure p (hPa)."""
#     _qs = qs(p, T)
#     return q  / _qs
