import os
import warnings
import subprocess
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

from iscaxr.util import nearest_val, absmax

from iscaxr import cmap





def make_video(filepattern, output, framerate=5):
    command = 'ffmpeg -framerate {framerate:d} -y -i {filepattern!s} -c:v libx264 -r 6 -pix_fmt yuv420p -vf scale=3200:-2 {output!s}'
    command = command.format(filepattern=filepattern, framerate=framerate, output=output)
    subprocess.call([command], shell=True)

def neutral_levels(array, nlevels=13, zero_fac=0.05, max_fac=0.95):
    """Create a set of colour levels with a white region +/-zero_fac*100% either side of zero.
        `zero_fac` sets the extent of the white region either side of zero as a fraction of the array max.
        `max_fac` sets the extent of the levels as a fraction of the array max.
        """
    m = absmax(array.values)
    nlev = (nlevels-1)//2
    levels = np.concatenate([np.linspace(-max_fac, -zero_fac, nlev), np.linspace(zero_fac, max_fac, nlev)])
    return levels*m

def sensible_tick_labels(levels, numticks=9):
    """Create nicely-rounded tick values given a set of colour levels."""
    rounders = [1, 2, 3, 5, 10]
    levels = np.asarray(levels)
    erp = np.ceil(np.log10(absmax(levels))) - 1
    mul = nearest_val(absmax(levels*10**(-erp)), [1,2,3,5,10])
    return np.linspace(-1, 1, numticks)*(10**erp)*mul

def save_figure(fig, filename, overwrite=False):
    """A wrapper around `plt.savefig`.
    - Prevents overwriting existing files unless specified.
    - Uses `bbox_inches='tight'` by default.
    """
    if os.path.isfile(filename) and not overwrite:
        warnings.warn("File %r already exists. Not overwriting" % filename)
    else:
        fig.savefig(filename, bbox_inches='tight')


def plot_lat_press(field, domain, cmap='RdBu_r', ax=None, center0=True):
    """Plot a 2D meshgrid of data defined on pfull-lat levels."""
    if ax is None:
        fig, ax = plt.subplots()
    pp = ax.pcolormesh(domain.latb, domain.phalf, field.transpose('pfull', 'lat'), cmap=cmap)
    if center0:
        absmax = np.max(np.abs(field))
        pp.set_clim(-absmax, absmax)
    cbar = plt.colorbar(pp)
    ax.set_ylim(domain.phalf.max(), domain.phalf.min())
    ax.set_xlim(-90, 90)
    ax.set_xlabel('Latitude ($\\degree$)')
    ax.set_xticks([-90, -45, 0, 45, 90], ['90$\\degree$ S', '45$\\degree$ S', '0$\\degree$', '45$\\degree$ N', '90$\\degree$ N'])
    ax.set_ylabel('Pressure (hPa)')
    return pp, cbar

def plot_lat_lon(field, domain, ax=None, center0=True, overscale=1., **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(ax, GeoAxes):
        transform = ccrs.PlateCarree()
    else:
        transform = ax.transData
    pp = ax.pcolormesh(domain.lonb, domain.latb, field.transpose('lat', 'lon'), transform=transform, **kwargs)
    if center0:
        absmax = np.max(np.abs(field))*overscale
        pp.set_clim(-absmax, absmax)

    return pp

def render_lat_lon_labels(ax, domain):
    ax.set_ylim(domain.latb.max(), domain.latb.min())
    ax.set_ylim(-90, 90)

    ax.set_ylabel('Latitude ($\\degree$)')
    ax.set_yticks([-90, -45, 0, 45, 90], ['90$\\degree$ S', '45$\\degree$ S', '0$\\degree$', '45$\\degree$ N', '90$\\degree$ N'])

    ax.set_xlabel('Longitude ($\\degree$)')
    l_ticks = [0, 45, 90, 135, 180, 225, 270, 315]
    ax.set_xlim(domain.lonb.min(), domain.lonb.max())
    ax.set_xticks(l_ticks, ['%d$\\degree$ E'%l for l in l_ticks])