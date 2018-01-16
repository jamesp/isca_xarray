import os
import warnings
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from xarray.plot.utils import _load_default_cmap

from iscaxr.util import nearest_val, absmax

seq_cmap = _load_default_cmap()
div_cmap = plt.cm.RdBu_r


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