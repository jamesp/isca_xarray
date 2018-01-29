from xarray.plot.utils import _load_default_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


bwb = LinearSegmentedColormap.from_list('bwb', ((0,0,0), (1,1,1), (0,0,0)))
gwg = LinearSegmentedColormap.from_list('gwg', ((.059, .282, 0), (1,1,1), (.059, .282, 0)))
BrWBr = LinearSegmentedColormap.from_list('BrWBr', ((.333, .118, 0), (1,1,1), (.333, .118, 0)))

sequential = _load_default_cmap()
divergent = plt.cm.RdBu_r
