from xarray.plot.utils import _load_default_cmap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, hex2color
import matplotlib.pyplot as plt


bwb = LinearSegmentedColormap.from_list('bwb', ((0,0,0), (1,1,1), (0,0,0)))
gwg = LinearSegmentedColormap.from_list('gwg', ((.059, .282, 0), (1,1,1), (.059, .282, 0)))
BrWBr = LinearSegmentedColormap.from_list('BrWBr', ((.333, .118, 0), (1,1,1), (.333, .118, 0)))
RdBkBl = LinearSegmentedColormap.from_list('RdBkBl', ((1, .0862, .329), (0,0,0), (36/255., 123/255., 160/255.)))
RdWtBl = LinearSegmentedColormap.from_list('RdWtBl', ((1, .0862, .329), (1,1,1), (36/255., 123/255., 160/255.)))
sequential = _load_default_cmap()
divergent = plt.cm.RdBu_r

precip = LinearSegmentedColormap('precip',
        {'red':   ((0.0, 0.0, 0.0),
                    (0.1, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.1, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.1, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

        'alpha':  ((0.0, 0.0, 0.0),
                   (0.1, 0.0, 0.0),
                   (1.0, 1.0, 1.0))})

depth = LinearSegmentedColormap.from_list('depth', [(198/255, 172/255, 143/255), (80/255, 117/255, 191/255)])

palatte = ListedColormap(
    [
        hex2color('#ef476f'),
        hex2color('#ffd166'),
        hex2color('#06d6a0'),
        hex2color('#118ab2'),
        hex2color('#073b4c'),
    ]
)

jptemp = LinearSegmentedColormap.from_list('jptemp',
    [(0., hex2color('#FFFFFF')),
     (0.4, hex2color('#118ab2')),
     (0.8, hex2color('#ffd166')),
     (1., hex2color('#ef476f')),

    ])