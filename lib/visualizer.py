import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



model_map = {'hard':('royalblue', 0), 'haem': ('green', 1), 'haem_sub': ('seagreen', 2),
             'hacm':('red', 3), 'hacm_sub':('firebrick', 4)}
shape_map = {'crp': 'o', 'cls': 'v'}
decod_map = {'greedy': 'lightgray', 'beam4' : 'black'}
mode_map  = {'mle': -1, 'mrt': 0, 'rl': 1}

FULLMODEL = 25
BASELINE = 'hard'
FILTER_OUTLIERS = False
fn = sys.argv[1]
short_fn = os.path.basename(fn).replace('.tsv', '')

inp = []
with open(fn) as f: 
    for _, _, ml, al, md, dc, ac, sd, sp in csv.reader(f, delimiter='\t'):
        sp = int(sp)
        if sp == FULLMODEL:
            inp.append((ml, al, md, dc, float(ac), float(sd), sp))
        else:
            print 'Discarding for lack of support: ', ml, al, md, dc, float(ac), float(sd), sp
if FILTER_OUTLIERS:
    accs = np.array([i[4] for i in inp])
    mean_accs = np.mean(accs)
    baseline = np.array([i[0] != BASELINE for i in inp], dtype=np.bool_)
    # only drop outliers from below and never drop baseline
    outlier_filter = is_outlier(accs, thresh=4).astype(bool) * (accs < mean_accs) * baseline
else:
    outlier_filter = [False]*len(inp)

# color = model, shape = align, edgecolor = dec, x = mode
fig, ax = plt.subplots()
tick_labels = []
tick_positions = []
#ax.grid(ls='dashed', c='lightgray')
for (model, align, mode, dec, acc, std, sup), outlier in zip(inp, outlier_filter):
    c, x = model_map[model]
    x = 2*x + mode_map[mode]/2.
    if x not in tick_positions:
        tick_positions.append(x)
        tick_labels.append(mode[:2])
    if outlier:
        print 'Discarded as outlier: ', model, align, mode, dec, acc, std, sp
        continue
    ax.scatter(x, float(acc), s=70,
               color=c, marker=shape_map[align], edgecolor=decod_map[dec])
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0)
ax.set_title(short_fn)

fig.savefig(short_fn + '.png', dpi=400)