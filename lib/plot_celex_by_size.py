import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

#2PKE curve:
results = '../results/analysis/'
plotdata = [('celex2PKE',
             dict(hard=[], haem=[], hacm=[],
                  drey=[(0.7751, None), (0.8070, None), (0.8546, None), (0.8746, None)],
                  nwfst=[(0.6172, None), (0.7538, None), (0.8462, None), (0.8550, None)])),
            ('celex13SIA',
             dict(hard=[], haem=[], hacm=[],
                  drey=[(0.7773, None), (0.8073, None), (0.8450, None), (0.8750, None)],
                  nwfst=[(0.7502, None), (0.7709, None), (0.8358, None), (0.8496, None)]))]
samples = [('0050', 50), ('0100', 100), ('0300', 300), ('', 500)]
for plotname, data in plotdata:
    for sample, s in samples:
        fn = results + plotname + sample + '.results.tsv'
        with open(fn) as f:
            for _, _, ml, al, md, dc, acc, std, spt in csv.reader(f, delimiter='\t'):
                if al == 'crp' and md == 'mle' and dc == 'beam4' and ml in ['hard', 'hacm', 'haem']:
                    assert int(spt) == 15 or int(spt) == 25
                    data[ml].append((float(acc), float(std)))


fig, axes = plt.subplots(nrows=1, ncols=2, dpi=400)
x = [50, 100, 300, 500]
for i, ((plname, pldata), ax) in enumerate(zip(plotdata, axes.reshape(-1))):
    ax.grid(alpha=0.2)
    ax.plot(x, [m*100 for m, _ in pldata['hard']], marker='v', alpha=0.8, label=r"HA$^*$",  c='green', linestyle='-.')
    ax.plot(x, [m*100 for m, _ in pldata['nwfst']], marker='v', alpha=0.8, label=r"NWFST",  c='blue', linestyle='-.')
    ax.plot(x, [m*100 for m, _ in pldata['drey']], marker='v', alpha=0.8, label=r"LAT", c='black', linestyle='-.')
    
    #ax.plot(x, [m*100 for m, _ in pldata['hacm']], marker='o', alpha=0.8, label=r"CL", c='red', linestyle='dashed')
    ax.plot(x, [m*100 for m, _ in pldata['haem']], marker='o', alpha=0.75, label=r"CA", c='red',
            linestyle='dashed', linewidth=3)
    ax.set_xticks(x)
    if i == 1:
        ax.legend(loc=4, fontsize=13)
    ax.set_title(plname, fontsize=18)
fig.savefig('../results/analysis/celex_by_size.png')
