from __future__ import division
import glob
import re
import os
import sys
from collections import defaultdict
import compute_results

# expect to be written to file?
TOFILE = sys.argv[1]=='tofile' if len(sys.argv) > 1 else None
SYSTEMS = sys.argv[2].split(',') if len(sys.argv) > 2 else None
SETTINGS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['low', 'medium', 'high']

SKIP_HIGH = True

lang_dict = defaultdict(lambda: defaultdict(dict))
seen_systems = set()
results_dict = defaultdict(lambda: defaultdict(list))

if SYSTEMS:
    system_paths = []
    for system in SYSTEMS:
        if system != 'SOFT':
            for setting in SETTINGS:
                system_paths += glob.glob('../results/'+system+'/*'+setting+'.best.dev')
        else:
            for setting in SETTINGS:
                system_paths += glob.glob('../results/'+system+'/*'+setting+'.external_eval.txt.dev')
else:
    for setting in SETTINGS:
        system_paths = (glob.glob('../results/*/*'+setting+'.best.dev') +
                        glob.glob('../results/SOFT/*'+setting+'.external_eval.txt.dev'))

for fn in system_paths:
#for fn in (glob.glob('../results/*/*best.dev') +
#    glob.glob('../results/SOFT/*external_eval.txt.dev')):

    m = re.match('.*/results/(.+?)/(.+?)_(.+?)\.(?:best|external_eval\.txt)\.dev', fn)
    system, lang, setting = m.groups()
    with open(fn) as f:
        for line in f:
            m = re.match('Prediction\sAccuracy\s=\s([\d\.]+)$', line)
            if m:
                result = float(m.group(1))
                lang_dict[lang][setting][system] = result
                results_dict[setting][system].append(result)
                seen_systems.add(system)
                break

# official baseline
if not SYSTEMS or 'BASELINE' in SYSTEMS:
    for setting in SETTINGS:
        for fn in glob.glob('../results/BASELINE/*'+setting+'-out'):
            basename = os.path.basename(fn)
            system = 'BASELINE'
            lang, _ , _ = basename.rsplit('-', 2)
#            if SKIP_HIGH and setting == 'high':
#                continue
            gold_fn = os.path.join('../data/all/task1/', lang + '-dev')
            acc, avg_edit = compute_results.compute_stats(gold_fn=gold_fn, pred_fn=fn)
            result = acc / 100
            lang_dict[lang][setting][system] = result
            results_dict[setting][system].append(result)
            seen_systems.add(system)

#settings = ['low', 'medium', 'high']
systems = sorted(seen_systems)
num_systems = len(systems)
lang_dict = {k : dict(v) for k, v in lang_dict.iteritems()}
results_dict = {setting :
                {system : sum(results) / len(results)
                 for system, results in d.iteritems()}
                for setting, d in results_dict.iteritems()}

# settings are rows, systems are cols

def settings2system_dict_print(d):
    num_cols = num_systems + 1
    padding_size = (130 - num_cols) / num_cols
    formatting = '|'.join(['{:^%d}' % padding_size]*num_cols)
    # somehow, I can't get highlighting to work with `formatting`
    max_value_formatting = (('\033[92m{:^%d.3f}\033[0m' % padding_size))
#                        if not TOFILE else '*{:.3f}*')
    header = formatting.format('SETTING', *systems)

    #if header:
    print header
    for setting, systems_dict in d.items():
        max_value = max(systems_dict.values())
        results = []
        for system in systems:
            value = systems_dict.get(system)
            if value is not None:
                result = (max_value_formatting.format(value)
                          if value == max_value else ('%.3f' % value))
            else:
                result = 'n/a'
            results.append(result)
        print formatting.format(setting.upper(), *results)


def settings2system_dict_print_to_file(d):
    num_cols = num_systems + 2
    padding_size = (130 - num_cols) / num_cols
    formatting = '|'.join(['{:^%d}' % padding_size]*num_cols)
    max_value_formatting = (('\033[92m{:^%d.3f}\033[0m' % padding_size))
#                            if not TOFILE else '*{:.3f}*')
    header = formatting.format('LANGUAGE', 'SETTING', *systems)
    #if header:
    print header + '\n'
    for lang in sorted(d.keys()):
        for setting,systems_dict in d[lang].items():
            max_value = max(systems_dict.values())
            results = []
            for system in systems:
                value = systems_dict.get(system)
                if value is not None:
                    result = (max_value_formatting.format(value)
                              if value == max_value else ('%.3f' % value))
                else:
                    result = 'n/a'
                results.append(result)
            print formatting.format(lang, setting.upper(), *results)

if TOFILE:
    settings2system_dict_print_to_file(lang_dict)

else:
    for lang in sorted(lang_dict):
        print '{:_^130}'.format(' ' + lang + ' ')
        settings2system_dict_print(lang_dict[lang])
        print '\n'

    print '{:=^130}'.format(' averaged over languages ')
    settings2system_dict_print(results_dict)
