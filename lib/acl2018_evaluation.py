#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals,print_function, division
from optparse import OptionParser
import os
import sys
import codecs
import re
import numpy as np
import glob
from collections import defaultdict, namedtuple,Counter
from operator import attrgetter
"""
Module for XXX

"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


MODEL_PATTERN = re.compile('x-(?P<model>.*?)-a(?P<align>(crp|cls)).*?-m(?P<mode>(mle|rl\w*|mrt\w*))-x')
Stats = namedtuple('Stats', 'config, con, dec, model, align, mode, mean, std, num_models')
options = None

SUPPORT_STATISTICS = Counter() #   Keep track on the support for a config,con,dec,model,align,mode,foldname tuple (foldname can be __MEAN__)

def grep_aggregate(wildcard):
    """

Accuracy: 0.83
Mean Levenshtein: 0.409
Mean Normalized Levenshtein: 0.047384838201
Mean Reciprocal Rank: 0.83

Aggregate
Accuracy: 0.83
Mean Levenshtein: 0.409
Mean Normalized Levenshtein: 0.047384838201
Mean Reciprocal Rank: 0.83


    """
    print('#AGGREGATING ON',wildcard, file=sys.stderr)
    aggregate_seen = False
    for file in glob.iglob(wildcard):
        #print(file, file=sys.stderr)
        with codecs.open(file, 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if aggregate_seen:
                    if 'L' in options.mode:
                        m = re.search(r'^Mean Normalized Levenshtein:\s*([\d.]+)',line)
                    else:
                        m = re.search(r'^Accuracy:\s*([\d.]+)',line)

                    if m:
                        yield (file,float(m.group(1)))
                        aggregate_seen = False
                        break

                if line.startswith('Aggregate'):
                    aggregate_seen = True
                    #print('Aggregate found',file=sys.stderr)


def aggregate_results(results):
    """
    Add special keys for folds and seeds
    __MEAN__
    __SD__
    {(u'x-haem-acls-pcelex-n200_1-w100_20_100_T-e30_10-oADADELTA_0-mmle-x', u'dev', u'greedy'): {u'finnish-task1': {u'1': 0.978723404255}, u'navajo-task1': {u'1': 0.986914600551}}, (u'x-haem-acls-pcelex-n200_1-w100_20_100_T-e30_10-oADADELTA_0-mmle-x', u'test', u'beam4'): {u'finnish-task1': {u'1': 0.966699107181},
    """

    for (config,con,dec),folds in results.iteritems():
        m = MODEL_PATTERN.match(config)
        if m:
            mode = m.groupdict()['mode']  # mle, rl, mrt, ...
            model = m.groupdict()['model']  # haem, hacm, hard, ...
            align = m.groupdict()['align']  # crp, cls ...
        else:
            mode, model, align = '', '', ''
        # mean accuracies across seeds for each fold
        foldaccuracies = []
        # we count number of models over folds and seeds
        num_individual_models = 0

        for foldname,fold in folds.items():
            if 'Q' in options.mode:
                seedaccurracies = fold.values()[:1] if fold.values() else []  # pick one
#                SUPPORT_STATISTICS[(config,con,dec,model,align,mode,foldname)] += 1
            else:
                seedaccurracies = []
                for seed_acc in fold.values():
                    seedaccurracies.append(seed_acc)
                    SUPPORT_STATISTICS[(config,con,dec,model,align,mode,foldname)] += 1
            # aggregate on fold level
            fold['__MEAN__'] = float(np.mean(seedaccurracies))
            fold['__SD__'] = float(np.std(seedaccurracies))
            l = len(seedaccurracies)
            num_individual_models += l
            SUPPORT_STATISTICS[(config,con,dec,model,align,mode,'__MEAN__')] += l
            SUPPORT_STATISTICS[(config,con,dec,model,align,mode,'__SD__')] += l

            # statistics over seeds for this fold
            fold['__STATS__'] = fold['__MEAN__'], fold['__SD__'], l
            foldaccuracies.append(fold['__MEAN__'])
        # aggregate on (config, condition, decoding) level
        folds['__MEAN__'] = float(np.mean(foldaccuracies))
        folds['__SD__'] = float(np.std(foldaccuracies))
        # statistics over folds for this (config, condition, decoding)
        folds['__STATS__'] = folds['__MEAN__'], folds['__SD__'], num_individual_models

def output_aggregated_results(results, options):
    output = []
    for (config,con,dec),folds in results.iteritems():
        m = MODEL_PATTERN.match(config)
        if m:
            mode = m.groupdict()['mode']  # mle, rl, mrt, ...
            model = m.groupdict()['model']  # haem, hacm, hard, ...
            align = m.groupdict()['align']  # crp, cls ...
        else:
            mode, model, align = '', '', ''

        for foldname, fold in folds.iteritems():
            # print('FOLD',fold,file=sys.stderr)


            if foldname == '__STATS__':
                if 'q' in options.mode:
                    mean, std, num_models = fold
                    output.append(Stats(config, con, dec, model, align, mode, mean, std, num_models))

            elif type(fold) == float: # aggregated fold result __MEAN__ or __SD__

                if not 'S' in options.mode:
                    if 'm' in options.mode or 'M' in options.mode:
                        print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%s" %(config,con,dec,model,align,mode,foldname,foldname,fold,SUPPORT_STATISTICS[(config,con,dec,model,align,mode,foldname)]))

            else:
                if 'M' in options.mode: continue
                for seedname,seed in fold.iteritems():
                    if seedname == '__STATS__': continue
                    if seedname.isdigit():  # regular seed, not__MEAN__ or __SD__
                        if 's' in options.mode:
                            print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%s" %(config,con,dec,model,align,mode,foldname,seedname,seed,"1"))

                    else:
                        if 'm' in options.mode:
                            print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" %(config,con,dec,model,align,mode,foldname,seedname,seed,SUPPORT_STATISTICS[(config,con,dec,model,align,mode,foldname)]))


    if output:
        # q mode, sort the results
        output.sort(key=attrgetter('con', 'model', 'align', 'mode', 'dec', 'mean', 'std'), reverse=True)
        for n in output:
            print("%s\t%s\t%s\t%s\t%s\t%s\t%.6f\t%.6f\t%d" %
                  (n.con, n.config, n.model, n.align, n.mode, n.dec, n.mean, n.std, n.num_models))

def process(options=None,args=None):
    """
    Do the processing
    """
    resultpath = args[0]
    if resultpath.endswith('/'):
        resultpath = resultpath[0:-1]
    if not os.path.exists(resultpath):
        print('#ERROR: RESULTPATH NOT EXISTING',file=sys.stderr)
        exit(1)
    m = re.search(r'^(.+?)/?',resultpath)
    if m:
        dataset = m.group(1)
    else:
        print('#ERROR: MALFOROMED RESULTPATH',file=sys.stderr)
        exit(1)

#    ../results/sigmorphon2016/x-haem-acrp-pcelex-n200_1-w100_20_100_T-e30_10-oADADELTA_0-mmle-x/georgian-task1/s_1/f.greedy.test.eval
    pattern = re.compile(r'''(?x)
   ^{RESULTPATH}/              # Result starts from current directory
   (?P<config>.*?)/
   (?P<fold>.*?)/
   s(eed)?_(?P<seed>.*?)/
   (f\.|.*?txt\_)(?P<dec>.*?)\.(?P<con>test|dev)\.eval$
    '''.format(RESULTPATH=resultpath))

    # (config,con,dec) -> fold -> seed -> acc
    # f
    results = {}
    for file, acc in grep_aggregate(options.glob.format(RESULTPATH=resultpath)):

        d = pattern.match(file)
        if d:

            config = d.group('config')
            con = d.group('con')  # test or dev
            fold = d.group('fold')
            seed = d.group('seed')
            dec = d.group('dec')

            ######## filter down to the results we are actually interested in
            if options.fold_filter:
                if not re.search(options.fold_filter, fold):
                    continue
            # Deal with test/eval
            if 'D' in options.mode and con != 'dev':
                continue
            if 'T' in options.mode and con != 'test':
                continue
            # Only ensembles
            if  'E' in options.mode and not seed.startswith('0'):
                #print('IGNORED',seed, file=sys.stderr)
                continue
            # Deal with ensembles
            if seed.startswith('0') and not ('e' in options.mode or 'E' in options.mode) :
                #print('IGNORED',seed, file=sys.stderr)
                continue


            # ignore all decoder settings than the ones in the filter
            if not re.search(options.decoder_filter, dec):
                continue

            if options.debug:
                print('#file',file,file=sys.stderr)


            if (config,con,dec) not in results:
                results[(config,con,dec)] = {}
            if fold not in results[(config,con,dec)]:
                results[(config,con,dec)][fold] = {}
            if seed not in results[(config,con,dec)][fold]:
                results[(config,con,dec)][fold][seed] = acc
        else:
            print('Failed to parse: ', resultpath, file, file=sys.stderr)
    aggregate_results(results)
    output_aggregated_results(results, options)

def main():
    """
    Invoke this module as a script
    """
    global options
    parser = OptionParser(
        usage = '%prog [OPTIONS] RESULTPATH',
        version='%prog 0.99', #
        description='Calculate results on acl2018 datasets',
        epilog='Contact simon.clematide@uzh.ch'
        )
    parser.add_option('-l', '--logfile', dest='logfilename',
                      help='write log to FILE', metavar='FILE')
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_option('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')
    parser.add_option('-g', '--glob',
                      action='store', dest='glob', default='{RESULTPATH}/x*x/*/s*/*eval',
                      help='change file globbing for accessing evaluation results (%default)')
    parser.add_option('-f', '--fold_filter',
                      action='store', dest='fold_filter', default=None,
                      help='only use folds matching (re.search) the specified regular expression on the fold name (e.g. "^english" for all folds starting with the string english)  (Default "%default")')
    parser.add_option('-D', '--decoder_filter',
                      action='store', dest='decoder_filter', default="greedy|beam4",
                      help='''used on decoding mode label; matches (re.search) with the specified regular expression (Default "%default")''')
    parser.add_option('-m', '--mode',
                      action='store', dest='mode', default='ms',
                      help='''compatibel characters can be combined
                           s: individual seed results;
                           S: only individual seed results;
                           m: mean/sd values (on seeds and folds);
                           M: mean/sd (on folds only);
                           e: include ensembles;
                                  E: only ensembles;
                                  T: only test results;
                                  D: only dev results
                           q: sort the results by accuracy
                           L: evaluate on edit distance, not on Accuracy
                                  ''')

    (options, args) = parser.parse_args()
    if options.debug:
        print("options=",options,file=sys.stderr)

    if len(args) < 1:
        print('# RESULTPATH needed')
        parser.print_help()
        exit(1)
    options.mode = set(options.mode)
    process(options=options,args=args)


if __name__ == '__main__':
    main()
