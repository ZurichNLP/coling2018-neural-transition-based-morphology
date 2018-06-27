#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
import os
import sys
import codecs
import collections
from sklearn.metrics import cohen_kappa_score
"""
Ensemble system for result files

"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

# dict maps row number of entry to counter with tuple (LEMMA, WF, MORPH)
SYSTEMS = collections.defaultdict(collections.Counter)
ROWS = collections.Counter()
VOTES = collections.Counter()
TIE_STATS = collections.Counter()
GOLDEN = None

def read_golden(filepath):
    """
    Does not work right now!
    """
    global GOLDEN
    GOLDEN = []
    with codecs.open(filepath,'r',encoding="utf-8") as f:
        for l in f:
            l = l.rstrip()
            GOLDEN.append(l)

def read_files(args):
    global SYSTEMS, ROWS
    for arg in args:
        with codecs.open(arg,'r',encoding="utf-8") as f:
            for i,l in enumerate(f):
                ROWS[arg] += 1
                SYSTEMS[i][l.strip()] += 1
        print >> sys.stderr, '#INFO, file %s contains %d items'%(arg,ROWS[arg])
    if len(set(ROWS.values())) > 1:
        print >> sys.stderr, '#WARNING, number of predictions not equal', ROWS.values()

def pprint_ties(distribution, i, options):
    if options.debug:
        print >> sys.stderr, "\t".join('#VOTES:'+str(c) + k for (k,c) in distribution.most_common())
    else:
        distr = distribution.most_common()
        #print >> sys.stderr, '# GOLDEN', GOLDEN[i], distr[0]
        if GOLDEN[i] == distr[0][0].split('\t')[2]:
            label = 'CORRECT'
        else:
            label = 'WRONG'
        if (len(distr) > 1 and distr[0][1] == distr[1][1]) or label == 'WRONG':
            print >> sys.stderr, label,"[",GOLDEN[i],"]", "\t".join('#VOTES:'+str(c)+" " + k for (k,c) in distribution.most_common())



def process(options=None,args=None):
    """
    Do the processing
    """
    if options.debug:
        print >>sys.stderr, options
    if options.golden_file:

        read_golden(options.golden_file)

    read_files(args)
    for i in sorted(SYSTEMS):
        nbest = SYSTEMS[i].most_common(3)  # returns list of item * count pairs [('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631), ('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)]
        if nbest[0][1] == 1:
            TIE_STATS['tie1'] += 1
           # pprint_ties(SYSTEMS[i])
        elif len(nbest) > 1:
            if nbest[0][1] == nbest[1][1]:
                TIE_STATS['tie'+str(nbest[0][1])] += 1
               # pprint_ties(SYSTEMS[i])


            else:
                TIE_STATS['notie'] += 1
        else:
            TIE_STATS['notie'] += 1
        pprint_ties(SYSTEMS[i],i, options)
        print >> sys.stdout, nbest[0][0]


    print >> sys.stderr, TIE_STATS


def main():
    """
    Invoke this module as a script
    """
# global options
    parser = OptionParser(
        usage = '%prog [OPTIONS] SYS1 SYS2 ... SYSN > ENSEMBLE',
        version='%prog 0.99', #
        description='Read files from command line and produce ensemble on stdout',
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
    parser.add_option('-e', '--ensemble',
                      action='store', dest='ensemble', default=3,
                      help='build ensemble system from %default outputs by majority vote')
    parser.add_option('-g', '--golden_file',
                      action='store', dest='golden_file', default=None,
                      help='read and process the golden solutions (1 column only)')

    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options


    process(options=options,args=args)


if __name__ == '__main__':
    main()
