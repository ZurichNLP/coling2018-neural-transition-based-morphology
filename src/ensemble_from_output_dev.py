#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals
import argparse
import logging
import os
import random
import codecs
from collections import Counter
from itertools import izip
import numpy as np
import re
import sys
import os.path

# Prevent Unicode Encoding errors
sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


parser = argparse.ArgumentParser(
                                 description="""
                                     This takes a.txt files and does shuffling of lines.
                                     """, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--lang",
                    help="Language")
parser.add_argument("--pred_out",
                    help="File to write ensemble predictions")
parser.add_argument("--result_out",
                    help="File to write ensemble accuracy")
#parser.add_argument("--dev",
#                    help="Development file")
parser.add_argument("--input",
                    help="The input files: the first one is the ground truth", nargs='+')
parser.add_argument("--test_only",
                    help="Run predictions for the test set", action='store_true')

parser.add_argument("--max_strategy",
                    help="Take predictions from the best performing system. The default is ensemble strategy.", action='store_true')

parser.add_argument("--nbest",
                    help="Take ensemble of n-best performing systems. Only applicable to ensemble strategy. Default (either omit the option or --nbest=0) is ensemble from all input systems.", type=int)


def best_dev_system(predict_in_fnames, dev, nbest, test_only = False):
    accuracies = []
    for file in predict_in_fnames:
        corr = 0
        total = 0
        if test_only:
            #Identify the file with prediction on the dev set
            if not "test.test" in file:
                print >> sys.stderr, "ERROR: Prediction file on test should be of the format *.test.test.*, check the file {}".format(file)
            file = file.replace("test.test", "dev")
            if not os.path.exists(file):
                print >> sys.stderr, "ERROR: File {} does not exist! Check the input prediction files.".format(file)


        for i,line in enumerate(codecs.open(file, 'r', encoding='utf-8')):
            total += 1
            lemma, word, morph = line.strip().split('\t')
            lemma_dev, word_dev, morph_dev = dev[i].strip().split('\t')
            if word == word_dev:
                corr += 1
        accuracies.append(corr/total)
    print >> sys.stdout, "Accuracies of individual systems: " + str(accuracies)
    print >> sys.stdout, "Best system is {} ({})".format(predict_in_fnames[np.argmax(accuracies)],accuracies[np.argmax(accuracies)])
    #return np.argmax(accuracies)
    accuracies = np.array(accuracies)
    ranks = accuracies.argsort()[::-1][:nbest]
    print >> sys.stdout, "N-best system ranks: {}".format(ranks)
    return ranks #the first index is the system with best dev accuracy

def evaluate(predict_in_fnames,  pred_out, result_out, nbest_index, dev, max_strategy = False, test_only = False):

    # Collect prediction files
    predict_in = []
    #for f in predict_in_fnames:
    for i in nbest_index:
        predict_in.append(codecs.open(predict_in_fnames[i], 'r', encoding='utf8')) #the first index in nbest_index is the system with best dev accuracy

    corr = 0
    total = 0

    replaced = 0

    for line in zip(*predict_in):

        # Collect predictions
        w = []
        for elem in line:
            w.append(elem.strip().split('\t')[1])
        lemma = elem.strip().split('\t')[0]
        morf = elem.strip().split('\t')[2]

        # Predict
        if max_strategy == True:
            # max strategy
            w_pred = w[0] #the first index in nbest_index is the system with best dev accuracy
        else:
            # ensemble strategy
            pred = Counter(w).most_common(1)[0] # (pred,freq)
            #if pred[1] != 1:
            if not all(x==Counter(w).values()[0] for x in Counter(w).values()): #no ties
                w_pred = pred[0]
            else:
                #hack for ties
                w_pred = w[0] #the first index in nbest_index is the system with best dev accuracy

        # hack against repeting character
        if re.search(r"(.)\1{3,}", w_pred):  # at least 3 chars
            if w_pred.startswith(u'jäääär'):
                print >> sys.stdout, "{}: We do not replace funny estonian words...".format(w_pred)
            else:
                #print "Replacing " + w_pred + " --> " + re.sub(r"(.)\1{4,}",lemma, w_pred)
                #w_pred = re.sub(r"(.)\1{4,}",lemma, w_pred)
                print >> sys.stdout,"Replacing " + w_pred + " --> " + lemma
                w_pred = lemma
                replaced += 1

        if not test_only:
            # Check dev set performance
            lemma_dev, word_dev, morph_dev = dev[total].strip().split('\t')
            if w_pred == word_dev:
                corr += 1

        # Write prediction
        pred_out.write(u'\t'.join([lemma,w_pred,morf]) + '\n')

        total += 1

    if not test_only:
        # Write accuracy result
        ens_accuracy = corr/total
        result_out.write("Prediction Accuracy = " + str(ens_accuracy))
        print >> sys.stdout, "Ensemble Accuracy: " + str(ens_accuracy)

    # Write replacement result
    print >> sys.stdout, "Total charachters replaced: " + str(replaced)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('shuffle')
    args = parser.parse_args()

    if args.test_only:
        test_only = True
    else:
        test_only = False

    if args.max_strategy:
        max_strategy = True
    else:
        max_strategy = False

    lang = args.lang

    predict_in_fnames = []
    for file in args.input:
        predict_in_fnames.append(file)
    print >> sys.stdout, 'INPUT FILES:', predict_in_fnames
    if args.nbest:
        if args.nbest==0: #take all input files for ensembles
            nbest = len(predict_in_fnames)
        else:
            nbest = args.nbest
    else:
        nbest = len(predict_in_fnames) #default: take all input files for ensembles


    pred_out_fname = args.pred_out
    pred_out = codecs.open(pred_out_fname, 'w', encoding='utf8')

    result_fname = args.result_out
    result_out = codecs.open(result_fname, 'w', encoding='utf8')

    dev_fname = "../data/all/task1/{}-dev".format(lang)
    #dev_fname = args.dev
    dev = codecs.open(dev_fname, 'r', encoding='utf8').readlines()

    nbest_index = best_dev_system(predict_in_fnames, dev, nbest, test_only) #nbest performing systems
    evaluate(predict_in_fnames, pred_out, result_out, nbest_index, dev, max_strategy, test_only)


#python ensemble_from_output_dev.py --pred_out ../results/DUMB_MIX_ENS/pred.txt --result_out ../results/DUMB_MIX_ENS/res.txt --input ../results/DUMB_MIX_ENS_ens_0/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_1/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_2/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_3/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_4/russian_low.best.dev.predictions --lang russian

#python ensemble_from_output_dev.py --pred_out ../results/DUMB_MIX_ENS/pred.txt --result_out ../results/DUMB_MIX_ENS/res.txt --input ../results/DUMB_MIX_ENS_ens_0/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_1/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_2/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_3/russian_low.best.dev.predictions ../results/DUMB_MIX_ENS_ens_4/russian_low.best.dev.predictions --lang russian --test_only
