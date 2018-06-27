# coding: utf-8

from __future__ import division
import editdistance
import os
import csv
import sys

def read_file(fn, only_forms=False):
    with open(fn) as f:
        for row in csv.reader(f, delimiter='\t'):
            if only_forms:
                (form,) = row
            else:
                lemma, form, features = row
            yield form.decode('utf8')


def compute_stats(gold_fn, pred_fn):
    count = 0
    correct = 0
    sum_edit = 0
    for g, p in zip(read_file(gold_fn), read_file(pred_fn)):
        count += 1
        if g == p:
            correct += 1
        else:
            sum_edit += editdistance.eval(g, p)
    accuracy = correct * 100 / count
    average_edit_distance = sum_edit / count
    return accuracy, average_edit_distance

if __name__ == "__main__":

    #GOLD_FN = '../data/all/task1/russian-dev'
    #PRED_FN = GOLD_FN
    _, GOLD_FN, PRED_FN = sys.argv
    acc_avg_edit = compute_stats(GOLD_FN, PRED_FN)
    #lang = os.path.basename(GOLD_FN).rsplit('-', 1)[0]
    print ('Accuracy / Average edit distance: %.2f%% / %.3f' % acc_avg_edit)
