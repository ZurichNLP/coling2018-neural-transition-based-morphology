# coding: utf-8

from __future__ import division
import os
import sys
import codecs

def dev_copy_file(f_in,f_out):
    dev_out=codecs.open(f_out,'w','utf8')
    with codecs.open(f_in,'r','utf8') as dev:
        for row in dev:
            #lemma, form, features = row.strip().split('/t')
            items = row.strip().split('\t')
            try:
                dev_out.write("\t".join([items[0],items[0],items[2]])+"\n")
                #dev_out.write("\t".join([lemma,lemma,features]))
            except:
                print row

if __name__ == "__main__":

    #GOLD_FN = '../data/all/task1/russian-dev'
    #PRED_FN = '../data/all/copy_data/russian-dev'
    _, DEV_FN, DEV_COPY_FN = sys.argv
    dev_copy_file(DEV_FN, DEV_COPY_FN)
