#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
import os
import sys
import codecs
import multiprocessing
import re
import subprocess
import time

"""
Module for launching experiments on datasets

MUST BE CALLED DIRECTLY FROM lib subdirectory. All "absolute" paths have to be relative to current directory in lib.

Assumptions:
 - there are different datasets {D} represented as directories
 - each dataset consists of at least one fold {F_N} represented as a directory
 - every fold consists of a file test.txt, train.txt, dev.txt
 - every fold has results for several seeds {S_N}, N=0 is reserved for ensembling; all represented as directories
 - every experimental configuration {X} represented as a directory
 - the computed results {R} consist of several files with consistent filenames over all datasets: f.model f.train.txt_beam6.test.predictions etc.

The results are stored according to the following schema:
 {D}/{X}/{F_N}/{S_N}/{R}


A configuration file specifies all necessary parameters to lauch a set of experiments
 -

"""

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

# Dummy Values
options = None
args = None
reloadpath = "DUMMY"


hidden_dim = 200
layers     = 1


CMDPREFIX = ''

SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '../results/celex'


# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '/mnt/storage/hex/users/makarov/morphological-reinflection/data/celex'

DATASET_FOLDS = [
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_0.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_3.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_4.']




DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'test.txt'
DATASET_TESTFILE_NAME = 'test.txt'


MODEL_CONFIGS = [('haem', '', 'haem'),
                 #('haem', '--substitution', 'haem_sub'),
                 #('hacm', '', 'hacm'),
                 #('hacm', '--copy-as-substitution', 'hacm_sub'),
                 #('hard', '', 'hard')
                 ]

# ALIGN CALL, name:
ALIGN = ('--align-smart', 'crp'), #('--align-cls', 'cls')

# DEFAULT PARAMS
# General parameters not depending an a specific dataset
PARAMS = dict(
    # DYNET PARAMS
    MEM          = 1000,
    AUTOBATCH    = 0,
    # MODEL
    INPUT        = 100,
    FEAT         = 20,
    ACTION       = 100,
    PARAMTYING   = '--param-tying',  # same params for actions & characters
    EHIDDEN      = hidden_dim,
    DHIDDEN      = hidden_dim,
    HIDDEN       = hidden_dim,
    ELAYERS      = layers,
    DLAYERS      = layers,
    LAYERS       = layers,
    VANILLALSTM  = '',  # use CoupledLSTM
    MLP          = 0,   # no hidden-layer in classifier
    NONLIN       = 'ReLU',
    # OPTIMIZATION
    OPTIMIZATION = 'ADADELTA',
    BATCHSIZE    = 1,
    DECBATCHSIZE = 25,  # batchsize used in decoding
    PICKLOSS     = '',  # use dev acc for early stopping
    DROPOUT      = 0,  # no dropout
    PRETRAIN     = '--pretrain-epochs=0',  # skip pretraining
    # DATA
    WRAPS        = 'both',  # wrap lemma, word with opening & closing boundary tags
    ITERATIONS   = 150,  # for crp aligner
    VERBOSE      = '',
    # NOT USED
    BEAMWIDTH    = 0,  # beam search not used during training
    L2           = 0,   # no l2 regularization
    # RL / MRT
    SAMPLESIZE   = 20,
    SNEG         = 1,
    PATIENCE     = 10,
    EPOCHS       = 50,
    MLE_MODE     = 'mle'

)

RESULTS_DIR = ('x-{MODELNAME}-a{ALIGNNAME}-p{DATASET}-'
               'n{HIDDEN}_{LAYERS}{RNNEXTRA}-w{INPUT}_{FEAT}_{ACTION}{INPEXTRA}-'
               'e{EPOCHS}_{PATIENCE}-o{OPTIMIZATION}_{DROPOUT}{OPTEXTRA}-'
               'm{MODE}{MODEEXTRA}-x')
MLE_RESULTS_DIR = ('x-{MODELNAME}-a{ALIGNNAME}-p{DATASET}-'
               'n{HIDDEN}_{LAYERS}{RNNEXTRA}-w{INPUT}_{FEAT}_{ACTION}{INPEXTRA}-'
               'e{MLE_EPOCHS}_{MLE_PATIENCE}-o{OPTIMIZATION}_{DROPOUT}{OPTEXTRA}-'
               'm{MLE_MODE}{MODEEXTRA}-x')


CALL = """{CMDPREFIX} python run_transducer.py --dynet-seed {SEED} --dynet-mem {MEM} --dynet-autobatch {AUTOBATCH} \
 --transducer={TRANSDUCER} {DATAFORMAT} \
 --input={INPUT} --feat-input={FEAT} --action-input={ACTION} {POSEMB} \
 --enc-hidden={EHIDDEN} --dec-hidden={DHIDDEN} --enc-layers={ELAYERS} --dec-layers={DLAYERS} \
 {VANILLALSTM} --mlp={MLP} --nonlin={NONLIN} {MODE_OPTIONS}  \
 --dropout={DROPOUT} --optimization={OPTIMIZATION} --l2={L2} \
 --batch-size={BATCHSIZE} --decbatch-size={DECBATCHSIZE} \
 --patience={PATIENCE} --epochs={EPOCHS} {PICKLOSS} \
 {ALIGN} --tag-wraps={WRAPS} --iterations={ITERATIONS} {PARAMTYING} \
 {SUBSTITUTION} --mode={MODE}  {VERBOSE} --beam-width={BEAMWIDTH} --beam-widths={BEAMWIDTHS} \
 {PRETRAIN} --sample-size={SAMPLESIZE} --scale-negative={SNEG} \
 {TRAINPATH} {DEVPATH} {RESULTSPATH} --test-path={TESTPATH} --reload-path={RELOADPATH} 2>&1 > {OUTPATH}  && touch {DONEPATH}
"""




def linecount(filename):
    return sum(1 for _ in open(filename, 'rbU'))

def get_train_dev_test_foldpath():
    return [(f,[f + d for d in [DATASET_TRAINFILE_NAME, DATASET_DEVFILE_NAME, DATASET_TESTFILE_NAME ]]) for f in DATASET_FOLDS]


def launch_make(calls, options):
    """
    create Makefile from calls
    """
    if not 'r' in options.mode:
        mle_calls = [call for call in calls if call[2] == ""]
    else:
        mle_calls = []
    target_defs = []
    target_rules = []
    for (call, donepath,reloadpath) in mle_calls:
        target_defs.append("target-files += %s" % donepath)
        target_rules.append("%s : %s\n\tmkdir -p $(@D) && %s" % (donepath,"",call))

    if not 'e' in options.mode:
        non_mle_calls = [call for call in calls if call[2] != ""]
    else:
        non_mle_calls = []

    for (call, donepath,reloadpath) in non_mle_calls:
        target_defs.append("target-files += %s" % donepath)
        target_rules.append("%s : %s\n\tmkdir -p $(@D) && %s" % (donepath,reloadpath+"/f.model.done",call))

    print '# NUMBER OF MLE CALLS',len(mle_calls)
    print '# NUMBER OF NON-MLE CALLS',len(non_mle_calls)
    print "\n".join(target_defs)
    print "\ntarget:$(target-files)\n"
    print "\n\n".join(target_rules)
    print "SHELL:=/bin/bash"

def launch(call):
    sys.stderr.flush()
    print >> sys.stderr, '# Launching ...' +re.sub(r'''.*(\s\S+f.model).*''',r'\1',call.strip())
    sys.stderr.flush()
    if 'x' in options.mode:
        print >> sys.stderr, '# CALL', call
        subprocess.call(call, shell=True)
    else:
        print  call

def launch_parallel(calls):
    calls_len = len(calls)
    bunches_of_calls = [calls[i:i+options.parallel]
                        for i in range(0, calls_len, options.parallel)]
    for i, bunch_of_calls in enumerate(bunches_of_calls):
        then = time.time()
        assert len(bunch_of_calls) <= options.parallel
        pool = multiprocessing.Pool()
        pool.map(launch, bunch_of_calls)
        pool.close()
        pool.join()
        now = time.time() - then
        sys.stderr.flush()
        print >> sys.stderr, '# Finished with {} calls out of {}; last call took {:.1f} min'.format(min((i+1)*options.parallel, calls_len), calls_len,now / 60.)
        sys.stderr.flush()
#        print >> sys.stderr, '#Finished in {:.1f} min'.format(now / 60.)
#        print >> sys.stderr


def process(options,args):
    """
    Do the processing
    """

    PARAMS.update(MORE_PARAMS)
    inputextra = ''.join(['T' if PARAMS['PARAMTYING'] else '',
                          'A' if PARAMS['POSEMB'] == '--avm-feat-format' else ''])
    inputextra = '_' + inputextra if inputextra else '' 
    optextra = '_l2' if PARAMS['L2'] else ''
    if options.epochs > 0:
        PARAMS['EPOCHS'] = options.epochs
    calls = []
    for mode in MODE_CONFIGS:
        m_config_len = len(mode)
        if m_config_len == 3:
            MODE, MODE_OPTIONS, MODEEXTRA = mode
        elif  m_config_len == 5:
            MODE, MODE_OPTIONS, MODEEXTRA, MLE_PATIENCE, MR_PATIENCE = mode

        PARAMS['MODE'] = MODE
        PARAMS['MODE_OPTIONS'] = MODE_OPTIONS
        #PARAMS['MODEEXTRA'] = MODEEXTRA

        for FOLD_ID, (train, dev, test) in get_train_dev_test_foldpath():

            # PATHS
            TRAINPATH = DATASET_PATH + train
            DEVPATH   = DATASET_PATH + dev
            TESTPATH  = DATASET_PATH + test


            # compute trainfile size in lines
            if options.debug and not os.path.exists(TRAINPATH):
                trainfilesize = 42
                print >> sys.stderr, '#INFO: TRAINPATH DOES NOT EXIST; ASSUMING 42 TRAINING ITEMS in DEBUG MODE'
            else:
                trainfilesize = linecount(TRAINPATH)

            # Finetuning of the parameters according to mode, training set size etc.
            # see https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/transducer_settings.md
            if  DATASET_SHORTNAME == 'nck15' or DATASET_SHORTNAME == 'ddn13':
                if trainfilesize > 50000:
                    PARAMS['EPOCHS'] = 20
                if trainfilesize > 200000:
                    PARAMS['EPOCHS'] = 5



            print >> sys.stderr, '#MODEL=',PARAMS['MODE']
            # sometimes we have different epochs and patiences depending on the mode
            # the result path of the reloaded models still needs to know the corresponding mle values
            PARAMS['MLE_EPOCHS'] = PARAMS['EPOCHS']
            PARAMS['MLE_PATIENCE'] = PARAMS['PATIENCE']


            if  DATASET_SHORTNAME == 'celex':
                PARAMS['MLE_PATIENCE'] = 10
                PARAMS['PATIENCE'] = 15

            elif DATASET_SHORTNAME.startswith('celex'):
                # celex by task
                PARAMS['MLE_PATIENCE'] = 10
                PARAMS['PATIENCE'] = 15

            elif DATASET_SHORTNAME == 'sgm2017low':
                PARAMS['MLE_PATIENCE'] = 15
                PARAMS['PATIENCE'] = 20

            elif DATASET_SHORTNAME == 'sgm2017medium':
                PARAMS['MLE_PATIENCE'] = 10
                PARAMS['PATIENCE'] = 15

            elif DATASET_SHORTNAME == 'sgm2017high':
                pass

            if m_config_len == 5:
                PARAMS['MLE_PATIENCE'] = MLE_PATIENCE
                PARAMS['PATIENCE'] = MR_PATIENCE


            for TRANSDUCER, SUBSTITUTION, MODELNAME in MODEL_CONFIGS:

                for (ALIGNCALL, ALIGNNAME), SEED in ((a, s) for a in ALIGN_CONFIGS for s in SEEDS):
                    # RESULTS
                    if PARAMS['MODE'] != 'mle':
                        CURRENT_RESULTS_DIR= RESULTS_DIR
                    else:
                        CURRENT_RESULTS_DIR= MLE_RESULTS_DIR
                    RESULTSPATH = os.path.join(RESULTSDIR,
                                               CURRENT_RESULTS_DIR.format(MODELNAME=MODELNAME, ALIGNNAME=ALIGNNAME,
                                                                  DATASET=DATASET_SHORTNAME,
                                                                  RNNEXTRA = '',
                                                                  MODEEXTRA = MODEEXTRA,
                                                                  OPTEXTRA = optextra,
                                                                  INPEXTRA=inputextra,
                                                                  **PARAMS),
                                               '{}'.format(FOLD_ID),
                                               's_{}'.format(SEED))

                    if PARAMS['MODE'] != 'mle':


                        RELOADPATH = os.path.join(RESULTSDIR,
                                                  MLE_RESULTS_DIR.format(MODELNAME=MODELNAME, ALIGNNAME=ALIGNNAME,
                                                                     DATASET=DATASET_SHORTNAME,
                                                                     RNNEXTRA = '',
                                                                     MODEEXTRA = '',
                                                                     OPTEXTRA = optextra,
                                                                     INPEXTRA=inputextra,
                                                                     **PARAMS),
                                                  '{}'.format(FOLD_ID),
                                                  's_{}'.format(SEED))

                    else:
                        RELOADPATH = ''

                    if 'x' in options.mode and not os.path.exists(RESULTSPATH):
                        os.makedirs(RESULTSPATH)

                    OUTPATH = os.path.join(RESULTSPATH, 'output.stdout')
                    DONEPATH = os.path.join(RESULTSPATH, 'f.model.done')
                    call = CALL.format(SEED=SEED,
                                       TRANSDUCER=TRANSDUCER,
                                       SUBSTITUTION=SUBSTITUTION,
                                       ALIGN=ALIGNCALL,
                                       TRAINPATH=TRAINPATH,
                                       DEVPATH=DEVPATH,
                                       TESTPATH=TESTPATH,
                                       RESULTSPATH=RESULTSPATH,
                                       OUTPATH=OUTPATH,
                                       DONEPATH=DONEPATH,
                                       RELOADPATH=RELOADPATH,
                                       CMDPREFIX=CMDPREFIX,
                                       **PARAMS)
                    if os.path.exists(DONEPATH):
                        print >> sys.stderr, '# MODEL EXISTS: NOT REGENERATING', DONEPATH
                    elif 'm' in options.mode: # makefile mode
                        calls.append((call,DONEPATH,RELOADPATH))
                    else:
                        call = 'mkdir -p {} && {}'.format(RESULTSPATH, call)
                        calls.append(call)


    calls = sorted(set(calls))
    if 'm' in options.mode:
        launch_make(calls, options)
        return
    if not 'r' in options.mode:
        # mle must be called first
        mle_calls = [call for call in calls if "--mode=mle " in call]
        print >> sys.stderr, '# Starting all mle calls',len(mle_calls)
        launch_parallel(mle_calls)
        print >> sys.stderr, '# Finished all mle calls',len(mle_calls)

    # all others
    non_mle_calls = [call for call in calls if  not "--mode=mle " in call]
    print >> sys.stderr, '# Starting all non-mle calls',len(non_mle_calls)

    if not 'e' in options.mode:
        launch_parallel(non_mle_calls)
        print >> sys.stderr, '# Finished all non-mle calls',len(non_mle_calls)
    print >> sys.stderr, '# Finished all calls',len(calls)
def main():
    """
    Invoke this module as a script
    """
    global options, args, SEEDS
    parser = OptionParser(
        usage = '%prog [OPTIONS] CONFIGFILE1 [CONFIGFILE2...]',
        version='%prog 0.99', #
        description='Launch experiments given one or more config files overwriting globals from left to right',
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
    parser.add_option('-s', '--seeds',
                      action='store', dest='seeds', default="1,2,3,4,5",
                      help='comma-separated dynet seed numbers (%default)')
    parser.add_option('-m', '--mode',
                      action='store', dest='mode', default='n',
                      help= ('execution mode n=only print, x=execute e=mle models only r=mrt/rl models only m=generate Makefile output (default %default)'
                            ' Example: -m xe executes all mle models ; -m r prints the commands for the mrt/rl models'))
    parser.add_option('-e', '--epochs',
                      action='store', dest='epochs', default=0, type = int,
                      help= 'force the epochs to be set to the specified value (for testing only); overwrite the specified configurations!')
    parser.add_option('-j', '--parallel',
                      action='store', dest='parallel', default=5,type=int,
                      help='number of simultaneous processes run in parallel (%default)')


    (options, args) = parser.parse_args()
    if options.debug:
        print >> sys.stderr, "options=",options
    if len(args) < 1:
        print >> sys.stderr, '#ERROR: Configuration file needed'
        parser.print_help()
        exit(1)
    for arg in args:
        if options.debug: print >>sys.stderr,'Reading config file file',arg
        try:
            execfile(arg,globals())
        except Exception, e:
            print arg
            print globals()
            raise e
    if options.seeds:
        SEEDS = [int(n) for n in options.seeds.split(',')]

    process(options,args)


if __name__ == '__main__':
    main()
