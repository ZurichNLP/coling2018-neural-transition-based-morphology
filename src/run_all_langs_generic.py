"""Runs a script on selected languages in parallel, supports ensemble training

Usage:
  run_all_langs_generic.py [--dynet-seed=SEED] [--dynet-mem=MEM] [--input=INPUT] [--feat-input=FEAT] [--hidden=HIDDEN] [--epochs=EPOCHS]
  [--layers=LAYERS] [--optimization=OPTIMIZATION] [--pool=POOL] [--langs=LANGS] [--script=SCRIPT] [--prefix=PREFIX]
  [--augment] [--merged] [--task=TASK] [--ensemble=ENSEMBLE] [--eval_dev | --eval_all] [--regimes=REGIME]
  [--src_path=SRC_PATH] [--results_path=RESULTS_PATH] [--data_path=DATA_PATH] [--params=PARAMS]

Options:
  -h --help                     show this help message and exit
  --dynet-seed=SEED             dynet seed
  --dynet-mem=MEM               allocates MEM bytes for dynet
  --input=INPUT                 input vector dimensions
  --feat-input=FEAT             feature input vector dimension
  --hidden=HIDDEN               hidden layer dimensions
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM
  --pool=POOL                   amount of processes in pool
  --langs=LANGS                 languages separated by comma
  --script=SCRIPT               the training script to run
  --prefix=PREFIX               the output files prefix
  --augment                     whether to perform data augmentation
  --merged                      whether to train on train+dev merged
  --task=TASK                   the current task to train
  --ensemble=ENSEMBLE           the amount of ensemble models to train, 1 if not mentioned
  --eval_dev                    run only evaluation without training on the DEV set
  --eval_all                    run only evaluation without training on both the DEV and TEST sets
  --regimes=REGIME              low,medium,high
  --src_path=SRC_PATH           source files directory path
  --results_path=RESULTS_PATH   results file to be written
  --data_path=DATA_PATH         sigmorphon root containing data (but not src/evalm.py)
  --params=PARAMS               parameters to pass on to the script, e.g. --align_dumb for hard_attention.py, --nbest=12 for printing nbest list of 12, --beam for using beam serach instead of greedy search

"""

import os
import time
import datetime
import docopt
from multiprocessing import Pool

# load default values for paths, NN dimensions, some training hyperparams
from defaults import (SRC_PATH, RESULTS_PATH, DATA_PATH,
                      INPUT_DIM, FEAT_INPUT_DIM, HIDDEN_DIM, LAYERS,
                      EPOCHS, OPTIMIZATION, DYNET_MEM, DYNET_SEED,DYNET_SEED_STEP)

LANGS = ['albanian', 'arabic', 'armenian', 'bulgarian', 'catalan', 'czech', 'danish', 'dutch', 'english',
         'faroese', 'finnish', 'french', 'georgian', 'german', 'hebrew', 'hindi', 'hungarian', 'icelandic',
         'italian', 'latvian', 'lower-sorbian', 'macedonian', 'navajo', 'northern-sami', 'norwegian-nynorsk',
         'persian', 'polish', 'portuguese', 'quechua', 'russian', 'scottish-gaelic', 'serbo-croatian', 'slovak',
         'slovene', 'spanish', 'swedish', 'turkish', 'ukrainian', 'urdu', 'welsh']
REGIME = ['low', 'medium'] #['low', 'medium', 'high']
POOL = 4

def main(src_dir, results_dir, sigmorphon_root_dir, input_dim, hidden_dim, epochs, layers,
         optimization, feat_input_dim, pool_size, langs, script, prefix, task, augment, merged, ensemble,
         eval_dev, eval_all, regimes, rest_params):
    parallelize_training = True
    params = []
    print 'now training langs: ' + str(langs)
    print 'in regimes: ' + str(regimes)
    for regime in regimes:
        for lang in langs:

            # check if an ensemble was requested
            ensemble_paths = []
            if ensemble > 1:
                for e in xrange(ensemble):

                    # create prefix for ensemble
                    ens_prefix = prefix + '_ens_{}'.format(e)

                    if not (eval_dev or eval_all):#eval_only:
                        # should train ensemble model: add params set for parallel model training execution
                        params.append([DYNET_SEED + DYNET_SEED_STEP*e,DYNET_MEM, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization,
                                       results_dir, sigmorphon_root_dir, src_dir, script, ens_prefix, task, augment,
                                       merged, '', eval_dev, eval_all, regime, rest_params])
                    else:
                        # eval ensemble by generating a list of existing ensemble model paths and then passing it to
                        # the script as a single concatenated string parameter
                        ensemble_paths.append('{}/{}/{}_{}'.format(results_dir, ens_prefix, lang, regime))
                if (eval_dev or eval_all):#eval_only:
                    concat_paths = ','.join(ensemble_paths)
                    params.append([DYNET_SEED, DYNET_MEM, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization,
                                   results_dir, sigmorphon_root_dir, src_dir, script, prefix, task, augment,
                                   merged, concat_paths, eval_dev, eval_all, regime, rest_params])
            else:
                # train a single model
                params.append([DYNET_SEED, DYNET_MEM, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization,
                               results_dir, sigmorphon_root_dir, src_dir, script, prefix, task, augment, merged,
                               ensemble_paths, eval_dev, eval_all, regime, rest_params])

    # train models for each lang/ensemble in parallel or in loop
    if parallelize_training:
        pool = Pool(int(pool_size) * ensemble, maxtasksperchild=1)
        print 'now training {} langs {} regimes in parallel, {} ensemble models per lang'.format(len(langs), len(regimes), ensemble)
        pool.map(train_language_wrapper, params)
    else:
        print 'now training {0} langs {} regimes in loop, {} ensemble models per lang'.format(len(langs), len(regimes), ensemble)
        for p in params:
            train_language(*p)

    print 'finished training all models'


def train_language_wrapper(params):
    train_language(*params)


def train_language(dynet_seed, dynet_mem, epochs, feat_input_dim, hidden_dim, input_dim, lang, layers, optimization, results_dir,
                   sigmorphon_root_dir, src_dir, script, prefix, task, augment, merged, ensemble_paths, eval_dev, eval_all,
                   regime, rest_params):

    # augment is really implemented anywhere
    if augment:
        augment_str = '--augment'
    else:
        augment_str = ''

    if eval_dev or eval_all:
        eval_str = '--eval'
    else:
        eval_str = ''

    if len(ensemble_paths) > 0:
        ensemble_str = '--ensemble={}'.format(ensemble_paths)
    else:
        ensemble_str = ''

    start = time.time()

    # default sigmorphon 2017 format
    train_path = '{}/task{}/{}-train-{}'.format(sigmorphon_root_dir, task, lang, regime)
    if task == 2:
        dev_path = '{}/task{}/{}-uncovered-dev'.format(sigmorphon_root_dir, task, lang)
        test_path = '{}/task{}/{}-covered-dev'.format(sigmorphon_root_dir, task, lang)
    else:
        dev_path = '{}/task{}/{}-dev'.format(sigmorphon_root_dir, task, lang)
        #we don't know how test files will be named.
        test_path = ''

    # same for all
    results_dir='{}/{}'.format(results_dir, prefix)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = '{}/{}_{}'.format(results_dir, lang, regime)

    if merged:
        train_path = '../data/sigmorphon_train_dev_merged/{}-task{}-merged'.format(lang, task)

    # test set optional
    test_path = '--test_path=' + test_path if eval_all else ''
    # list of optional params to pass on to script
    rest = ' '.join(['--' + p for p in rest_params]) if rest_params else ''

    # train on train, evaluate on dev for early stopping, finally eval on train
    command = 'python {0} --dynet-seed {1} --dynet-mem {2} --input={3} --hidden={4} --feat-input={5} --epochs={6} --layers={7} \
        --optimization={8} {9} {10} {11} {12} {13} {14} {15} {16}\
        '.format(script, dynet_seed, dynet_mem, input_dim, hidden_dim, feat_input_dim, epochs, layers, optimization,
                 eval_str, augment_str, ensemble_str, train_path, dev_path, results_path, test_path, rest)
    print '\n' + command +'\n'
    os.system(command)

    end = time.time()
    print 'finished {} in regime in {}'.format(lang, regime, str(end - start))


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['--src_path']:
        src_dir_param = arguments['SRC_PATH']
    else:
        src_dir_param = SRC_PATH
    if arguments['--results_path']:
        results_dir_param = arguments['--results_path']
    else:
        results_dir_param = RESULTS_PATH
    if arguments['--data_path']:
        sigmorphon_root_dir_param = arguments['--data_path']
    else:
        sigmorphon_root_dir_param = DATA_PATH
    if arguments['--input']:
        input_dim_param = int(arguments['--input'])
    else:
        input_dim_param = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim_param = int(arguments['--hidden'])
    else:
        hidden_dim_param = HIDDEN_DIM
    if arguments['--feat-input']:
        feat_input_dim_param = int(arguments['--feat-input'])
    else:
        feat_input_dim_param = FEAT_INPUT_DIM
    if arguments['--epochs']:
        epochs_param = int(arguments['--epochs'])
    else:
        epochs_param = EPOCHS
    if arguments['--layers']:
        layers_param = int(arguments['--layers'])
    else:
        layers_param = LAYERS
    if arguments['--optimization']:
        optimization_param = arguments['--optimization']
    else:
        optimization_param = OPTIMIZATION
    if arguments['--pool']:
        pool_size_param = arguments['--pool']
    else:
        pool_size_param = POOL
    if arguments['--langs']:
        langs_param = [l.strip() for l in arguments['--langs'].split(',')]
    else:
        langs_param = LANGS
    if arguments['--script']:
        script_param = arguments['--script']
    else:
        print 'script is mandatory'
        raise ValueError
    if arguments['--prefix']:
        prefix_param = arguments['--prefix']
    else:
        print 'prefix is mandatory'
        raise ValueError
    if arguments['--task']:
        task_param = arguments['--task']
    else:
        task_param = '1'
    if arguments['--augment']:
        augment_param = True
    else:
        augment_param = False
    if arguments['--merged']:
        merged_param = True
    else:
        merged_param = False
    if arguments['--ensemble']:
        ensemble_param = int(arguments['--ensemble'])
    else:
        ensemble_param = 1
    if arguments['--eval_dev']:
        eval_dev_param = True
    else:
        eval_dev_param = False
    if arguments['--eval_all']:
        eval_all_param = True
    else:
        eval_all_param = False
    if arguments['--regimes']:
        regime_param = [l.strip() for l in arguments['--regimes'].split(',')]
    else:
        regime_param = REGIME
    if arguments['--params']:
        # key-value pairs (xyz=abc) or Boolean (align_smart)
        rest_params = [l.strip() for l in arguments['--params'].split(',')]
    else:
        rest_params = None

    print arguments

    main(src_dir_param, results_dir_param, sigmorphon_root_dir_param, input_dim_param, hidden_dim_param, epochs_param,
         layers_param, optimization_param, feat_input_dim_param, pool_size_param, langs_param, script_param,
         prefix_param, task_param, augment_param, merged_param, ensemble_param, eval_dev_param, eval_all_param, regime_param, rest_params)
