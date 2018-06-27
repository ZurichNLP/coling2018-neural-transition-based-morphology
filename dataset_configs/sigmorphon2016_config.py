import models

SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '../../paper2018/results/sigmorphon2016'
#RESULTSDIR = '/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/sigmorphon2016'


DATASET_SHORTNAME = 'sigmorphon2016'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/sigmorphon2016/data/'

DATASET_FOLDS = [
#'arabic-task1',
#'finnish-task1',
#'georgian-task1',
#'german-task1',
#'hungarian-task1',
'maltese-task1',
#'navajo-task1',
#'russian-task1',
#'spanish-task1',
#'turkish-task1'
]



DATASET_TRAINFILE_NAME = '-train'
DATASET_DEVFILE_NAME = '-dev'
DATASET_TESTFILE_NAME = '-test'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),
                 #('hacm', '', 'hacm'),
                 #('haem', '--substitution', 'haem_sub'),
                 #('hacm', '--copy-as-substitution', 'hacm_sub')
                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [('--align-smart', 'crp'),
                 #('--align-cls', 'cls')
                ]


# the last element in a tuple is the string that goes into the results directory name

MODE_CONFIGS = [#('mle', '', ''),
                # ('rl', 'DUMMY', ''),
                # ('mrt', 'DUMMY', ''),
                #('ss', '', '', 5, 5),
                ('mrt', '--alpha=1', '1', 5, 5),
                #models.dynamic_oracles(5, 5),
                #models.lols(5, 5),
                #('ss', '--ss-optimal-oracle --ss-loss=nll --ss-beta=1 --ss-global-rollout --ss-pretrain-epochs=10', 'di', 5, 5),
                #('ss', '--ss-optimal-oracle --ss-loss=nll --ss-beta=0.5 --ss-global-rollout --ss-pretrain-epochs=10', 'roi', 5, 5),
                #models.lols_margin(5, 5),
                #('ss', '--ss-optimal-oracle --ss-loss=softmax-margin --ss-beta=0.5 --ss-global-rollout --ss-pretrain-epochs=10', 'roim', 5, 5),
               ]



MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 5,
    EPOCHS   = 30,
    #PRETRAIN = '--pretrain-epochs=5',
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '--pos-emb',  #  '--avm-feat-format',   
    # DECODING
    BEAMWIDTHS   = '4'
)
