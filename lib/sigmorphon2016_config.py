SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/sigmorphon2016'  # '../results/sigmorphon2016'


DATASET_SHORTNAME = 'sigmorphon2016'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/sigmorphon2016/data/'

DATASET_FOLDS = [
'arabic-task1',
#'finnish-task1',
#'georgian-task1',
#'german-task1',
#'hungarian-task1',
'maltese-task1',
'navajo-task1',
#'russian-task1',
#'spanish-task1',
'turkish-task1'
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

MODE_CONFIGS = [('mle', '', ''),
                # ('rl', 'DUMMY', ''),
                # ('mrt', 'DUMMY', '')
               ]




MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 5,
    EPOCHS   = 30,
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '--avm-feat-format',   #--pos-emb'
    # DECODING
    BEAMWIDTHS   = '4'
)
