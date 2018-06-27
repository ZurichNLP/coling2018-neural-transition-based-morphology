# celex tasks:
# rP 2PKE 2PIE 13SIA

SEEDS = [1, 2, 3, 4, 5]

RESULTSDIR = '/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/'

DATASET_PATH = '../data/celex-by-task/'

DATASET_FOLDS = ['0/', '1/', '2/', '3/', '4/']

DATASET_TRAINFILE_NAME = 'train'
DATASET_DEVFILE_NAME = 'dev'
DATASET_TESTFILE_NAME = 'test'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),
                 ('hacm', '', 'hacm'),
                 ('haem', '--substitution', 'haem_sub'),
                 ('hacm', '--copy-as-substitution', 'hacm_sub')
                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [('--align-smart', 'crp'),
                 ('--align-cls', 'cls')
                ]


# the last element in a tuple is the string that goes into the results directory name

MODE_CONFIGS = [('mle', '', 'mle'),
                 #('rl', 'DUMMY', 'rl'),
                 ('mrt', 'DUMMY', 'mrt')
               ]


MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '--no-feat-format',  # and no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)