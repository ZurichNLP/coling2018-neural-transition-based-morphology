SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '../results/segmentation'


DATASET_SHORTNAME = 'segmentation'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/canonical-segmentation/data/'    # Segmentation was used for the original dataset and could not be easily exchanges as submodule


#
DATASET_FOLDS = [
'english_0_',
'english_1_',
'english_2_',
'english_3_',
'english_4_',
'german_0_',
'german_1_',
'german_2_',
'german_3_',
'german_4_',
'indonesian_0_',
'indonesian_1_',
'indonesian_2_',
'indonesian_3_',
'indonesian_4_'
]


DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'dev.txt'
DATASET_TESTFILE_NAME = 'test.txt'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),
                 ('hacm', '', 'hacm'),
                 #('haem', '--substitution', 'haem_sub'),
                 #('hacm', '--copy-as-substitution', 'hacm_sub')
                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [('--align-smart', 'crp'),
                 #('--align-cls', 'cls')
                ]


# the last element in a tuple is the string that goes into the results directory name

MODE_CONFIGS = [('mle', '', 'mle'),
                 ('rl', 'DUMMY', 'rl'),
                 ('mrt', 'DUMMY', 'mrt')]




MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '--no-feat-format',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
