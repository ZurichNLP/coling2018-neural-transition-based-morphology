SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '../results/nck15'


DATASET_SHORTNAME = 'nck15'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/nck15/data/'


# base_forms_de_noun_dev.txt      base_forms_es_verb_train.txt
DATASET_FOLDS = [
    'dutch_',
    'french_']




DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'dev.txt'
DATASET_TESTFILE_NAME = 'test.txt'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),
                 ('hacm', '', 'hacm'),
                 ('haem', '--substitution', 'haem_sub'),
                 ('hacm', '--copy-as-substitution', 'hacm_sub')]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [('--align-smart', 'crp'), ('--align-cls', 'cls')]


# the last element in a tuple is the string that goes into the results directory name

MODE_CONFIGS = [('mle', '', 'mle'),
                 ('rl', 'DUMMY', 'rl'),
                 ('mrt', 'DUMMY', 'mrt')]




MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 5,
    EPOCHS   = 50, # dynamically computed in launch according to the training file size
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
