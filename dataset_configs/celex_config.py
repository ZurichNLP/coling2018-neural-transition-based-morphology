

SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '../../paper2018/results/newcelex' #'/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/newcelex'


DATASET_SHORTNAME = 'celex'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/celex/data/'

DATASET_FOLDS = [
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_0.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_2.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_3.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_4.']




DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'dev.txt'
DATASET_TESTFILE_NAME = 'test.txt'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [
        ('hard', '', 'hard'),
        ('haem', '', 'haem'),
                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [
    ('--align-smart', 'crp'),
    ('--align-cls', 'cls')
                ]



mle_patience = 10
# Lets put patiencesfor mle as 4th and the rl patience as  5th
# # the last element in a tuple is the string that goes into the results directory name
# 1. MODE --mode={MODE}
# 2. MODE_OPTIONS go into the run_transducer.py call as options
# 3. MODEEXTRA goes into the directory name x-...m{MODE}{MODEEXTRA}...-x
# 4. MLE Patience
# 5. Secondary Training Patience (reinforcement patience)
MODE_CONFIGS = [
                ('mle', '', ''),
                ('mrt', '--alpha=1', '1', mle_patience, 20),
                 ]




MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 50,
    #PRETRAIN = '--pretrain-epochs=15',
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4'
)

